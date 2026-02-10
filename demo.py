import streamlit as st
from datamint import Api
import numpy as np
from datamint.entities.annotations import VolumeSegmentation
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from operator import itemgetter
import math

st.set_page_config(layout="wide", page_title="TMJ Segmentation Demo")
st.title("🦷 TMJ Segmentation Demo")

PROJECT_NAME = "TMJ Test"
api = Api()
SPLIT = "test"

# ============================================================
# Geometry (https://github.com/aswahd/TMJ-Disk-Dislocation-Classification/blob/main/inference/tmj_mask_calculations_from_pred.py)
# ============================================================
def get_mean_point(list_of_tuples):
    """Mean (x,y) rounded to int. """
    
    if len(list_of_tuples) == 0:
        return None
    
    x_mean = round(sum(p[0] for p in list_of_tuples) / len(list_of_tuples))
    y_mean = round(sum(p[1] for p in list_of_tuples) / len(list_of_tuples))
    
    return (x_mean, y_mean)

def get_extrema_point(list_of_tuples, extrema_wanted):
    """
    Original convention:
      - 'max' returns the HIGHEST point -> min y (because y grows downward)
      - 'min' returns the LOWEST point  -> max y
    """
    if len(list_of_tuples) == 0:
        return None
    if extrema_wanted == 'max':
        return min(list_of_tuples, key=itemgetter(1))
    elif extrema_wanted == 'min':
        return max(list_of_tuples, key=itemgetter(1))
    raise ValueError("extrema_wanted must be 'max' or 'min'")

def get_extrema_point_x(list_of_tuples, extrema_wanted):
    """Min/max in x."""
    if len(list_of_tuples) == 0:
        return None
    if extrema_wanted == 'min':
        return min(list_of_tuples, key=itemgetter(0))
    elif extrema_wanted == 'max':
        return max(list_of_tuples, key=itemgetter(0))
    raise ValueError("extrema_wanted must be 'min' or 'max'")

def b_line(x0, y0, x1, y1):
    """Bresenham line algorithm. Returns list of (x,y)."""
    points_in_line = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points_in_line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points_in_line.append((x, y))
    return points_in_line

def get_distances_between_two_points(point1, point2):
    """Return (x_dist, y_dist, euclid_dist)."""
    x_distance = abs(point1[0] - point2[0])
    y_distance = abs(point1[1] - point2[1])
    e_distance = math.sqrt((x_distance ** 2) + (y_distance ** 2))
    
    return (x_distance, y_distance, e_distance)

def getShortestDistance(s1, s2):
    """
    Find closest pair between two point sets.
    Input: lists/arrays of (x,y)
    Output: (closest_from_s1, closest_from_s2) as tuples.
    """
    A = np.asarray(s1, dtype=np.float32)
    B = np.asarray(s2, dtype=np.float32)
    sd = cdist(A, B)
    x, y = np.where(sd == np.min(sd))
    p1 = tuple(A[x[0]].astype(int))
    p2 = tuple(B[y[0]].astype(int))
    return p1, p2

def find_midpoint(alist):
    """Midpoint between first and last."""
    x_mid = round((alist[0][0] + alist[-1][0]) / 2)
    y_mid = round((alist[0][1] + alist[-1][1]) / 2)
    
    return (x_mid, y_mid)

def make_disc_line(c_disc):
    """
    Returns: (best_fit_points, disc_length, top_right_best, bot_left_best)
    """
    c_disc = list(set(c_disc))
    
    if len(c_disc) == 0:
        return None, 0.0, None, None

    top_right_x = get_extrema_point_x(c_disc, "max")[0]
    top_right_y = get_extrema_point(c_disc, "max")[1]
    bot_left_x = get_extrema_point_x(c_disc, "min")[0]
    bot_left_y = get_extrema_point(c_disc, "min")[1]

    top_right_point = (top_right_x, top_right_y)
    bot_left_point = (bot_left_x, bot_left_y)

    xs = np.array([p[0] for p in c_disc], dtype=np.float32)
    ys = np.array([p[1] for p in c_disc], dtype=np.float32)

    a, b = np.polyfit(xs, ys, 1)

    best_fit_points = []
    for x_val in xs:
        y_val = round(a * x_val + b)
        best_fit_points.append((int(round(x_val)), int(y_val)))

    top_right_best = get_extrema_point(best_fit_points, 'max')
    bot_left_best = get_extrema_point(best_fit_points, 'min')

    disc_length = get_distances_between_two_points(top_right_best, bot_left_best)[2]

    return best_fit_points, disc_length, top_right_best, bot_left_best

def make_emcon_line(c_eminence, c_condyle, left_low_disc_point):
    """
    Port of original make_emcon_line:
    - remove duplicates
    - filter eminence points to the right of left_low_disc_point.x
    - pick highest eminence point (min y)
    - keep eminence points up to highest_eminence_point.x
    - closest pair between condyle and cleaned eminence
    - generate bresenham line between those two points
    Returns: (emcon_line_points, [x_len, y_len, e_len], emcon1, emcon2)
    """
    c_eminence = list(set(c_eminence))
    c_condyle = list(set(c_condyle))

    updated_c_eminence = [p for p in c_eminence if p[0] >= left_low_disc_point[0]]
    if len(updated_c_eminence) == 0 or len(c_condyle) == 0:
        return None, None, None, None

    highest_eminence_point = get_extrema_point(updated_c_eminence, 'max')

    cleaned_c_eminence = [p for p in updated_c_eminence if p[0] <= highest_eminence_point[0]]
    if len(cleaned_c_eminence) == 0:
        return None, None, None, None

    emcon1, emcon2 = getShortestDistance(c_condyle, cleaned_c_eminence)
    emcon_line_points = b_line(emcon1[0], emcon1[1], emcon2[0], emcon2[1])

    emcon_x_len, emcon_y_len, emcon_e_len = get_distances_between_two_points(emcon1, emcon2)
    return emcon_line_points, [emcon_x_len, emcon_y_len, emcon_e_len], emcon1, emcon2

def calculate_ratios(disc_mask_display, emcon_line_points):
    """
    Uses determinant sign relative to the emcon line endpoints (highest vs lowest).
    Returns (left_percent, right_percent) = (d<0 %, d>0 %) as in original code's output order.
    """
    disc_pts_yx = np.argwhere(disc_mask_display > 0)
    if len(disc_pts_yx) == 0:
        return 0.0, 0.0

    # convert disc points to (x,y)
    c_disc = [(int(p[1]), int(p[0])) for p in disc_pts_yx]
    c_disc = list(set(c_disc))

    ep1 = get_extrema_point(emcon_line_points, 'max') 
    ep2 = get_extrema_point(emcon_line_points, 'min')  

    dlz = []  # d < 0
    dgz = []  # d > 0
    # d = (x-x1)(y2-y1) - (y-y1)(x2-x1)
    for (x, y) in c_disc:
        d = (x - ep1[0]) * (ep2[1] - ep1[1]) - (y - ep1[1]) * (ep2[0] - ep1[0])
        if d > 0:
            dgz.append((x, y))
        elif d < 0:
            dlz.append((x, y))
        else:
            pass

    total = len(dlz) + len(dgz)
    if total == 0:
        return 0.0, 0.0

    left_percent = (len(dlz) / total) * 100.0
    right_percent = (len(dgz) / total) * 100.0
    return left_percent, right_percent

@st.cache_resource
def get_api_and_resources():
    """ 
    Gets the Datamint API and a mapping of resources for the specified project and split.
    """
    
    api = Api()
    resources = list(api.resources.get_list(project_name=PROJECT_NAME, tags=f"split:{SPLIT}"))
    res_dict = {r.filename: r for r in resources}
    return api, res_dict

api, resource_map = get_api_and_resources()

st.sidebar.header("Settings")
selected_filename = st.sidebar.selectbox("Select a Resource", list(resource_map.keys()))
selected_res = resource_map[selected_filename]

@st.cache_data
def load_nifti_obj(res_id):
    """Loads NIfTI data from a Datamint resource by its ID."""
    res = api.resources.get_by_id(res_id)
    data = res.fetch_file_data(auto_convert=True)
    
    if hasattr(data, "get_fdata"):
        return data.get_fdata()
    
    return np.array(data)

if "combined_mask" not in st.session_state:
    st.session_state.combined_mask = None
if "pred_voxels" not in st.session_state:
    st.session_state.pred_voxels = {}
if "last_res" not in st.session_state:
    st.session_state.last_res = None

if st.session_state.last_res != selected_res.id:
    st.session_state.combined_mask = None
    st.session_state.last_res = selected_res.id

nifti_obj = load_nifti_obj(selected_res.id)

if hasattr(nifti_obj, "get_fdata"):
    vol = nifti_obj.get_fdata()
    spacing = nifti_obj.header.get_zooms()[:2]
else:
    vol = np.array(nifti_obj)
    spacing = (1.0, 1.0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Volume")
    slice_idx = st.slider("Select Slice Index", 0, vol.shape[2] - 1, vol.shape[2] // 2)
    slice_img = vol[:, :, slice_idx].astype(np.float32)

    image = np.flip(np.rot90(slice_img, k=3), axis=1)
    vmin, vmax = np.percentile(image, 1), np.percentile(image, 99)
    image = np.clip((image - vmin) / (vmax - vmin + 1e-8), 0, 1)
    st.image(image, caption=f"Slice {slice_idx}", width=500, clamp=True)

with col2:
    st.subheader("Segmentation Result")

    if st.button("🚀 Run Segmentation"):
        with st.spinner("Running inference..."):
            start = time.time()
            job = api.inference.predict_image(
                model_name='TMJ Test_adapted',
                model_alias='latest',
                resource_id=selected_res.id
            )
            job.wait()

            h, w, z = vol.shape
            temp_mask = np.zeros((h, w, z), dtype=np.uint8)
            temp_voxels = {}
            CLASS_MAP = {"Disc": 1, "Condyle": 2, "Eminence": 3}

            for pred in job.result_data['predictions'][0]:
                ann = VolumeSegmentation(**pred)
                class_name = ann.name if ann.name else list(ann.class_map.values())[0]
                temp_voxels[class_name] = int((ann.segmentation_data > 0).sum())
                if class_name in CLASS_MAP:
                    temp_mask[ann.segmentation_data > 0] = CLASS_MAP[class_name]

            st.session_state.combined_mask = temp_mask
            st.session_state.pred_voxels = temp_voxels
            st.session_state.inf_time = time.time() - start

    if st.session_state.combined_mask is not None:
        mask_slice = st.session_state.combined_mask[:, :, slice_idx]
        mask_display = np.flip(np.rot90(mask_slice, k=3), axis=1)

        overlay_rgb = np.stack([image] * 3, axis=-1)
        COLORS_RGB = {1: [0, 0, 1], 2: [0, 1, 0], 3: [1, 1, 0]}

        alpha = 0.5
        for val, color in COLORS_RGB.items():
            pixels = (mask_display == val)
            if pixels.any():
                overlay_rgb[pixels] = (1 - alpha) * overlay_rgb[pixels] + alpha * np.array(color)

        st.image(overlay_rgb, caption=f"Overlay - Slice {slice_idx}", width=500)

        st.sidebar.subheader("Voxel Stats")
        for name, count in st.session_state.pred_voxels.items():
            st.sidebar.write(f"{name}: {count} voxels")
        st.sidebar.write(f"It took {st.session_state.inf_time:.1f} seconds")

        if st.button("📏 Calculate Dislocation"):
            m_disc = (mask_display == 1)
            m_cond = (mask_display == 2)
            m_emin = (mask_display == 3)

            if m_disc.any() and m_cond.any() and m_emin.any():
                c_disc = [(int(p[1]), int(p[0])) for p in np.argwhere(m_disc)]
                c_cond = [(int(p[1]), int(p[0])) for p in np.argwhere(m_cond)]
                c_emin = [(int(p[1]), int(p[0])) for p in np.argwhere(m_emin)]

                c_disc = list(set(c_disc))
                c_cond = list(set(c_cond))
                c_emin = list(set(c_emin))

                disc_centroid = get_mean_point(c_disc)

                _, _, top_right_best, bot_left_best = make_disc_line(c_disc)

                if disc_centroid is None or bot_left_best is None:
                    st.error("Cannot calculate: disc geometry failed (empty or degenerate).")
                else:
                    emcon_line_points, _, emcon1, emcon2 = make_emcon_line(c_emin, c_cond, bot_left_best)

                    if emcon_line_points is None:
                        st.error("Cannot calculate: EMCON line failed (insufficient points after filtering).")
                    else:
                        emcon_line_centroid = find_midpoint(emcon_line_points)
                        dist_px = get_distances_between_two_points(disc_centroid, emcon_line_centroid)[2]
                        dist_mm = dist_px * float(spacing[0])

                        pa, pp = calculate_ratios(m_disc, emcon_line_points)

                        st.sidebar.markdown("---")
                        st.sidebar.subheader("Dislocation Metrics")
                        st.sidebar.warning(f"Distance: {dist_mm:.2f} mm")
                        
                        if dist_mm < 10.0:
                            st.sidebar.success("Normal")
                        else:
                            st.sidebar.error("Anterior Dislocated (ADD)")
                        
                        st.sidebar.write(f"Anterior (PA): {pa:.2f}%")
                        st.sidebar.write(f"Posterior (PP): {pp:.2f}%")

                        fig, ax = plt.subplots()
                        ax.imshow(overlay_rgb)
                        ax.scatter(disc_centroid[0], disc_centroid[1], color='red', s=40, label='Q (Disc Center)')
                        ax.scatter(emcon_line_centroid[0], emcon_line_centroid[1], color='cyan', s=40, label='P (EMCON Midpoint)')

                        xs = [p[0] for p in emcon_line_points]
                        ys = [p[1] for p in emcon_line_points]
                        ax.plot(xs, ys, 'm--', label='EMCON Line')

                        if emcon1 is not None and emcon2 is not None:
                            ax.scatter(emcon1[0], emcon1[1], color='lime', s=20, label='Closest Condyle pt')
                            ax.scatter(emcon2[0], emcon2[1], color='yellow', s=20, label='Closest Eminence pt')

                        ax.legend(fontsize='x-small')
                        ax.axis('off')
                        st.pyplot(fig)
            else:
                st.error("Cannot calculate: Missing structures in this slice.")
