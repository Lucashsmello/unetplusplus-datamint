import streamlit as st
from datamint import Api
import numpy as np
from datamint.entities.annotations import VolumeSegmentation
import time

st.set_page_config(layout="wide", page_title="TMJ Segmentation Demo")
st.title("🦷 TMJ Segmentation Demo")

PROJECT_NAME = "TMJ Test"
api = Api()
proj = api.projects.get_by_name(PROJECT_NAME)
SPLIT = "test"


@st.cache_resource 
def get_api_and_resources():
    """ 
    Gets the Datamint API and a mapping of resources for the specified project and split.
    """
    
    api = Api()
    resources = list(api.resources.get_list(
        project_name=PROJECT_NAME,
        tags=f"split:{SPLIT}"
    ))
    
    # Create a mapping from filename to resource
    res_dict = {r.filename: r for r in resources}
    return api, res_dict

api, resource_map = get_api_and_resources()

st.sidebar.header("Settings")
if not resource_map:
    st.sidebar.warning(f"No resources found for the project {PROJECT_NAME} and split {SPLIT}.")
    st.stop()

selected_filename = st.sidebar.selectbox("Select a Resource", list(resource_map.keys()))
selected_res = resource_map[selected_filename]

col1, col2 = st.columns(2)

@st.cache_data
def load_nifti_data(res_id):
    """ Loads NIfTI data from a Datamint resource by its ID."""
    res = api.resources.get_by_id(res_id)
    data = res.fetch_file_data(auto_convert=True)
    
    if hasattr(data, "get_fdata"):
        return data.get_fdata()
    
    return np.array(data)

with col1:
    # Display the input volume slice
    st.subheader("Input Volume")
    vol = load_nifti_data(selected_res.id)
    
    # Select a slice to display 
    slice_idx = st.slider("Select Slice Index", 0, vol.shape[2] - 1, vol.shape[2] // 2)
    
    # Display the selected slice 
    slice_img = vol[:, :, slice_idx].astype(np.float32)
    image = np.rot90(slice_img, k=3)
    image = np.flip(image, axis=1) 
    vmin = np.percentile(image, 1)
    vmax = np.percentile(image, 99)
    image = np.clip(image, vmin, vmax)
    image = (image - vmin) / (vmax - vmin + 1e-8)
    
    st.image(image, caption=f"Slice {slice_idx}", width=500, clamp=True)

with col2:
    st.subheader("Segmentation Result")
    
    if st.button("🚀 Run Segmentation"):
        
        with st.spinner("Running inference on Datamint GPU..."):
            
            start = time.time()
            
            job = api.inference.predict_image(
                model_name='TMJ Test_adapted', 
                model_alias='latest', 
                resource_id=selected_res.id
            )
            job.wait()
            
            end = time.time()
            
            length = end - start
        
            h, w, z = vol.shape
            combined_mask = np.zeros((h, w, z), dtype=np.uint8)
            pred_voxels = {}
            
            CLASS_MAP = {
                    "Disc": 1,
                    "Condyle": 2,
                    "Eminence": 3,
                }
            
            predictions_list = job.result_data['predictions'][0]
            
            for pred in predictions_list:
                annotation = VolumeSegmentation(**pred)
                mask3d = annotation.segmentation_data 
                                
                class_name = annotation.name
                if not class_name and annotation.class_map:
                    class_name = list(annotation.class_map.values())[0]
                
                if not class_name:
                    continue
                    
                n_voxels = (mask3d > 0).sum()
                pred_voxels[class_name] = n_voxels
            
                if class_name in CLASS_MAP:
                    combined_mask[mask3d > 0] = CLASS_MAP[class_name]
                    
                mask_slice = combined_mask[:, :, slice_idx]

            mask_display = np.rot90(mask_slice, k=3)
            mask_display = np.flip(mask_display, axis=1)

            base_slice = vol[:, :, slice_idx].astype(np.float32)
            image_base = np.rot90(base_slice, k=3)
            image_base = np.flip(image_base, axis=1)
            
            vmin, vmax = np.percentile(image_base, [1, 99])
            image_base = np.clip((image_base - vmin) / (vmax - vmin + 1e-8), 0, 1)

            overlay_rgb = np.stack([image_base] * 3, axis=-1)
                
            COLORS_RGB = {
                1: [0, 0, 1],   # Disc - Blue
                2: [0, 1, 0],   # Condyle - Green
                3: [1, 1, 0]    # Eminence - Yellow
            }

            alpha = 0.5 
            for val, color in COLORS_RGB.items():
                pixels_da_classe = (mask_display == val)
                if pixels_da_classe.any():
                    for i in range(3): 
                        overlay_rgb[pixels_da_classe, i] = (1 - alpha) * overlay_rgb[pixels_da_classe, i] + alpha * color[i]

            st.image(overlay_rgb, caption=f"Segmentation Result - Slice {slice_idx}", width=500)
            
            st.sidebar.subheader("Voxel Stats (Prediction)")
            for name, count in pred_voxels.items():
                st.sidebar.write(f"{name}: {count} voxels")
            
            st.sidebar.write(f"It took {'{:.0f}'.format(length)} seconds.")

            st.success("Done.")