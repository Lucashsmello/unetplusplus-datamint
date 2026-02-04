from datamint.mlflow import set_project
from datamint import Api 
from TMJDataset import TMJDataset2D
import lightning as L
from adapter import UNetPPSegmentationAdapter
from importlib import reload
from datamint.mlflow.flavors import datamint_flavor
from mlflow import set_experiment
import mlflow
from datamint.mlflow import set_project
from datamint import Api
import numpy as np
import nibabel as nib
import os

if __name__ == "__main__":   
    PROJECT_NAME = "TMJ Test"
    
    CLASS_MAP = {
    "Disc": 1,
    "Condyle": 2,
    "Eminence": 3,
    }
         
    adapter = UNetPPSegmentationAdapter()
    reload(datamint_flavor)
    
    set_project(PROJECT_NAME)
    set_experiment(f'{PROJECT_NAME}_deployment') 
    api = Api(verify_ssl=False)

    ''' Log the adapter model to MLflow '''
    ADAPTED_MODEL_NAME = f"{PROJECT_NAME}_adapted"
    with mlflow.start_run(run_name="unetpp_segmentation_adapter"): 
        model_info = datamint_flavor.log_model(
            adapter,
            registered_model_name=ADAPTED_MODEL_NAME,
        )

    print(f"✅ Adapter logged successfully!")
    print(f"Model URI: {model_info.model_uri}")
    
    ''' Testing local inference '''
    loaded_model = mlflow.pyfunc.load_model(f'models:/{ADAPTED_MODEL_NAME}/latest')
    test_dataset = TMJDataset2D(
        split='test',
    )
    test_resource = test_dataset.resources[0]
    print(f"Testing with: {test_resource.filename}")
    
    ''' Perform prediction '''
    predictions = adapter.predict_image([test_resource])

    print(f"\n Prediction successful!")
        
    ''' Load original volume to get metadata (Affine/Header) '''
    original_nii = test_resource.fetch_file_data(auto_convert=True)
    if hasattr(original_nii, "get_fdata"):
        h, w, z = original_nii.shape 
        combined_mask = np.zeros((h, w, z), dtype=np.uint8)
    else:
        vol_np = np.array(original_nii)
        h, w, z = vol_np.shape
        combined_mask = np.zeros((h, w, z), dtype=np.uint8)

    print(f"Volume Shape: {h}x{w}x{z}")
    print(f"Number of predicted classes: {len(predictions[0])}")

    ''' Dictionary to hold predicted voxel counts per class '''
    pred_voxels = {}

    for ann in predictions[0]:
        n_voxels = (ann.mask > 0).sum()
        pred_voxels[ann.name] = n_voxels
        
        ''' Calculate volume percentage '''
        vol_perc = (n_voxels / ann.mask.size) * 100
        print(f"  - [PRED] {ann.name}: {n_voxels} voxels ({vol_perc:.4f}% of total volume)")
        
        if ann.name in CLASS_MAP:
            combined_mask[ann.mask > 0] = CLASS_MAP[ann.name]

    print('\n--- Ground Truth Comparison ---')
    gt_annotations = api.annotations.get_list(
        resource=test_resource,
        annotation_type='segmentation'
    )

    for ann in gt_annotations:
        class_name = ann.identifier 
        raw_gt = ann.fetch_file_data(use_cache=True)
        
        if hasattr(raw_gt, "get_fdata"):
            mask_gt = raw_gt.get_fdata()
        else:
            mask_gt = np.array(raw_gt)
            
        n_voxels_gt = (mask_gt > 0).sum()
        gt_perc = (n_voxels_gt / mask_gt.size) * 100
        
        print(f"  - [GT]   {class_name}: {n_voxels_gt} voxels ({gt_perc:.4f}% of total volume)")
        
        if class_name in pred_voxels:
            error = abs(pred_voxels[class_name] - n_voxels_gt) / (n_voxels_gt + 1e-8)
            print(f"Volumetric Difference: {error:.2%}")

    ''' Save combined prediction mask as NIfTI '''
    prediction_filename = f"pred_local_{test_resource.filename}"
    new_img = nib.Nifti1Image(combined_mask, original_nii.affine, original_nii.header)
    nib.save(new_img, prediction_filename)

    print(f"\n Prediction saved to: {os.path.abspath(prediction_filename)}")
    
    ''' Deploy the adapted model '''
    job = api.deploy.start(
        model_name=ADAPTED_MODEL_NAME,
        model_alias="latest",
        with_gpu=True,  
    )

    print(f"🚀 Deployment job started!")
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Model: {job.model_name}")

    job = api.deploy.get_by_id(job.id)

    print(f"Job Status: {job.status}")
    print(f"Progress: {job.progress_percentage}%")

    if job.error_message:
        print(f"Error: {job.error_message}")

