import gc
import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SSIMMetric
from monai.networks.utils import one_hot

# ================= CONFIGURATION =================
DATASET = "adni"  # or "abdominal_mri"
TEST_CSV_PATH = '/midtier/sablab/scratch/alm4065/preprocess_for_transmorph/adni_pairs_transmorph_squashed.csv' # or '/midtier/sablab/scratch/alm4065/preprocess_for_transmorph/new_data_pairs_for_transmorph.csv' 
MODEL_DIR = 'experiments/TransMorph_adni_mse_1_diffusion_0.02'
COMMON_ROOT = '/midtier/sablab/scratch/alm4065/adni_transmorph_squashed' # or "/midtier/sablab/scratch/alm4065/abdominal_mri/"
OUTPUT_DIR = os.path.join(MODEL_DIR, 'inference_results') 

RESULTS_CSV = os.path.join(MODEL_DIR, 'evaluation_results_original_grid.csv')
# =================================================

def get_mirror_path(moving_path, fixed_path, suffix="_moved"):
    """Reconstructs the path, appending the target fixed ID to prevent overwrites."""
    rel_path = os.path.relpath(moving_path, COMMON_ROOT)

    if DATASET == "adni":
        # Extract the unique folder name of the fixed image (e.g., 'imageTs/ADNI_002067_0011.nii.gz')
        fixed_id = os.path.basename(fixed_path).split(".")[0]
    elif DATASET == "abdominal_mri":
        # Extract the unique folder name of the fixed image (e.g., '00000000_PrM_...')
        fixed_id = os.path.basename(os.path.dirname(fixed_path))
    else:
        raise ValueError("Unsupported dataset. Please choose 'abdominal_mri' or 'adni'.")

    # Inject the fixed_id into the filename so it is unique to this specific pair
    if rel_path.endswith(".nii.gz"):
        rel_path = rel_path.replace(".nii.gz", f"_to_{fixed_id}{suffix}.nii.gz")
    elif rel_path.endswith(".nii"):
        rel_path = rel_path.replace(".nii", f"_to_{fixed_id}{suffix}.nii")

    return os.path.join(OUTPUT_DIR, rel_path)

def get_true_original_path(csv_path):
    """
    Swaps the preprocessed root directory for the raw original directory
    and gracefully handles .nii vs .nii.gz inconsistencies.
    """
    if pd.isna(csv_path) or not isinstance(csv_path, str):
        return None
        
    old_root = "/midtier/sablab/scratch/alm4065/adni_transmorph_squashed/" # or "/midtier/sablab/scratch/alm4065/abdominal_mri/"
    new_root = "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset1007_ADNI/" # or "/midtier/sablab/scratch/mch4003/"
    
    # Swap the base directory
    expected_path = csv_path.replace(old_root, new_root)
    
    # Check if the exact path from the CSV exists (.nii.gz)
    if os.path.exists(expected_path):
        return expected_path
        
    # If it doesn't exist, try falling back to the .nii extension
    if expected_path.endswith('.nii.gz'):
        fallback_path = expected_path[:-3] # Strips the '.gz'
        if os.path.exists(fallback_path):
            return fallback_path
            
    # Return the expected path anyway if neither exists, so the main loop can skip/warn properly
    return expected_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    # Initialize MONAI Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=100, reduction="mean")
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0) 

    df = pd.read_csv(TEST_CSV_PATH)
    df = df[df["train"]==False]
    results = []

    print(f"Starting evaluation of {len(df)} cases on ORIGINAL grid...")

    with torch.no_grad():
        for idx, row in df.iterrows():
            # Ground truth original targets
            true_fixed_img_path = get_true_original_path(row['fixed_img_path'])
            true_fixed_seg_path = get_true_original_path(row.get('fixed_seg_path', None))

            # Original moving image
            true_moving_img_path = get_true_original_path(row['moving_img_path'])
            true_moving_seg_path = get_true_original_path(row.get('moving_seg_path', None))
            
            # Resampled predictions
            moved_img_path = get_mirror_path(row['moving_img_path'], row['fixed_img_path'], suffix='_moved_orig_grid')
            moved_seg_path = get_mirror_path(row['moving_seg_path'], row['fixed_seg_path'], suffix='_moved_orig_grid')

            if not os.path.exists(moved_img_path) or not os.path.exists(true_fixed_img_path):
                print(f"Skipping case {idx}: Missing files.")
                continue
            else:
                print(f"Fixed: {true_fixed_img_path}, Moved: {moved_img_path}")

            # --- Evaluate Images (SSIM) ---
            true_fixed_nii = nib.load(true_fixed_img_path)
            true_fixed_img = true_fixed_nii.get_fdata()
            moved_img = nib.load(moved_img_path).get_fdata()

            fixed_img_t = torch.tensor(true_fixed_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            moved_img_t = torch.tensor(moved_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            ## --- Normalize Fixed Image to [0, 1] for accurate SSIM ---
            f_min = fixed_img_t.min()
            f_max = fixed_img_t.max()
            
            # Prevent division by zero in weird edge cases
            if f_max > f_min:
                fixed_img_norm_t = (fixed_img_t - f_min) / (f_max - f_min)
            else:
                fixed_img_norm_t = fixed_img_t

            # Ensure moved image is strictly clipped to [0, 1] just in case of model artifacts
            moved_img_norm_t = torch.clamp(moved_img_t, min=0.0, max=1.0)

            # Compute SSIM on the normalized tensors
            ssim_metric.data_range = 1.0 # Both are now exactly [0, 1]
            ssim_val = ssim_metric(moved_img_norm_t, fixed_img_norm_t).item()
            ssim_metric.reset()

            # --- Evaluate Segmentations (Dice & HD) ---
            dice_val = np.nan
            hd_val = np.nan

            if true_fixed_seg_path and os.path.exists(true_fixed_seg_path) and os.path.exists(moved_seg_path):
                true_fixed_seg = nib.load(true_fixed_seg_path).get_fdata()
                moved_seg = nib.load(moved_seg_path).get_fdata()
                
                # Extract voxel spacing and convert safely to standard Python floats
                spacing = [float(s) for s in true_fixed_nii.header.get_zooms()[:3]]
                
                # Load as integers and send to GPU
                fixed_seg_t = torch.tensor(true_fixed_seg, dtype=torch.int64, device=device).unsqueeze(0).unsqueeze(0)
                moved_seg_t = torch.tensor(moved_seg, dtype=torch.int64, device=device).unsqueeze(0).unsqueeze(0)

                # Find max classes to dynamically one-hot encode
                num_classes = max(int(fixed_seg_t.max().item()), int(moved_seg_t.max().item())) + 1
                
                if num_classes > 1:
                    fixed_seg_onehot = one_hot(fixed_seg_t, num_classes=num_classes)
                    moved_seg_onehot = one_hot(moved_seg_t, num_classes=num_classes)
        
                    # Compute Dice
                    dice_metric(y_pred=moved_seg_onehot, y=fixed_seg_onehot)
                    dice_val = dice_metric.aggregate().item()
                    dice_metric.reset() 
                    
                    # Compute HD in true physical millimeter space
                    hd_metric(y_pred=moved_seg_onehot, y=fixed_seg_onehot, spacing=spacing)
                    hd_val = hd_metric.aggregate().item()
                    hd_metric.reset()

                    del fixed_seg_onehot
                    del moved_seg_onehot
                    del fixed_seg_t
                    del moved_seg_t
                    gc.collect()
                else:
                    print(f"Case {idx} has no foreground segmentation classes. Skipping Dice/HD.")

            else:
                print(f"Fixed seg path is {os.path.exists(true_fixed_seg_path)}: {true_fixed_seg_path}, Moved seg path is {os.path.exists(moved_seg_path)}: {moved_seg_path}")

            # --- Record Results ---
            results.append({
                'fixed_img': true_fixed_img_path,
                'moving_img': true_moving_img_path,
                'moved_img': moved_img_path,
                'Dice': dice_val,
                'HD': hd_val,
                'SSIM': ssim_val,
            })
            
            print(f"Case {idx:03d} | Dice: {dice_val:.4f} | HD: {hd_val:.4f}mm | SSIM: {ssim_val:.4f}")

    # --- Summarize and Save ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    
    print("-" * 40)
    print("Original Grid Evaluation Complete!")
    print(f"Average SSIM: {results_df['SSIM'].mean():.4f}")
    print(f"Average Dice: {results_df['Dice'].mean():.4f}")
    print(f"Average HD: {results_df['HD'].mean():.4f} mm")
    print(f"Detailed results saved to: {RESULTS_CSV}")

if __name__ == '__main__':
    main()