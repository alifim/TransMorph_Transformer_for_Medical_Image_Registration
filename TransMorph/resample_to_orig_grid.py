import os
import pandas as pd
import numpy as np
import nibabel as nib
import nibabel.processing

# ================= CONFIGURATION =================
TEST_CSV_PATH = "/midtier/sablab/scratch/alm4065/preprocess_for_transmorph/adni_pairs_transmorph_squashed.csv"  # '/midtier/sablab/scratch/alm4065/preprocess_for_transmorph/new_data_pairs_for_transmorph.csv'
MODEL_DIR = "experiments/TransMorph_adni_mse_1_diffusion_0.02/"
COMMON_ROOT = "/midtier/sablab/scratch/alm4065/adni_transmorph_squashed"  # or "/midtier/sablab/scratch/alm4065/abdominal_mri/"
OUTPUT_DIR = os.path.join(MODEL_DIR, "inference_results")
# =================================================


def get_mirror_path(moving_path, fixed_path, suffix="_moved"):
    """Reconstructs the path, appending the target fixed ID to prevent overwrites."""
    rel_path = os.path.relpath(moving_path, COMMON_ROOT)

    # Extract the unique folder name of the fixed image (e.g., 'imageTs/ADNI_002067_0011.nii.gz')
    fixed_id = os.path.basename(fixed_path).split(".")[0]

    # Inject the fixed_id into the filename so it is unique to this specific pair
    if rel_path.endswith(".nii.gz"):
        rel_path = rel_path.replace(".nii.gz", f"_to_{fixed_id}{suffix}.nii.gz")
    elif rel_path.endswith(".nii"):
        rel_path = rel_path.replace(".nii", f"_to_{fixed_id}{suffix}.nii")

    return os.path.join(OUTPUT_DIR, rel_path)


def get_true_original_path(csv_path):
    """Swaps the preprocessed root directory for the raw original directory."""
    old_root = "/midtier/sablab/scratch/alm4065/adni_transmorph_squashed/"  # or "/midtier/sablab/scratch/alm4065/abdominal_mri/"
    new_root = "/midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_raw_data_base/Dataset1007_ADNI/"  # or "/midtier/sablab/scratch/mch4003/"
    return csv_path.replace(old_root, new_root)


def main():
    print("Running strict physical space resampling with Nibabel...")
    df = pd.read_csv(TEST_CSV_PATH)
    df = df[df["train"] == False]

    print(f"Resampling {len(df)} cases back to true original fixed image grid...")

    for idx, row in df.iterrows():
        fixed_img_path_csv = row["fixed_img_path"]
        moving_img_path_csv = row["moving_img_path"]
        moving_seg_path_csv = row["moving_seg_path"]

        # Get True Original Fixed Image (Target Space)
        true_fixed_img_path = get_true_original_path(fixed_img_path_csv)

        if not os.path.exists(true_fixed_img_path):
            print(
                f"Warning: True fixed image not found at {true_fixed_img_path}. Skipping."
            )
            continue

        true_fixed_nii = nib.load(true_fixed_img_path)

        # Get existing 160^3 outputs
        moved_img_path = get_mirror_path(
            moving_img_path_csv, fixed_img_path_csv, suffix="_moved"
        )
        moved_seg_path = get_mirror_path(
            moving_seg_path_csv, fixed_img_path_csv, suffix="_moved"
        )

        # New paths for ITK-SNAP compatible files
        itk_img_path = get_mirror_path(
            moving_img_path_csv, fixed_img_path_csv, suffix="_moved_orig_grid"
        )
        itk_seg_path = get_mirror_path(
            moving_seg_path_csv, fixed_img_path_csv, suffix="_moved_orig_grid"
        )

        if not os.path.exists(moved_img_path):
            print(f"Skipping Case {idx}: {moved_img_path} not found.")
            continue

        # --- Resample Image (order=1 is Trilinear) ---
        moved_img_nii = nib.load(moved_img_path)

        resampled_img_nii = nibabel.processing.resample_from_to(
            moved_img_nii,
            true_fixed_nii,
            order=1,  # Trilinear interpolation for smooth MRI intensities
            cval=0.0,  # Pad with 0s if outside the field of view
        )
        nib.save(resampled_img_nii, itk_img_path)

        # --- Resample Segmentation (order=0 is Nearest Neighbor) ---
        if os.path.exists(moved_seg_path):
            moved_seg_nii = nib.load(moved_seg_path)

            resampled_seg_nii = nibabel.processing.resample_from_to(
                moved_seg_nii,
                true_fixed_nii,
                order=0,  # Nearest neighbor strictly protects integer labels
                cval=0.0,
            )

            # Safely cast to 16-bit int to satisfy NIfTI standard
            seg_data = np.round(resampled_seg_nii.get_fdata()).astype(np.int16)
            safe_seg_nii = nib.Nifti1Image(seg_data, resampled_seg_nii.affine)
            nib.save(safe_seg_nii, itk_seg_path)

        print(
            f"Case {idx:03d} -> Accurately resampled to physical shape: {true_fixed_nii.shape}. Fixed: {true_fixed_img_path}. Moved: {moved_img_path}"
        )

    print("-" * 40)
    print(
        "Resampling Complete! Your bodies should no longer look squished in ITK-SNAP."
    )


if __name__ == "__main__":
    main()
