import gc
import os
import glob
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from natsort import natsorted

from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
from data import datasets
import utils

# ================= CONFIGURATION =================
DATASET = "adni"  # or "abdominal_mri"
TEST_CSV_PATH = "/midtier/sablab/scratch/alm4065/preprocess_for_transmorph/adni_pairs_transmorph_squashed.csv"
MODEL_DIR = "experiments/TransMorph_adni_mse_1_diffusion_0.02/"

# The root to strip off so we can mirror the folder structure
COMMON_ROOT = "/midtier/sablab/scratch/alm4065/adni_transmorph_squashed"  # or "/midtier/sablab/scratch/alm4065/abdominal_mri/"
OUTPUT_DIR = os.path.join(MODEL_DIR, "inference_results")

# How many pairs to generate (set to len(df) if you want all of them)
MAX_CASES_TO_SAVE = 1000
# =================================================


def save_nifti(tensor, affine, save_path):
    """
    Converts a PyTorch tensor [1, 1, D, H, W] to a NIfTI file and saves it.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Remove Batch and Channel dimensions: [D, H, W]
    array = tensor.detach().cpu().numpy()[0, 0]

    # Create NIfTI image using the provided affine
    nii_img = nib.Nifti1Image(array, affine)
    nib.save(nii_img, save_path)
    print(f"Saved: {save_path}")


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


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Setup Model
    config = CONFIGS_TM["TransMorph"]
    if DATASET == "abdominal_mri":
        # Update Model Config for your 160x160x160 data
        config.img_size = (160, 160, 160)
        config.window_size = (
            5,
            5,
            5,
        )  # Window size for attention (160 is divisible by 5)
    elif DATASET == "adni":
        # Same image size as the paper, so we can use the same config
        pass
    else:
        raise ValueError(
            "Unsupported dataset. Please choose 'abdominal_mri' or 'adni'."
        )
    img_size = config.img_size
    model = TransMorph.TransMorph(config)
    model.cuda()

    checkpoints = natsorted(glob.glob(MODEL_DIR + "*.pth.tar"))
    best_model_path = checkpoints[-1]
    print(f"Loading model: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Setup Warping Functions
    reg_model_label = utils.register_model(img_size, "nearest").cuda()
    reg_model_img = utils.register_model(img_size, "bilinear").cuda()

    # Data Loaders
    test_set = datasets.CSVPairDatasetwithTransform(TEST_CSV_PATH, mode="val")
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load CSV with Pandas
    df_csv = test_set.df

    print(
        f"Generating structured NIfTI files for the first {MAX_CASES_TO_SAVE} cases..."
    )

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            if idx >= MAX_CASES_TO_SAVE:
                break

            # Move to GPU
            data_gpu = [t.cuda() for t in data]
            x_moving = data_gpu[0]
            y_fixed = data_gpu[1]
            x_seg = data_gpu[2]

            # Forward Pass
            x_in = torch.cat((x_moving, y_fixed), dim=1)
            output = model(x_in)
            flow = output[1]

            # Warp the Moving Image and Moving Label
            moved_img = reg_model_img([x_moving.float(), flow])
            moved_seg = reg_model_label([x_seg.float(), flow])

            # --- Read row and get paths ---
            row = df_csv.iloc[idx]
            fixed_img_path = row["fixed_img_path"]
            moving_img_path = row["moving_img_path"]
            moving_seg_path = row["moving_seg_path"]

            # Extract Affine from the Preprocessed Fixed Image (Target Space)
            fixed_nii = nib.load(fixed_img_path)
            target_affine = fixed_nii.affine

            # --- Construct Mirrored Output Paths ---
            # We base the folder structure on the MOVING image's original path
            out_img_path = get_mirror_path(
                moving_img_path, fixed_img_path, suffix="_moved"
            )
            out_seg_path = get_mirror_path(
                moving_seg_path, fixed_img_path, suffix="_moved"
            )

            # Save the Outputs
            save_nifti(moved_img, target_affine, out_img_path)
            save_nifti(moved_seg, target_affine, out_seg_path)

            print("-" * 30)

            # 1. Delete all variables holding GPU memory
            del data_gpu, x_moving, y_fixed, x_seg
            del x_in, output, flow, moved_img, moved_seg

            # 2. Force Python to run garbage collection
            gc.collect()

            # 3. Force PyTorch to empty its VRAM cache
            torch.cuda.empty_cache()

    print(f"\nDone! Outputs generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        main()
    else:
        print("No GPU found!")
