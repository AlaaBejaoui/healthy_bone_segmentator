import torch
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

if __name__ == "__main__":

    assert torch.cuda.is_available() , "No GPU available!"

    input_dir = Path("input/")
    output_dir = Path("output/")

    for ct_file in input_dir.glob("*.nii.gz"):

        ct_output_dir = output_dir / ct_file.stem.split(".")[0]
        ct_output_dir.mkdir(exist_ok=True)

        print(f"Processing: {ct_file.name}")

        # Femur segmentation
        totalsegmentator(
            input=str(ct_file),
            output=str(ct_output_dir),
            roi_subset=["femur_left", "femur_right"],
            task="total",
            device="gpu",
            nr_thr_saving=1,
            verbose=True
        )

        # Patella, tibia, and fibula segmentation
        totalsegmentator(
            input=str(ct_file),
            output=str(ct_output_dir),
            task="appendicular_bones",
            device="gpu",
            nr_thr_saving=1,
            verbose=True
        )

        print(f"Finished processing: {ct_file.name}")
    