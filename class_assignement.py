import os
import time
from pathlib import Path
import logging
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from scipy.ndimage import center_of_mass

# logging.basicConfig(filename=f"{os.path.join(os.getcwd(), str(int(time.time())))}.log", encoding='utf-8', level=logging.DEBUG)
logging.basicConfig(filename=Path(os.getcwd()) / "log" / "splitting.log", encoding='utf-8', level=logging.DEBUG)

# Function to split components into two regions based on size
def connected_components(array):
    labeled_array, num_labels = label(array, connectivity=3, return_num=True)
    regions = regionprops(labeled_array)
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)

    # Initialize arrays for the two largest components
    sorted_labeled_array = np.zeros_like(labeled_array, dtype=np.uint8)
    for new_label, region in enumerate(sorted_regions[:2], start=1):
        sorted_labeled_array[labeled_array == region.label] = new_label

    first_component = (sorted_labeled_array == 1).astype(np.uint8)
    second_component = (sorted_labeled_array == 2).astype(np.uint8)
    return first_component, second_component

# Function to assign left and right based on proximity to femur center(s)
def assign_sides(reference_femur_left, reference_femur_right, target_array):

    # Step 1: Check femur segmentation status
    if (reference_femur_left is None) and (reference_femur_right is None):
        logging.info(f"No femur segmentations as reference available!")
        return None, None

    # Determine the femur reference(s)
    # Calculate center of mass for available femur(s)
    femur_centroids = {}
    if reference_femur_left is not None:
        femur_centroids['left'] = center_of_mass(reference_femur_left)
        logging.info(f"left femur segmentations as reference available!")
    if reference_femur_right is not None:
        femur_centroids['right'] = center_of_mass(reference_femur_right)
        logging.info(f"right femur segmentations as reference available!")

    # Step 3: Perform connected component analysis on the target array
    first_component, second_component = connected_components(target_array)

    if (first_component.sum() != 0) and (second_component.sum() == 0):

        logging.info(f"only one component available!")

        first_centroid = center_of_mass(first_component)

        if 'left' in femur_centroids and 'right' in femur_centroids:

            first_to_left = np.linalg.norm(np.array(first_centroid) - np.array(femur_centroids['left'])) 
            first_to_right = np.linalg.norm(np.array(first_centroid) - np.array(femur_centroids['right']))

            if (first_to_left < first_to_right): 
                logging.info(f"component is left!")
                return first_component, None # First is closer to left femur
            else:
                logging.info(f"component is right!")
                return None, first_component # First is closer to right femur
        
        else:

            if 'left' in femur_centroids:
                logging.info(f"only left femur segmentations as reference is available! component will be considered left! To be verified!")
                return first_component, None

            elif 'right' in femur_centroids:
                logging.info(f"only right femur segmentations as reference is available! component will be considered right! To be verified!")
                return None, first_component

            else:
                logging.error(f"ERROR!! ...")
                return None, None
            

    # Calculate center of mass for components
    first_centroid = center_of_mass(first_component)
    second_centroid = center_of_mass(second_component)
    logging.info(f"two components available!")

    # Assign left/right based on the closest femur component
    if 'left' in femur_centroids and 'right' in femur_centroids:
        
        # If both femurs are available, use each femur centroid as reference
        first_to_left = np.linalg.norm(np.array(first_centroid) - np.array(femur_centroids['left'])) 
        first_to_right = np.linalg.norm(np.array(first_centroid) - np.array(femur_centroids['right']))
        second_to_left = np.linalg.norm(np.array(second_centroid) - np.array(femur_centroids['left'])) 
        second_to_right = np.linalg.norm(np.array(second_centroid) - np.array(femur_centroids['right']))

        # First is closer to left femur
        if (first_to_left < first_to_right) and (second_to_left > second_to_right): 
            logging.info(f"first component is left, second component is right!")
            return first_component, second_component 
                  
        elif (first_to_left > first_to_right) and (second_to_left < second_to_right):
            logging.info(f"first component is right, second component is left!") 
            return second_component, first_component
        else:
            logging.error(f"ERROR!!! ...")
            return None, None

    
    else:

        # Only left femur is available; use it as the sole reference
        if 'left' in femur_centroids:

            first_to_left = np.linalg.norm(np.array(first_centroid) - np.array(femur_centroids['left'])) 
            second_to_left = np.linalg.norm(np.array(second_centroid) - np.array(femur_centroids['left'])) 
            
            if (first_to_left < second_to_left):
                logging.info(f"first component is left, second component is right!")
                return first_component, second_component
            else:
                logging.info(f"first component is right, second component is left!")
                return second_component, first_component
            
        elif 'right' in femur_centroids:

            first_to_right = np.linalg.norm(np.array(first_centroid) - np.array(femur_centroids['right']))
            second_to_right = np.linalg.norm(np.array(second_centroid) - np.array(femur_centroids['right']))

            if (first_to_right < second_to_right):
                logging.info(f"first component is right, second component is left!")
                return second_component, first_component
            else:
                logging.info(f"first component is left, second component is right!")
                return first_component, second_component
            
        else:
            logging.error(f"ERROR!!!! ...")
            return None, None
        
def process(source_dir):

    # Define classes to keep
    keywords = ("femur", "patella", "tibia", "fibula")
    
    # Iterate through files and remove those not matching keywords
    for file in source_dir.glob("*.nii.gz"):
        if not any(keyword in file.stem for keyword in keywords):
            file.unlink()  # Deletes the file
            logging.info(f"{file} deleted!")
    
    
    # Define the path to the file
    femur_left_path = source_dir / "femur_left.nii.gz"
    femur_right_path = source_dir / "femur_right.nii.gz"
    
    # Check if femur_left exists, then load it
    if femur_left_path.exists():
        femur_left_img = nib.load(femur_left_path)
        femur_left_affine = femur_left_img.affine
        femur_left_header = femur_left_img.header
        femur_left = femur_left_img.get_fdata().astype(np.uint8)
        if (femur_left.sum() == 0):
            femur_left = None
            femur_left_path.unlink()  # Deletes the file
            logging.info(f"{femur_left_path} deleted because empty!")    
    else:
        femur_left = None
    
    # Check if femur_right exists, then load it
    if femur_right_path.exists():
        femur_right_img = nib.load(femur_right_path)
        femur_right_affine = femur_right_img.affine
        femur_right_header = femur_right_img.header
        femur_right = femur_right_img.get_fdata().astype(np.uint8)
    
        if (femur_right.sum() == 0):
            femur_right = None
            femur_right_path.unlink()  # Deletes the file
            logging.info(f"{femur_right_path} deleted because empty!")
    else:
        femur_right = None
    
    # Process each bone class using the femur as reference
    for bone_class in ("patella", "tibia", "fibula"):
        bone_path = source_dir / f"{bone_class}.nii.gz"
        bone_img = nib.load(bone_path)
        bone_affine = bone_img.affine
        bone_header = bone_img.header
        bone = bone_img.get_fdata().astype(np.uint8)
        
        if (bone.sum() == 0):
            bone_path.unlink()
            logging.info(f"{bone_path} deleted because empty!")
    
        else:
        
            left_bone, right_bone = assign_sides(femur_left, femur_right, bone)
    
            if (left_bone is not None) and (right_bone is not None):
                nib.save(nib.Nifti1Image(left_bone, bone_affine, bone_header), source_dir / f"{bone_class}_left.nii.gz")
                nib.save(nib.Nifti1Image(right_bone, bone_affine, bone_header), source_dir / f"{bone_class}_right.nii.gz")
                logging.info(f"left and right components saved!")
                bone_path.unlink()
                logging.info(f"{bone_path} deleted after assigning side!")
    
            elif (left_bone is not None) and (right_bone is None):
                nib.save(nib.Nifti1Image(left_bone, bone_affine, bone_header), source_dir / f"{bone_class}_left.nii.gz")
                logging.info(f"left component saved!")
                bone_path.unlink()
                logging.info(f"{bone_path} deleted after assigning side!")
            
            elif (right_bone is not None) and (left_bone is None):
                nib.save(nib.Nifti1Image(right_bone, bone_affine, bone_header), source_dir / f"{bone_class}_right.nii.gz")
                logging.info(f"right component saved!")
                bone_path.unlink()
                logging.info(f"{bone_path} deleted after assigning side!")
    
            else:
                logging.error(f"ERROR")

if __name__ == "__main__":

    root_dir = Path("output/")

    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            process(source_dir=subdir)


            
    
            
    
    
    
    
    