import os
import re
import nibabel as nib
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def extract_subject_id(file_name):
    # Use regular expression to extract "IXI0xx" from the file name
    match = re.search(r'IXI\d+', file_name)
    return match.group() if match else None

def nifti_to_images(nifti_path, output_folder):
    # Load NIFTI image
    nifti_img = nib.load(nifti_path)
    image_data = nifti_img.get_fdata()

    # Extract subject ID from the file name
    subject_id = extract_subject_id(os.path.basename(nifti_path))

    if subject_id:
        # Create output folder for the subject if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Determine the range of slices to process
        start_slice = 10
        end_slice = image_data.shape[-1] - 5

        # Iterate through selected slices and save images
        for i in range(start_slice, end_slice):
            # Extract the slice
            slice_data = ndi.rotate(image_data[:, :, i], 90)

            # Plot and save the slice as an image
            plt.imshow(slice_data, cmap='gray')
            plt.axis('off')  # Remove axis labels
            image_name = f'{subject_id}_slice_{i}.png'
            plt.savefig(os.path.join(output_folder, image_name), bbox_inches='tight', pad_inches=0)
            plt.close()
        print(f"Image saved at {os.path.join(output_folder, image_name)}")

nifti_folder = 'D:/Medical_Imagery_SRCNN/images/IXI-T1'
output_folder = 'D:/Medical_Imagery_SRCNN/images/IXI-T1-PNG'

for file_name in os.listdir(nifti_folder):
    if file_name.endswith('.nii'):
        nifti_path = os.path.join(nifti_folder, file_name)
        nifti_to_images(nifti_path, output_folder)