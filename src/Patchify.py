import os
from PIL import Image

# A function to calculate image size
# How to use: calculate_image_size('../dataset/images/001a.jpg', True)
def calculate_image_size(image_path:str, print_on:bool=False)->list[int, int, int]:
    # Open an image
    image = Image.open(image_path)

    # Get image dimensions
    width, height = image.size

    # Calculate size in pixels
    total_pixels = width * height

    if print_on:
        print(f"Width: {width}, Height: {height}, Total Pixels: {total_pixels}")

    return width, height, total_pixels

# A function to cxreate patch out of an image 
def create_patch_from_image(image_path:str, patch_size:int)->dict[list[str], list[Image.Image], int]:
    """
    Slices a single image into patches of size patch_size x patch_size.
    Saves the patches in the output directory.
    """
    try:
        image = Image.open(image_path)
        width, height = image.size  # Get image dimensions

        # Get the original file extension
        file_extension = os.path.splitext(image_path)[1].lower()  # Get file extension, e.g., .jpg, .png
        patch_dict = {'patch_name':[], 'patch_img':[], 'count':0}
        patch_id = 0
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = image.crop((x, y, x + patch_size, y + patch_size))  # Extract patch
                
                if patch.size == (patch_size, patch_size):  # Ensure full patch
                    patch_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_patch_{patch_id}{file_extension}"
                    patch_id += 1

                    patch_dict['patch_img'].append(patch)
                    patch_dict['patch_name'].append(patch_filename)
                    patch_dict['count'] = patch_id
        
        return patch_dict

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# A function to create patch out of all dir images and save
def create_patch_from_dir(data_dir: str, output_dir: str, patch_size: int = 256, default_img_types=('png', 'jpg', 'jpeg', 'bmp'))->bool:
    """
    Loops over all images in the data_dir, slices them, and saves the patches into the output directory.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all images in the dataset
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(default_img_types):
                image_path = os.path.join(data_dir, filename)
                patch_dict = create_patch_from_image(image_path, patch_size)

                if patch_dict:
                    for i, patch in enumerate(patch_dict['patch_img']):
                        patch_filename = patch_dict['patch_name'][i]
                        patch.save(os.path.join(output_dir, patch_filename))
        # print("Image slicing complete!")
        return True
    
    except Exception as e:
        print(f"Error processing {output_dir}: {e}")
        return None

def resize_to_square(image_path: str, output_path: str, size: int):
    """
    Resizes an image to a square size of size x size while maintaining aspect ratio.
    Adds black padding to the side where necessary.

    Args:
    - image_path (str): Path to the input image.
    - output_path (str): Path where the resized square image will be saved.
    - size (int): The desired width and height of the resized square image.
    
    Returns:
    - None
    """
    try:
        # Open the image
        image = Image.open(image_path)
        original_width, original_height = image.size

        # Calculate the scaling factor to maintain aspect ratio
        if original_width > original_height:
            new_width = size
            new_height = int((size / original_width) * original_height)
        else:
            new_height = size
            new_width = int((size / original_height) * original_width)

        # Resize the image with aspect ratio maintained
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new black square image (size x size)
        final_image = Image.new("RGB", (size, size), (0, 0, 0))

        # Calculate padding
        left_padding = (size - new_width) // 2
        top_padding = (size - new_height) // 2

        # Paste the resized image into the center of the black square
        final_image.paste(resized_image, (left_padding, top_padding))

        # Save the final image
        final_image.save(output_path)

    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")

# A function to create resize all dir images and save
def resize_images_in_directory(input_dir: str, output_dir: str, size: int):
    """
    Applies resize_to_square to all images in the input directory and its subdirectories,
    saving them in the output directory.

    Args:
    - input_dir (str): Path to the directory containing images.
    - output_dir (str): Path where resized images will be saved.
    - size (int): The desired square size for all images (e.g., 256, 512).
    
    Returns:
    - None
    """
    # Check if output directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Traverse through the directory and its subdirectories
        for root, _, files in os.walk(input_dir):
            for file in files:
                # Check if the file is an image based on its extension
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                    image_path = os.path.join(root, file)

                    # Define the output path where the resized image will be saved
                    relative_path = os.path.relpath(image_path, input_dir)
                    output_path = os.path.join(output_dir, relative_path)

                    # Create the subdirectory in the output directory if necessary
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Apply the resizing function to each image
                    resize_to_square(image_path, output_path, size)
        # print("Resizing complete for all images!")
        return True

    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None

def apply_preprocessing(image_path:str='../dataset/images/', 
                        mask_path:str='../dataset/masks/',
                        resize_size:int=1024,
                        patch_size:int=256):
    
    resize_images_in_directory(image_path, '../dataset/images_resized/', resize_size)
    resize_images_in_directory(mask_path, '../dataset/masks_resized/', resize_size)

    create_patch_from_dir('../dataset/images_resized/', '../dataset/images_patch/', patch_size)
    create_patch_from_dir('../dataset/masks_resized/',  '../dataset/masks_patch/', patch_size)

    return
