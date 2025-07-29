import cv2
from PIL import Image, ImageOps
import numpy as np
import pywt
import torch
from skimage.feature import hog
from skimage import exposure
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Tuple, Union

def crop_unpadded(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    # Convert image to grayscale
    gray_image = ImageOps.grayscale(image)

    # Convert to NumPy array
    img_array = np.array(gray_image)

    # Create a binary mask where black (0) remains 0, and non-black is 1
    mask = img_array > 0

    # Get coordinates of non-black pixels
    coords = np.argwhere(mask)

    assert coords.size != 0, "Image is completely black"
    # if coords.size == 0:  # If the image is completely black, return the original
    #     return image

    # Determine bounding box
    top, left = coords.min(axis=0)      # Smallest row and column (top-left)
    bottom, right = coords.max(axis=0)  # Largest row and column (bottom-right)
    # Crop the image using calculated bounding box
    cropped_img = image.crop((left, top, right + 1, bottom + 1))  # +1 to include the last pixel
    return cropped_img, (top, left, bottom, right)

def pad_cropped(cropped_image: Image.Image, 
                original_size: tuple[int, int], 
                crop_coords: tuple[int, int, int, int]) -> Image.Image:
    # Create a new image with the original size
    padded_image = Image.new("RGB", original_size, 0)
    # Paste the cropped image back to its original position
    padded_image.paste(cropped_image, (crop_coords[1], crop_coords[0]))
    return padded_image

def correction_for_padded_img(filtering_func: callable) -> callable:
    def wrapper(x:Image.Image, *args, **kwargs):
        org_size = x.size
        x, padd_info = crop_unpadded(x)  # Apply A before
        x = filtering_func(x, *args, **kwargs)  # Apply B (whichever function is wrapped)
        if isinstance(x, tuple):
            x = tuple(pad_cropped(img, org_size, padd_info) for img in x)
        else:
            x = pad_cropped(x, org_size, padd_info)  # Apply C after
        return x
    return wrapper

@correction_for_padded_img
def calc_texture_channel(image_input: Union[str, Image.Image], 
                         alpha: float = 0.1, 
                         enhance_contrast: bool = True) -> Image.Image:
    """
    Extracts high-frequency texture from an image using FFT-based high-pass filtering.

    Parameters:
        image_input (str | PIL.Image): File path or PIL image.

    Returns:
        PIL.Image: Grayscale texture map of the input image.
    """
     # Load grayscale image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("L")
    else:
        image = image_input.convert("L")

    gray = np.array(image).astype(np.float32) / 255.0
    h, w = gray.shape

    # FFT
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)

    # High-pass filter
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * alpha)
    mask = np.ones_like(fft_shifted)
    mask[cy - r:cy + r, cx - r:cx + r] = 0

    filtered_fft = fft_shifted * mask
    inverse = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real

    # Normalize to 0–1
    texture = (inverse - inverse.min()) / (inverse.max() - inverse.min() + 1e-8)

    # Optional contrast enhancement
    if enhance_contrast:
        lower = np.percentile(texture, 5)
        upper = np.percentile(texture, 95)
        texture = np.clip((texture - lower) / (upper - lower + 1e-8), 0, 1)

    texture_img = (texture * 255).astype(np.uint8)
    return Image.fromarray(texture_img, mode="L")

@correction_for_padded_img
def calc_depth_channel(image_input: Union[str, Image.Image])->Image.Image:
    """
    Estimates the depth of an image using Depth-Anything-V2-Small-hf model.

    Parameters:
        image_url (str): URL of the image to process.

    Returns:
        PIL.Image: Grayscale depth map of the input image.
    """
    # Handle Input Image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input

    # Load model and processor
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    
    # Prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")
    
    # Perform depth estimation
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process depth estimation
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )
    
    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth_channel = depth.astype("uint8")

    # Convert to PIL image
    depth_channel = Image.fromarray(depth_channel, mode="L")
    
    return depth_channel

@correction_for_padded_img
def calc_HoG_channel(
    image_input: Union[str, Image.Image],
    pixels_per_cell: Tuple[int, int] = (8, 8), 
    cells_per_block: Tuple[int, int] = (2, 2), 
    orientations: int = 9
) -> Image.Image:

    # Handle Input Image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("L")
    else:
        image = image_input.convert("L")

    # Convert to NumPy array
    grey_array = np.array(image)

    # Compute HoG features (without visualization)
    _, hog_image = hog(
        grey_array, 
        orientations=orientations, 
        pixels_per_cell=pixels_per_cell, 
        cells_per_block=cells_per_block, 
        visualize=True  # No visualization, only raw features
    )
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_image = Image.fromarray((hog_image * 255).astype("uint8"), mode="L")

    return hog_image

@correction_for_padded_img
def calc_grey_channel(image_input: Union[str, Image.Image])->Image.Image:
    
    # Handle Input Image
    if isinstance(image_input, str):
        grey_channel = Image.open(image_input).convert("L")
    else:
        grey_channel = image_input.convert("L")

    return grey_channel

@correction_for_padded_img
def calc_Haar_channel(image_input: Union[str, Image.Image]) -> tuple[Image.Image, Image.Image, Image.Image]:
    
    # Handle Input Image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("L")
    else:
        image = image_input.convert("L")

    image = np.array(image)  # Convert to NumPy array
    original_shape = image.shape  
    
    # Approximation, Horizontal, Vertical, Diagonal coefficients
    _, (cH, cV, cD) = pywt.dwt2(image, 'haar')
    
    # Resize the coefficients back to the original image size
    cH = cv2.resize(cH, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
    cV = cv2.resize(cV, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
    cD = cv2.resize(cD, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)

    # Convert to PIL images
    cH = (cH - cH.min()) / (cH.max() - cH.min()) * 255
    cV = (cV - cV.min()) / (cV.max() - cV.min()) * 255
    cD = (cD - cD.min()) / (cD.max() - cD.min()) * 255

    cH = Image.fromarray(cH.astype("uint8"), mode="L")
    cV = Image.fromarray(cV.astype("uint8"), mode="L")
    cD = Image.fromarray(cD.astype("uint8"), mode="L")

    return cH, cV, cD

def calc_HHaar_channel(image_input: Union[str, Image.Image]) -> tuple[Image.Image, Image.Image]:
    return calc_Haar_channel(image_input)[0]

def calc_VHaar_channel(image_input: Union[str, Image.Image]) -> tuple[Image.Image, Image.Image]:
    return calc_Haar_channel(image_input)[1]

def calc_DHaar_channel(image_input: Union[str, Image.Image]) -> tuple[Image.Image, Image.Image]:
    return calc_Haar_channel(image_input)[2]

SUPP_CHNLS = {
    'grey': calc_grey_channel, 
    'depth': calc_depth_channel, 
    'hog': calc_HoG_channel, 
    'hhaar': calc_HHaar_channel, 
    'vhaar': calc_VHaar_channel, 
    'dhaar': calc_DHaar_channel,
    'texture': calc_texture_channel
}

def calc_supl_channels(image_input: Union[str, Image.Image], 
                       req_chnls:list[str]
                    ) -> dict[str, Image.Image]:

    assert req_chnls is not None, "No channels requested!"

    supp_chnls = {chnl: func(image_input) for chnl, func 
                  in SUPP_CHNLS.items() if chnl in req_chnls}
    
    return supp_chnls
