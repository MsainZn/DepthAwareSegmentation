from collections import Counter
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from Filters import calc_supl_channels
from torch.utils.data import Dataset
from typing import Union, Literal
import random as rnd

img_config = {
    "RGB": ((0, 0, 0), Image.Resampling.LANCZOS),
    "L": (0, Image.Resampling.NEAREST),
}

def is_img_multichannel(tensor: torch.Tensor)->bool:
    tensor = remove_btch_dim_switch_cpu(tensor)
    if tensor.shape[0] > 3:
        return True
    return False

def add_btch_dim_swtch_dvc(img_tensor:torch.Tensor, 
                           device:torch.device)->torch.Tensor:
    return img_tensor.unsqueeze(0).to(device)

def remove_btch_dim_switch_cpu(tensor: torch.Tensor)->torch.Tensor:
    return tensor.squeeze(0).cpu()

def tensor_to_numpy_1D(tensor: torch.Tensor)->np.ndarray:
    img_1D = remove_btch_dim_switch_cpu(tensor).numpy()
    # Normalize to [0, 1]
    img_1D = (img_1D - img_1D.min()) / (img_1D.max() - img_1D.min() + 1e-6)  
    return img_1D

def tensor_to_numpy_rgb(tensor: torch.Tensor)->np.ndarray:
    img_rgb = remove_btch_dim_switch_cpu(tensor)
    # Convert from (C, H, W) -> (H, W, C) for matplotlib
    img_rgb = img_rgb.permute(1, 2, 0).numpy()
    return img_rgb

def numpy_to_pil_img(img: np.ndarray, mode:Literal['L', 'RGB'])->Image.Image:
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img, mode)

def tensor_to_pil_rgb(tensor: torch.Tensor)->Image.Image:
    img = remove_btch_dim_switch_cpu(tensor)
    img = transforms.ToPILImage(mode='RGB')(img)
    return img

def tensor_to_pil_msk(tensor: torch.Tensor)->Image.Image:
    img = remove_btch_dim_switch_cpu(tensor)
    img = img.to(dtype=torch.uint8)
    img = Image.fromarray(img.numpy(), mode='L')
    return img

def tensor_to_pil_1D(tensor: torch.Tensor)->Image.Image:
    img = remove_btch_dim_switch_cpu(tensor)
    img = transforms.ToPILImage(mode='RGB')(img)
    return img

def calc_size_to_square(img: Image.Image, target_size: int)->tuple[int, int]:
    """Calculate new height and width for resizing while maintaining aspect ratio."""
    original_width, original_height = img.size

    # Calculate scaling factor to maintain aspect ratio
    if original_width > original_height:
        new_width = target_size
        new_height = int((target_size / original_width) * original_height)
    else:
        new_height = target_size
        new_width = int((target_size / original_height) * original_width)

    return new_height, new_width

def calc_pad_to_square(img: Image.Image, target_size: int)->tuple[int, int]:
    """Calculate padding for centering image in square."""
    new_width, new_height = calc_size_to_square(img, target_size)

    # Calculate padding
    left_padding = (target_size - new_width) // 2
    top_padding = (target_size - new_height) // 2

    return left_padding, top_padding

def pad_to_square_trg(img: Image.Image, img_type:Literal["RGB", "L"], target_size: int) -> Image.Image:
    """Pad and resize image to square while maintaining aspect ratio."""
    
    clr, smpl_type = img_config.get(img_type)
    
    # Calculate new height and width
    new_width, new_height = calc_size_to_square(img, target_size)
    # Calculate padding
    left_pad, top_pad = calc_pad_to_square(img, target_size)
    
    # Resize the image
    resized_image = img.resize((new_width, new_height), smpl_type)
    # Create a new square image
    final_image = Image.new(img_type, (target_size, target_size), clr)
    # Paste the resized image into the center
    final_image.paste(resized_image, (left_pad, top_pad))

    return final_image

def calc_dataset_imbalance(dataset: Dataset, 
                           num_class: int, 
                           stats: bool=True,
                           eps: float=1e-6,):
        
    assert len(dataset) > 0, "Dataset not processed yet!"
    
    # Create a tensor to store the class counts
    class_counts = torch.zeros(num_class)  
    class_occurrences = Counter()

    # Iterate over all data in the Dataset
    for i in range(len(dataset)):
        _ , labels = dataset[i]
        labels = labels.flatten()
        for c in range(num_class):
            class_counts[c] += torch.sum(labels == c).item()
        
        unique_classes = torch.unique(labels).cpu().numpy()
        for c in unique_classes:
            class_occurrences[c] += 1
    
    # Total number of pixels in the dataset
    total_pixels = torch.sum(class_counts).item()
    class_weights = total_pixels / (eps + (num_class * class_counts))
    
    if stats:
        # Fraction of pixels belonging to each class  
        class_fractions = 100 * (class_counts / total_pixels)
        print(f"Class distribution (in terms of pixel count):")
        for c in range(num_class):
            print(f"Class {c}: {class_counts[c]} pixels, {class_fractions[c]:.2f}% of total pixels")
        print("Class occurrences per image:")
        for c, count in class_occurrences.items():
            print(f"Class {c}: {count} images")

    return class_weights

def list_imgs_in_dir(image_dir:str, img_types:list[str]=None)->list[str]:
    assert os.path.exists(image_dir), f"Image directory not found: {image_dir}"
    images = sorted(os.listdir(image_dir))
    if img_types:
        images = [img for img in images if os.path.splitext(img)[1].lower() in img_types]

    return images

def get_ds_full_pth(img_dir:str, 
                    msk_fir:str, 
                    img_ext:list[str] = ['.jpg', '.jpeg', '.png'],
                    msk_ext:list[str] = ['.bmp'])->tuple[list[str], list[str]]:
    
    imgs = list_imgs_in_dir(img_dir, img_ext)
    msks = list_imgs_in_dir(msk_fir, msk_ext)

    assert len(imgs) == len(msks), "Number of imgs and msks do not match."

    for img, msk in zip(imgs, msks):
        assert os.path.splitext(img)[0] == os.path.splitext(msk)[0], \
                f"Image and mask file names do not match: {img} != {msk}"
        
    # append all image names to dir path
    full_img_pth = [os.path.join(img_dir, img) for img in imgs]
    full_msk_pth = [os.path.join(msk_fir, msk) for msk in msks]
    
    return full_img_pth, full_msk_pth

def get_cls_from_msks(msk_dir:str, msk_ext:list[str])->dict[int, int]:
    # Get all unique pixel values in the mask
    unique_pixels = set()

    msk_files = list_imgs_in_dir(msk_dir, msk_ext)
    msk_full_path = [os.path.join(msk_dir, msk) for msk in msk_files]
    
    assert len(msk_full_path) > 0, f"No mask files found in {msk_dir} with suffixes {msk_ext}"

    for msk in msk_full_path:
        msk = Image.open(msk).convert("L")
        unique_pixels.update(msk.getdata())
    
    # Sort the unique pixel values
    unique_pixels = list(sorted(unique_pixels))

    # Assign each unique pixel value to a class
    label2pixel = {i: pixel for i, pixel in enumerate(unique_pixels)}

    # Sort the dictionary by pixel values
    label2pixel = dict(sorted(label2pixel.items(), key=lambda item: item[1]))

    return label2pixel

def read_rgb_from_file(img_path:str)->Image.Image:
    try:
        image = Image.open(img_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def read_1D_img_from_file(img_path:str)->Image.Image:
    try:
        image = Image.open(img_path).convert('L')
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def ds_load(img_dir:str, msk_dir:str,
            trg_size:int, pixel2label:dict,
            img_ext:list[str], msk_ext:list[str],
            in_channel: list[str]=None)->list[dict[str: torch.Tensor]]:
    
    # Get full image and mask paths
    img_fpth, msk_fpth = get_ds_full_pth(img_dir, msk_dir, img_ext, msk_ext)
    ds_proc = []
    for img_input, msk_input in zip(img_fpth, msk_fpth):
        # Load and Tensorize image
        img = process_img(img_input, trg_size) if 'rgb' in in_channel else {}
        # Load and Tensorize multi-channel
        mcs = process_mcs(img_input, trg_size, in_channel) if in_channel else {}
        # Load and Tensorize mask
        msk = process_msk(msk_input, trg_size, pixel2label)
        # Check if either image or multi-channel data is provided
        assert len(img) > 0 or len(mcs)>0, \
            f"Either IMG or AUX channels should be provided to train based on Mask."
        # Combine All Dictionaries and save
        ds_proc.append(({**img, **mcs}, msk['msk']))
        # Show loading progress in percentage
        print(f"Loading: {len(ds_proc)/len(img_fpth)*100:.2f}%", end='\r')
    return ds_proc

def process_mcs(img_input: Union[str, Image.Image, torch.Tensor], 
                trg_size: int,
                in_channel)->dict[str:torch.Tensor]:
    # Handle input types
    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, str):
        img = read_rgb_from_file(img_input)
    elif isinstance(img_input, torch.Tensor) and img_input.ndim == 3:
        return img_input
    else:     
        raise ValueError("Invalid input type. Use either str, Image.Image or torch.Tensor")

    img_padded = pad_to_square_trg(img, 'RGB', trg_size)

    aux_mcs = {}
    supp_chnls = calc_supl_channels(image_input=img_padded, req_chnls=in_channel)
    if len(supp_chnls) > 0:
        aux_mcs = {ky: tensorize_1D(vl, trg_size) for ky, vl in supp_chnls.items() 
                   if vl is not None}
    return aux_mcs

def process_img(img_input: Union[str, Image.Image, torch.Tensor], 
                trg_size: int)->dict[str:torch.Tensor]:
    # Handle input types
    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, str):
        img = read_rgb_from_file(img_input)
    elif isinstance(img_input, torch.Tensor) and img_input.ndim == 3:
        return img_input
    else:     
        raise ValueError("Invalid input type. Use either str, Image.Image or torch.Tensor")

    # Tensorize the image
    img_tensor = tensorize_RGB(img, trg_size)

    return {'rgb': img_tensor}

def process_msk(msk_input: Union[str, Image.Image, torch.Tensor], 
                trg_size: int,
                pixel2label: dict[int:int])->torch.Tensor:
    # Handle input types
    if isinstance(msk_input, Image.Image):
        msk = msk_input
    elif isinstance(msk_input, str):
        msk = read_1D_img_from_file(msk_input)
    elif isinstance(msk_input, torch.Tensor) and msk_input.ndim == 2:
        return msk_input
    else:     
        raise ValueError("Invalid input type. Use either str, Image.Image or torch.Tensor")

    # Tensorize and Remove extra dimension added due to having 1D mask
    msk_tensor = tensorize_msk(msk, trg_size, pixel2label).squeeze(0)
    
    return {'msk': msk_tensor}

# Consider that this function also adds 1 dimension to 1D images when transformed into tensor
def tensorize_1D(img_input: Image.Image, target_size:int)->torch.Tensor:
    # Apply the padding and resizing function images
    img = pad_to_square_trg(img_input, 'L', target_size)
    # Convert 1D image to tensor
    img = transforms.ToTensor()(img)
    return img

def tensorize_RGB(img_input: Image.Image, target_size:int)->torch.Tensor:
    # Apply the padding and resizing function images
    img = pad_to_square_trg(img_input, 'RGB', target_size)
    # Convert to tensor
    img = transforms.ToTensor()(img)
    return img

# This function also adds 1 dimension to the mask tensor which we remove in process dataset
def tensorize_msk(msk_input: Image.Image, target_size:int, pixel2label:dict)->torch.Tensor:
    # Modify the mask pixel values to class indices
    msk = modfy_msk(msk_input, pixel2label)
    # Apply the padding and resizing function images
    msk = pad_to_square_trg(msk, 'L', target_size)
    # Convert to tensor and remove
    msk = transforms.PILToTensor()(msk).long()  # msk = torch.from_numpy(msk).long()
    return msk

def modfy_msk(msk: Image.Image, pixel2label: dict) -> Image.Image:
    """Convert grayscale mask (PIL Image) to class indices (PIL Image) without using NumPy."""
    # Get pixel values as a list
    pixels = list(msk.getdata())
    # Map grayscale values to class indices
    new_pixels = [pixel2label.get(p, 0) for p in pixels]  # Default to 0 if pixel not in dictionary
    # Create a new image with the same size and mode
    mask_class = Image.new("L", msk.size)
    # Assign the new pixel values
    mask_class.putdata(new_pixels)
    return mask_class

# This function sets the rand§§om seed for reproducibility in PyTorch and NumPy
def set_cuda_reproducible(seed=42):

    # Python
    rnd.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    return
