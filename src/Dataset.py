import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import math
import pickle
from random import randint
from Preprocess import ds_load, calc_dataset_imbalance, get_cls_from_msks, process_img, process_msk, process_mcs, get_ds_full_pth
from typing import Union

class SegmentationDataset(Dataset):
    
    def __init__(self, 
                 label2pixel:dict[int, int]=None,
                 image_dir:str=None,
                 mask_dir:str=None,
                 save_dir:str=None,
                 name:str=None,
                 in_channel:list[str]=None)->None:
        
        # Class Prams
        self.img_msk_prc:list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
        self.img_dir:str = image_dir
        self.msk_dir:str = mask_dir
        self.sve_dir:str = save_dir
        self.name:str = name
        self.img_size:int = None
        self.class_weights:torch.Tensor = None
        self.msk_file_ext:list[str] = []
        self.img_file_ext:list[str] = []

        self.in_channel = in_channel

        if label2pixel is not None:
            self.set_ds_properties(label2pixel)
        else:
            self.label2pixel = None
            self.pixel2label = None
            self.num_class = None       

    def set_img_ext(self, img_file_ext:Union[list[str], str])->'SegmentationDataset':

        if isinstance(img_file_ext, str):   
            self.img_file_ext.append(img_file_ext)
    
        if isinstance(img_file_ext, list):
            self.img_file_ext.extend(img_file_ext)

        return self
    
    def set_msk_ext(self, msk_file_ext:Union[list[str], str])->'SegmentationDataset':

        if isinstance(msk_file_ext, str):
            self.msk_file_ext.append(msk_file_ext)
        
        if isinstance(msk_file_ext, list):
            self.msk_file_ext.extend(msk_file_ext)

        return self

    def set_channels(self, in_channel:list[str])->'SegmentationDataset':
        
        self.in_channel = in_channel

        if 'rgb' in self.in_channel:
            self.in_channel_count = len(self.in_channel) + 2
        else:   
            self.in_channel_count = len(self.in_channel)

        return self

    def set_process_dirs(self, img_dir:str, msk_dir:str)->'SegmentationDataset':
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        return self
    
    def set_save_dir(self, sve_dir:str)->'SegmentationDataset':
        self.sve_dir = sve_dir
        return self
    
    # Load processed dataset from file
    @classmethod
    def load_from_file(cls, sve_dir: str, name: str)->'SegmentationDataset':
        """Loads a dataset instance from a file."""
        assert os.path.exists(sve_dir), f"Save directory not found: {sve_dir}"
        
        file_path = os.path.join(sve_dir, f"{name}.pkl")
        with open(file_path, "rb") as f:
            dataset: SegmentationDataset = pickle.load(f)
        
        dataset.sve_dir = sve_dir
        dataset.name = name

        print(f"Dataset loaded from {file_path}")
        return dataset

    # Save processed dataset to file
    def save_to_file(self, name:str=None)->'SegmentationDataset':        
        
        assert len(self) > 0, "No processed data to save!"
        assert os.path.exists(self.sve_dir), f"Directory not found: {self.sve_dir}"
        
        if name is None:
            name = f'dataset_{self.in_channel_count}x{self.img_size}x{self.img_size}-{"_".join(self.in_channel)}'

        self.name = name
        file_path = os.path.join(self.sve_dir, f"{self.name}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

        print(f"Dataset saved to {file_path}")
        return self
    
    # Class-to-pixel mapping
    def set_ds_properties(self, label2pixel:dict[int, int]=None)->'SegmentationDataset':
        self.label2pixel = label2pixel
        self.pixel2label = {v: k for k, v in self.label2pixel.items()}
        self.num_class = len(self.label2pixel)
        return self

    # Auto extract class labels from masks
    def manage_class_labels(self, msk_file_ext:list[str])->'SegmentationDataset':
        self.label2pixel = get_cls_from_msks(self.msk_dir, msk_file_ext)
        self.pixel2label = {v: k for k, v in self.label2pixel.items()}
        self.num_class = len(self.label2pixel)
        return self

    # Remove channels from all elements in the list of dictionaries
    def remove_channels(self, chnl:Union[str | list[str]])->'SegmentationDataset':
        
        # Check if the dataset is empty
        assert len(self) > 0, "Dataset is empty!"

        # change single input to list
        chnl = [chnl] if isinstance(chnl, str) else chnl            


        # find the substitute elements in both chnl and in_channel lists and put it in a new list
        common_elements = list(set(chnl) & set(self.in_channel))

        # Check if the channels are valid
        assert common_elements is not None, "No Common Elements!"
        
        # Update input channel count
        num_removed = len(chnl) if 'rgb' not in chnl else len(chnl) + 2
        self.in_channel_count -= num_removed 
        
        # Remove channels from each dictionary in the list
        for img_dict, _ in self:
            for ch in chnl:
                img_dict.pop(ch, None)
        
        return self
   
    # Process dataset
    def load_ds_to_mem(self, img_size:int, stats:bool=False)->'SegmentationDataset':
        # List all images and masks
        assert os.path.exists(self.img_dir), f"Image directory not found: {self.img_dir}"
        assert os.path.exists(self.msk_dir), f"Mask directory not found: {self.msk_dir}"
        assert len(self.in_channel)>0, "No input channels to process!"
        
        if self.label2pixel is None:
            print(f"Extract class labels from masks started...")
            self.manage_class_labels(self.msk_file_ext)
        
        assert self.pixel2label is not None, "Dataset properties not set!"

        print(f"Processing dataset started...")
        self.img_msk_prc = ds_load(self.img_dir, 
                                   self.msk_dir, 
                                   img_size,
                                   self.pixel2label,
                                   self.img_file_ext,
                                   self.msk_file_ext,
                                   self.in_channel)
        
        print('Calculating class weights...')
        self.class_weights = calc_dataset_imbalance(self, self.num_class, stats=stats)
        print('Dataset processing completed!')
        self.img_size = img_size
        return self
    
    def __str__(self):
        super().__str__()

        return f"""
        Dataset Name: {self.name}
        Image Directory: {self.img_dir}
        Mask Directory: {self.msk_dir}
        Save Directory: {self.sve_dir}
        Image Size: {self.img_size}
        Class Weights: {self.class_weights}
        Class Labels: {self.label2pixel}
        Number of Classes: {self.num_class}
        Input Channels: {self.in_channel}
        Input Channel Count: {self.in_channel_count}
        Image File Extensions: {self.img_file_ext}
        Mask File Extensions: {self.msk_file_ext}
        Number of Samples: {len(self.img_msk_prc)}
        """

    def __len__(self) -> int:
        return len(self.img_msk_prc)

    def __setitem__(self, idx: int, item: tuple[dict[str, torch.Tensor], torch.Tensor]) -> None:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("Item must be a tuple of (image_dict, mask_tensor)")
        
        img_dict, msk_tns = item
        self.img_msk_prc[idx] = (img_dict, msk_tns)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        img_dict, msk_tns = self.img_msk_prc[idx]
        return img_dict, msk_tns
    
    def __iter__(self):
        for item in range(len(self)):
            yield self[item]

# class LazySegmentationDataset(Dataset):
    
#     def __init__(self, 
#                  label2pixel:dict[int, int]=None,
#                  image_dir:str=None,
#                  mask_dir:str=None,
#                  save_dir:str=None,
#                  name:str=None,
#                  in_channel:list[str]=None)->None:
        
#         # Class Prams
#         self.img_msk_prc: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
#         self.img_dir:str = image_dir
#         self.msk_dir:str = mask_dir
#         self.sve_dir:str = save_dir
#         self.name:str = name
#         self.img_size:int = None
#         self.class_weights:torch.Tensor = None
#         self.msk_file_ext:list[str] = []
#         self.img_file_ext:list[str] = []

#         self.in_channel = in_channel

#         if label2pixel is not None:
#             self.set_ds_properties(label2pixel)
#         else:
#             self.label2pixel = None
#             self.pixel2label = None
#             self.num_class = None       

#     def set_img_ext(self, img_file_ext:Union[list[str], str])->'SegmentationDataset':

#         if isinstance(img_file_ext, str):   
#             self.img_file_ext.append(img_file_ext)
    
#         if isinstance(img_file_ext, list):
#             self.img_file_ext.extend(img_file_ext)

#         return self
    
#     def set_msk_ext(self, msk_file_ext:Union[list[str], str])->'SegmentationDataset':

#         if isinstance(msk_file_ext, str):
#             self.msk_file_ext.append(msk_file_ext)
        
#         if isinstance(msk_file_ext, list):
#             self.msk_file_ext.extend(msk_file_ext)

#         return self

#     def set_channels(self, in_channel:list[str])->'SegmentationDataset':
        
#         self.in_channel = in_channel

#         if 'rgb' in self.in_channel:
#             self.in_channel_count = len(self.in_channel) + 2
#         else:   
#             self.in_channel_count = len(self.in_channel)

#         return self

#     def set_process_dirs(self, img_dir:str, msk_dir:str)->'SegmentationDataset':
#         self.img_dir = img_dir
#         self.msk_dir = msk_dir
#         return self
    
#     def set_save_dir(self, sve_dir:str)->'SegmentationDataset':
#         self.sve_dir = sve_dir
#         return self
    
#     def lazy_load_ds_to_mem(self, img_size: int, stats: bool = False) -> 'SegmentationDataset':
#         assert os.path.exists(self.img_dir), f"Image directory not found: {self.img_dir}"
#         assert os.path.exists(self.msk_dir), f"Mask directory not found: {self.msk_dir}"
#         assert len(self.in_channel) > 0, "No input channels to process!"

#         if self.label2pixel is None:
#             print(f"Extract class labels from masks started...")
#             self.manage_class_labels(self.msk_file_ext)

#         self.img_size = img_size

#         self.img_msk_prc = get_ds_full_pth(self.img_dir, 
#                                            self.msk_dir, 
#                                            self.img_file_ext, 
#                                            self.msk_file_ext)
#         print("Using lazy loading, data will be processed on the fly.")
#         return self

#     # Load processed dataset from file
#     @classmethod
#     def load_from_file(cls, sve_dir: str, name: str)->'SegmentationDataset':
#         """Loads a dataset instance from a file."""
#         assert os.path.exists(sve_dir), f"Save directory not found: {sve_dir}"
        
#         file_path = os.path.join(sve_dir, f"{name}.pkl")
#         with open(file_path, "rb") as f:
#             dataset: SegmentationDataset = pickle.load(f)
        
#         dataset.sve_dir = sve_dir
#         dataset.name = name

#         print(f"Dataset loaded from {file_path}")
#         return dataset

#     # Save processed dataset to file
#     def save_to_file(self, name:str=None)->'SegmentationDataset':        
        
#         assert len(self) > 0, "No processed data to save!"
#         assert os.path.exists(self.sve_dir), f"Directory not found: {self.sve_dir}"
        
#         if name is None:
#             name = f'dataset_{self.in_channel_count}x{self.img_size}x{self.img_size}-{"_".join(self.in_channel)}'

#         self.name = name
#         file_path = os.path.join(self.sve_dir, f"{self.name}.pkl")
#         with open(file_path, "wb") as f:
#             pickle.dump(self, f)

#         print(f"Dataset saved to {file_path}")
#         return self
    
#     # Class-to-pixel mapping
#     def set_ds_properties(self, label2pixel:dict[int, int]=None)->'SegmentationDataset':
#         self.label2pixel = label2pixel
#         self.pixel2label = {v: k for k, v in self.label2pixel.items()}
#         self.num_class = len(self.label2pixel)
#         return self

#     # Auto extract class labels from masks
#     def manage_class_labels(self, msk_file_ext:list[str])->'SegmentationDataset':
#         self.label2pixel = get_cls_from_msks(self.msk_dir, msk_file_ext)
#         self.pixel2label = {v: k for k, v in self.label2pixel.items()}
#         self.num_class = len(self.label2pixel)
#         return self

#     # Remove channels from all elements in the list of dictionaries
#     def remove_channels(self, chnl:Union[str | list[str]])->'SegmentationDataset':
        
#         # Check if the dataset is empty
#         assert len(self) > 0, "Dataset is empty!"

#         # change single input to list
#         chnl = [chnl] if isinstance(chnl, str) else chnl            


#         # find the substitute elements in both chnl and in_channel lists and put it in a new list
#         common_elements = list(set(chnl) & set(self.in_channel))

#         # Check if the channels are valid
#         assert common_elements is not None, "No Common Elements!"
        
#         # Update input channel count
#         num_removed = len(chnl) if 'rgb' not in chnl else len(chnl) + 2
#         self.in_channel_count -= num_removed 
        
#         # Remove channels from each dictionary in the list
#         for img_dict, _ in self:
#             for ch in chnl:
#                 img_dict.pop(ch, None)
        
#         return self
   
#     # Process dataset
#     def load_ds_to_mem(self, img_size:int, stats:bool=False)->'SegmentationDataset':
#         # List all images and masks
#         assert os.path.exists(self.img_dir), f"Image directory not found: {self.img_dir}"
#         assert os.path.exists(self.msk_dir), f"Mask directory not found: {self.msk_dir}"
#         assert len(self.in_channel)>0, "No input channels to process!"
        
#         if self.label2pixel is None:
#             print(f"Extract class labels from masks started...")
#             self.manage_class_labels(self.msk_file_ext)
        
#         assert self.pixel2label is not None, "Dataset properties not set!"

#         print(f"Processing dataset started...")
#         self.img_msk_prc = ds_load(self.img_dir, 
#                                    self.msk_dir, 
#                                    img_size,
#                                    self.pixel2label,
#                                    self.img_file_ext,
#                                    self.msk_file_ext,
#                                    self.in_channel)
        
#         print('Calculating class weights...')
#         self.class_weights = calc_dataset_imbalance(self, self.num_class, stats=stats)
#         print('Dataset processing completed!')
#         self.img_size = img_size
#         return self
    
#     def __str__(self):
#         super().__str__()

#         return f"""
#         Dataset Name: {self.name}
#         Image Directory: {self.img_dir}
#         Mask Directory: {self.msk_dir}
#         Save Directory: {self.sve_dir}
#         Image Size: {self.img_size}
#         Class Weights: {self.class_weights}
#         Class Labels: {self.label2pixel}
#         Number of Classes: {self.num_class}
#         Input Channels: {self.in_channel}
#         Input Channel Count: {self.in_channel_count}
#         Image File Extensions: {self.img_file_ext}
#         Mask File Extensions: {self.msk_file_ext}
#         Number of Samples: {len(self.img_msk_prc)}
#         """

#     def __len__(self) -> int:
#         return len(self.img_msk_prc)

#     def __setitem__(self, idx: int, item: tuple[dict[str, torch.Tensor], torch.Tensor]) -> None:
#         if not isinstance(item, tuple) or len(item) != 2:
#             raise ValueError("Item must be a tuple of (image_dict, mask_tensor)")
        
#         img_dict, msk_tns = item
#         self.img_msk_prc[idx] = (img_dict, msk_tns)

#     def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
#         img_path = self.img_paths[idx]
#         msk_path = self.mask_paths[idx]

#         img = process_img(img_path, self.img_size) if 'rgb' in self.in_channel else {}
#         mcs = process_mcs(img_path, self.img_size, self.in_channel) if self.in_channel else {}
#         msk = process_msk(msk_path, self.img_size, self.pixel2label)
#         return {**img, **mcs}, msk['msk']
    
#     def __iter__(self):
#         for item in range(len(self)):
#             yield self[item]

# img_fpth, msk_fpth = get_ds_full_pth(img_dir, msk_dir, img_ext, msk_ext)
# ds_proc = []
# for img_input, msk_input in zip(img_fpth, msk_fpth):
#     # Load and Tensorize image
#     img = process_img(img_input, trg_size) if 'rgb' in in_channel else {}
#     # Load and Tensorize multi-channel
#     mcs = process_mcs(img_input, trg_size, in_channel) if in_channel else {}
#     # Load and Tensorize mask
#     msk = process_msk(msk_input, trg_size, pixel2label)
#     # Check if either image or multi-channel data is provided
#     assert len(img) > 0 or len(mcs)>0, \
#         f"Either IMG or AUX channels should be provided to train based on Mask."
#     # Combine All Dictionaries and save
#     ds_proc.append(({**img, **mcs}, msk['msk']))

def random_sample_dataset(dataset: Dataset) -> tuple[dict[str:torch.Tensor], torch.Tensor]:
    return dataset[randint(0, len(dataset)-1)]

def split_dataset(
    dataset: Dataset, 
    train_size: float = 0.7, 
    val_size: float = 0.1, 
    test_size: float = 0.2, 
    seed: int = 42
) -> tuple[Dataset, Dataset, Dataset]:
    
    # Ensure proportions sum to 1 within a small tolerance
    if not math.isclose(train_size + test_size + val_size, 1.0, rel_tol=1e-4):
        raise ValueError("Train, test, and Validation sizes must sum to 1.")

    # Set seed for deterministic results
    torch.manual_seed(seed)

    # Compute split sizes with rounding for precision
    total_size = len(dataset)
    train_len = round(train_size * total_size)
    test_len = round(test_size * total_size)

    # Ensure all samples are used (adjust validation set)
    val_len = total_size - (train_len + test_len)

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    return train_dataset, val_dataset, test_dataset

def add_btch_dim(input_dict:dict[str:torch.Tensor])-> dict[str:torch.Tensor]:
    # Add batch dimension to the input tensor
    for key in input_dict.keys():
        assert isinstance(input_dict[key], torch.Tensor), \
            f"Expected tensor for key '{key}', got {type(input_dict[key])}"
        input_dict[key] = input_dict[key].unsqueeze(0)
    return input_dict

def rm_btch_dim(input_dict:dict[str:torch.Tensor])-> dict[str:torch.Tensor]:
    # Remove batch dimension from the input tensor
    for key in input_dict.keys():
        assert isinstance(input_dict[key], torch.Tensor), \
            f"Expected tensor for key '{key}', got {type(input_dict[key])}"
        input_dict[key] = input_dict[key].squeeze(0)
    return input_dict

def collate_fn(batch:list[tuple[dict[str, torch.Tensor], torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    # Custom collate function to handle variable-sized images
    
    img_dict, masks = zip(*batch)  # Unzips the tuples into separate lists
    
    keys = img_dict[0].keys()
    batch_dict = {key: torch.stack([item[key] for item in img_dict], dim=0) for key in keys}
    batch_msk = torch.stack(masks, dim=0)

    return batch_dict, batch_msk
