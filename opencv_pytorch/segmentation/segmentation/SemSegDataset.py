from torch.utils.data import Dataset
import os
import torch
import cv2
import numpy as np


# dataset class implementation
class SemSegDataset(Dataset):    
    """ Generic Dataset class for semantic segmentation datasets.
    Folder containing the dataset should look like:
    - data_path
    -- images_folder
    -- masks_folder
    Names of images in the images_folder and masks_folder should match
    """
    
    def __init__(
        self,
        data_path,           # Path to the dataset folder.
        images_folder,       # Path to folder containing the images
        masks_folder,        # Path to folder containing ground truth masks
        num_classes,         # Number of classes 
        transforms=None,     # Transforms!
        class_names=None,    # List of class names
        mask_ext='png',
        inference_only=False # Set to true if no corresponding ground truth masks are available
    ):

        self.num_classes = num_classes
        self.transforms = transforms
        self.class_names = class_names
        self.inference_only = inference_only
        
        # get the map of image-mask pairs  
        
        names = os.listdir(os.path.join(data_path, images_folder))
        self.dataset = []
        for name in names:
            dict_entry = {}
            dict_entry["image"] = os.path.join(data_path, images_folder, name)
            if not self.inference_only:
                mask_name = os.path.splitext(name)[0] + "." + mask_ext
                dict_entry["mask"] = os.path.join(data_path, masks_folder, mask_name)
            self.dataset.append(dict_entry)                

    def get_num_classes(self):
        """Get number of classes in the dataset"""
        return self.num_classes

    def get_class_name(self, idx):
        """Get a specific class name"""
        class_name = ""
        if self.class_names is not None and idx < len(self.num_classes):
            class_name = self.class_names[idx]
        return class_name

    # get dataset's length
    def __len__(self):
        return len(self.dataset)
    
    def pad_to_stride(self, X, stride=32):
        
        if type(X) == torch.Tensor:
            h, w = X.shape[-2:]
        else:
            h, w = X.shape[:2]

        new_h = h
        if h % stride > 0:
            new_h = h + stride - h % stride
        new_w = w
        if w % stride > 0:
            new_w = w + stride - h % stride

        lower_h, upper_h = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
        lower_w, upper_w = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
        pads = ((lower_h, upper_h), (lower_w, upper_w))        

        extra_dims = len(X.shape)-2    
                
        if type(X) == torch.Tensor:
            pads = ((0, 0),) * extra_dims + pads
        else:
            pads = pads + ((0, 0),) * extra_dims
                
        padded = np.pad(X, pads)
        
        return padded

    # get item by index
    def __getitem__(self, idx):
        sample = {
            "image": cv2.imread(self.dataset[idx]["image"])[..., ::-1],
            "file": self.dataset[idx]["image"]
        }
        if not self.inference_only:
            sample["mask"] = cv2.imread(self.dataset[idx]["mask"], 0)
        
        # apply transforms to a sample
        if self.transforms is not None:
            sample = self.transforms(**sample)
            if not self.inference_only:
                sample["mask"] = sample["mask"].long()
            
        # pad to divisble by 32 along width and height
        sample['image'] = self.pad_to_stride(sample['image'], 32)
        if not self.inference_only:
            sample['mask'] = self.pad_to_stride(sample['mask'], 32)
        
        return sample