import numpy as np
from .dataset import BaseDataset
from ..aug.segaug import crop_aug
from ...models.registry import DATASET 

@DATASET.register("DarkzurichDataset")
class DarkzurichDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        target = self.target
        return crop_aug(image, label, 512, 1024,target, min_max_height=(800, 950), w2h_ratio=2)

@DATASET.register("MsDarkzurichDataset")
class MsDarkzurichDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        target = self.target
        return crop_aug(image, label, 512, 1024,target, min_max_height=(341, 950), w2h_ratio=2)
