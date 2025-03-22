import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F



### Designed for use with PIL images (before ToTensor)


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, img_size, patch_size):
        self.n_holes = n_holes
        self.length = img_size * patch_size
        

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        h = img.size(1)
        w = img.size(2)
        """
        _, h, w = F.get_dimensions(img)

        mask = np.ones((h, w), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h).astype(int)
            y2 = np.clip(y + self.length // 2, 0, h).astype(int)
            x1 = np.clip(x - self.length // 2, 0, w).astype(int)
            x2 = np.clip(x + self.length // 2, 0, w).astype(int)
            
            mask[y1: y2, x1: x2] = 0

        if torch.is_tensor(img):
            mask = torch.from_numpy(mask)      
            mask = mask.expand_as(img)

        else:
            mask = mask[:,:,np.newaxis]
            mask = np.tile(mask, 3)
 
        img = img * mask

        return Image.fromarray(img)
    

