import glob
import os
import random
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.common import show_img




class DatasetLoader(Dataset):
    def __init__(self, root, phase, transform=None):
        super().__init__()
        self.transform = transform

        self.image_paths = []
        self.image_labels = []
        self.class_name = os.listdir(os.path.join(root, phase))
        self.class_name.sort()
        for i, x in enumerate(self.class_name):
            temp = sorted(glob.glob(os.path.join(root, phase, x, "*")))
            self.image_labels.extend([i] * len(temp))
            self.image_paths.extend(temp)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
            
        return {"image": image, "label": self.image_labels[index]}

    def __len__(self):
        return len(self.image_paths)
    


def get_dataloader(dataset_path, img_size=224, batch_size=128):
    
    def worker_init_fn(worker_id, seed=1):
        random.seed(worker_id+seed)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = DatasetLoader(dataset_path, 'val', transform)
    
    print(f'Load dataset, Num of dataset: {len(dataset)}')
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    # 画像を12枚表示したい場合
    # show_img(save_path='/homes/ypark/code/dataset/TransformerAnalysis/fig/ex.png', dataloader=data_loader)
    
    return data_loader

 


        

