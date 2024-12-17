import kagglehub
import numpy as np
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.toolkit import split_images_labels


def build_transform(is_train):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    return t

class iData(object):
    use_path = True
    
    train_trsf = build_transform(True)
    test_trsf = build_transform(False) 
    common_trsf = []

    class_order = np.arange(28).tolist()

    def download_data(self):
        path = 'data/'
        data = ImageFolder(root=path)

        try:
            train_dataset, test_dataset = random_split(data, [0.7, 0.3])
        except ValueError:
            # For older PyTorch versions
            train_size = int(0.7 * len(data))
            test_size = len(data) - train_size
            train_dataset, test_dataset = random_split(data, [train_size, test_size])

        train_imgs = [data.imgs[i] for i in train_dataset.indices]
        test_imgs = [data.imgs[i] for i in test_dataset.indices]
            
        self.train_data, self.train_targets = split_images_labels(train_imgs)
        self.test_data, self.test_targets = split_images_labels(test_imgs)

        self.class_names = data.classes
        targets, n_samples = np.unique(self.train_targets, return_counts=True)
        samples_per_class = {self.class_names[target]: count for target, count in zip(targets, n_samples)}
        print('Train samples per class:', samples_per_class)
