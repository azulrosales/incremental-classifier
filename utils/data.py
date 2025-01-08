import os
import torch
import logging
import numpy as np
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from shutil import copy2
from utils.toolkit import split_images_labels, remap_targets

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

    def save_test_data(self, test_imgs):
        path = 'eval-data/'
        os.makedirs(path, exist_ok=True)
        for img_path, _ in test_imgs:
            class_folder = os.path.join(path, os.path.basename(os.path.dirname(img_path)))
            os.makedirs(class_folder, exist_ok=True)
            copy2(img_path, class_folder)

    def load_test_data(self):
        path = 'eval-data/'
        if not os.path.exists(path):
            print(f"Path '{path}' does not exist. No test data loaded.")
            return []
        data = ImageFolder(root=path)
        return data.imgs

    def load_data(self, known_classes):
        path = 'data/'
        data = ImageFolder(root=path)

        try:
            train_dataset, test_dataset = random_split(data, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
        except ValueError:
            # For older PyTorch versions
            train_size = int(0.7 * len(data))
            test_size = len(data) - train_size
            train_dataset, test_dataset = random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))

        train_imgs = [data.imgs[i] for i in train_dataset.indices]
        test_imgs = [data.imgs[i] for i in test_dataset.indices]

        self._train_data, self._train_targets = split_images_labels(train_imgs)
        self._test_data, self._test_targets = split_images_labels(test_imgs) 
        
        self._class_names = data.classes

        # Print number of training samples per class
        targets, n_samples = np.unique(self._train_targets, return_counts=True)
        samples_per_class = {self._class_names[target]: count for target, count in zip(targets, n_samples)}
        print('Train samples per class:', samples_per_class)

        new_classes = [cls for cls in self._class_names if cls not in known_classes]
        curr_idxs = list(range(len(self._class_names)))

        if len(new_classes) > 0:
            last_idx = len(known_classes)
            new_idxs = list(range(last_idx, last_idx + len(new_classes)))
        else:
            new_idxs = [known_classes.index(cls) for cls in self._class_names if cls in known_classes]
            replaced_classes = [cls for cls in self._class_names if cls in known_classes]
            logging.warning(f"Knowledge for {replaced_classes} will be replaced!!!")

        # Map indices
        self._train_targets = remap_targets(self._train_targets, curr_idxs, new_idxs)
        self._test_targets = remap_targets(self._test_targets, curr_idxs, new_idxs)

        # Map class names to the new indices
        all_classes = known_classes + new_classes
        self._class_mapping = {idx: name for idx, name in enumerate(all_classes)}

        logging.info(f"Class name mapping: {self._class_mapping}")

        remapped_test_imgs = [(img_path, self._class_mapping[label]) for img_path, label in test_imgs]
        self.save_test_data(remapped_test_imgs)
