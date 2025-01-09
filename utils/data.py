import os
import torch
import logging
import numpy as np
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from shutil import copyfile
from utils.toolkit import split_images_labels, remap_targets, build_transform


class iData(object):
    train_trsf = build_transform(True)
    test_trsf = build_transform(False) 
    common_trsf = []

    def save_test_data(self, img_paths, targets):
        path = 'eval-data/'
        os.makedirs(path, exist_ok=True)
        for img_path, target in zip(img_paths, targets):
            class_dir = os.path.join(path, str(target))
            os.makedirs(class_dir, exist_ok=True)
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(class_dir, img_name)
            copyfile(img_path, dest_path)

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

        self.save_test_data(self._test_data, self._test_targets)

        # Map class names to the new indices
        all_classes = known_classes + new_classes
        self._class_mapping = {idx: name for idx, name in enumerate(all_classes)}

        logging.info(f"Class name mapping: {self._class_mapping}")
