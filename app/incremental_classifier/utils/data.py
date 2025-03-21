import os
import torch
import numpy as np
import streamlit as st
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from shutil import copyfile
from ..utils.toolkit import split_images_labels, remap_targets, build_transform, st_log


class iData(object):
    train_trsf = build_transform(True)
    test_trsf = build_transform(False) 
    common_trsf = []

    def save_test_data(self, img_paths, targets):
        '''
        Save test data into separate directories based on class labels for evaluation.

        Args:
            img_paths (list): List of file paths for test images.
            targets (list): List of labels corresponding to the test images.
        '''
        path = '../checkpoint/eval_data/'
        os.makedirs(path, exist_ok=True)
        for img_path, target in zip(img_paths, targets):
            # Create directory for each class
            class_dir = os.path.join(path, str(target))
            os.makedirs(class_dir, exist_ok=True)
            # Copy image to the appropriate class directory
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(class_dir, img_name)
            copyfile(img_path, dest_path)

    def load_data(self, known_classes):
        '''
        Load and split image data into training and test datasets, remap targets for new classes,
        and save test data for evaluation.

        Args:
            known_classes (list): List of class names already known by the model.
        '''
        path = '../data/'
        try:
            data = ImageFolder(root=path)
        except FileNotFoundError: 
            st.error("💩 Couldn't find any class folder!")
            st.stop()

        try:
            train_dataset, test_dataset = random_split(data, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
        except ValueError:
            # For older PyTorch versions
            train_size = int(0.7 * len(data))
            test_size = len(data) - train_size
            train_dataset, test_dataset = random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))

        # Extract image paths and labels for training and test datasets
        train_imgs = [data.imgs[i] for i in train_dataset.indices]
        test_imgs = [data.imgs[i] for i in test_dataset.indices]

        self._train_data, self._train_targets = split_images_labels(train_imgs)
        self._test_data, self._test_targets = split_images_labels(test_imgs) 
        
        self._class_names = data.classes

        # Print number of training samples per class
        targets, n_samples = np.unique(self._train_targets, return_counts=True)
        samples_per_class = {self._class_names[target]: count for target, count in zip(targets, n_samples)}
        st_log(f'Train samples per class: {samples_per_class}')

        # Identify new and existing classes
        new_classes = [cls for cls in self._class_names if cls not in known_classes]
        curr_idxs = list(range(len(self._class_names)))

        if len(new_classes) > 0:
            # Assign new indices to new classes
            last_idx = len(known_classes)
            new_idxs = list(range(last_idx, last_idx + len(new_classes)))
        else:
            # Handle case when all classes are known
            new_idxs = [known_classes.index(cls) for cls in self._class_names if cls in known_classes]
            replaced_classes = [cls for cls in self._class_names if cls in known_classes]
            st_log(f"Knowledge for {replaced_classes} will be replaced!!!")

        # Map indices
        self._train_targets = remap_targets(self._train_targets, curr_idxs, new_idxs)
        self._test_targets = remap_targets(self._test_targets, curr_idxs, new_idxs)

        # Save test data for evaluation
        self.save_test_data(self._test_data, self._test_targets)

        # Map class names to the new indices
        all_classes = known_classes + new_classes
        self._class_mapping = {idx: name for idx, name in enumerate(all_classes)}

        st_log(f"Class name mapping: {self._class_mapping}")
