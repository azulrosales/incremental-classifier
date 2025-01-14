import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from .data import iData
from ..utils.toolkit import pil_loader, split_images_labels


class DataManager(object):
    def __init__(self, known_classes):
        self._setup_data(known_classes)

    def get_dataset(self, source, mode, load_test_data=True):
        if source == "train":
            data, targets = self._train_data, self._train_targets
        elif source == "test":
            TEST_DATA_PATH = '../checkpoint/eval_data/'
            if not os.path.exists(TEST_DATA_PATH) or not load_test_data:
                data, targets = self._test_data, self._test_targets
            else:
                test_imgs = ImageFolder(root=TEST_DATA_PATH)
                data, targets = split_images_labels(test_imgs.imgs)
            print(np.unique(targets, return_counts=True))
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        return DummyDataset(data, targets, trsf)

    def _setup_data(self, known_classes):
        idata = iData()
        idata.load_data(known_classes)

        # Data
        self._train_data, self._train_targets = idata._train_data, idata._train_targets
        self._test_data, self._test_targets = idata._test_data, idata._test_targets
        self._class_names = idata._class_names
        self._class_mapping = idata._class_mapping

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(pil_loader(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label
