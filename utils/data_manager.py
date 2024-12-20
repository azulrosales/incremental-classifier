import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iData


class DataManager(object):
    def __init__(self, known_classes):
        self._setup_data(known_classes)

    def get_dataset(self, source, mode):
        if source == "train":
            data, targets = self._train_data, self._train_targets
        elif source == "test":
            data, targets = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        return DummyDataset(data, targets, trsf, self.use_path)

    def _setup_data(self, known_classes):
        idata = iData()
        idata.load_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self._class_names = idata.class_names
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

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
        ###self._test_targets = remap_targets(self._test_targets, curr_idxs, new_idxs)

        # Map class names to the new indices
        all_classes = known_classes + new_classes
        self._class_mapping = {idx: name for idx, name in enumerate(all_classes)}

        logging.info(f"Class name mapping: {self._class_mapping}")


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def remap_targets(targets, idxs, new_idxs):
    mapping = {old: new for old, new in zip(idxs, new_idxs)}
    remapped_targets = np.vectorize(mapping.get)(targets)
    
    return remapped_targets

def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
