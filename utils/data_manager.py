from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iData
from utils.toolkit import pil_loader


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

        return DummyDataset(data, targets, trsf, self._use_path)

    def _setup_data(self, known_classes):
        idata = iData()
        idata.load_data(known_classes)

        # Data
        self._train_data, self._train_targets = idata._train_data, idata._train_targets
        self._test_data, self._test_targets = idata._test_data, idata._test_targets
        self._class_names = idata._class_names
        self._use_path = idata.use_path
        self._class_mapping = idata._class_mapping

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf


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
