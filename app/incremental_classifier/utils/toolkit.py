import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix


def count_parameters(model, trainable=False):
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): The model to count parameters in.
        trainable (bool): Whether to count only trainable parameters. Default is False.

    Returns:
        int: The number of parameters in the model.
    """
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def tensor2numpy(x):
    """
    Converts a PyTorch tensor to a NumPy array.

    Args:
        x (torch.Tensor): The tensor to convert.

    Returns:
        numpy.ndarray: The converted NumPy array.
    """
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def accuracy(y_pred, y_true, data_manager):
    """
    Calculates accuracy for each class and the overall accuracy.

    Args:
        y_pred (numpy.ndarray): The predicted labels.
        y_true (numpy.ndarray): The true labels.
        data_manager (DataManager): The data manager that holds the class mapping.

    Returns:
        dict: A dictionary containing the overall accuracy and per-class accuracy.
    """
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}

    # Total accuracy
    all_acc["Total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Per class accuracy
    unique_classes = np.unique(y_true)
    for class_id in unique_classes:
        idxes = np.where(y_true == class_id)[0]  
        class_name = data_manager._class_mapping[class_id]
        all_acc[class_name] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    return all_acc

def generate_confusion_matrix(y_true, y_pred, data_manager):
    """
    Generates and saves a confusion matrix plot.

    Args:
        y_true (numpy.ndarray): The true labels.
        y_pred (numpy.ndarray): The predicted labels.
        data_manager (DataManager): The data manager that holds the class mapping.
    """
    cm = confusion_matrix(y_true, y_pred.T[0])
    class_labels = [data_manager._class_mapping[i] for i in range(len(cm))]

    plt.figure(figsize=(18, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    path = '../checkpoint/'
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/confusion_matrix.png")
    plt.close()

def split_images_labels(imgs):
    """
    Splits the `imgs` attribute from ImageFolder into two arrays: images and labels.

    Args:
        imgs (list): List of tuples containing image paths and labels.

    Returns:
        tuple: A tuple containing two numpy arrays: images and labels.
    """
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def get_image_hash(img_path):
    """
    Generates a hash for an image based on its pixel data.

    Args:
        img_path (str): Path to the image file.

    Returns:
        str: MD5 hash of the image.
    """
    with Image.open(img_path) as img:
        return hashlib.md5(img.tobytes()).hexdigest()

def merge_img_lists(known_imgs, new_imgs):
    """
    Combines two lists of images while removing duplicates based on image hashes.

    Args:
        known_imgs (list): List of known images.
        new_imgs (list): List of new images to merge with the known images.

    Returns:
        list: The combined list of images with duplicates removed.
    """
    seen_hashes = {get_image_hash(img[0]) for img in known_imgs}
    unique_new_imgs = [img for img in new_imgs if get_image_hash(img[0]) not in seen_hashes]
    return known_imgs + unique_new_imgs

def remap_targets(targets, idxs, new_idxs):
    """
    Remaps target labels from old indices to new indices.

    Args:
        targets (numpy.ndarray): The original target labels.
        idxs (list): List of old indices.
        new_idxs (list): List of new indices.

    Returns:
        numpy.ndarray: The remapped target labels.
    """
    mapping = {old: new for old, new in zip(idxs, new_idxs)}
    remapped_targets = np.vectorize(mapping.get)(targets)
    
    return remapped_targets

def pil_loader(path):
    """
    Loads an image from a given path using PIL.

    Args:
        path (str): Path to the image file.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def build_transform(is_train):
    """
    Builds a set of image transformations for data augmentation or evaluation.

    Args:
        is_train (bool): Whether the transformations are for training or testing.

    Returns:
        list: A list of transformations to apply to the images.
    """
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

def st_log(content, color='#a6a6b3', font_family="'Courier New'"):
    """
    Logs content in Streamlit with custom styling.

    Args:
        content (str): The content to display in the Streamlit app.
        color (str): The color of the text. Default is '#a6a6b3'.
        font_family (str): The font family to use. Default is "'Courier New'".
    """
    st.markdown(
        f"""
        <p style="font-family: {font_family}; color: {color};">
            {content}
        </p>
        """,
        unsafe_allow_html=True
    )