import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from datetime import datetime


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def accuracy(y_pred, y_true, data_manager):
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
    cm = confusion_matrix(y_true, y_pred.T[0])
    class_labels = [data_manager._class_mapping[i] for i in range(len(cm))]

    plt.figure(figsize=(18, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    if not os.path.exists('confusion_matrices'):
        os.makedirs('confusion_matrices')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"confusion_matrices/confusion_matrix_{timestamp}.png")
    plt.close()

def split_images_labels(imgs):
    '''
    Splits trainset.imgs in ImageFolder
    '''
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def get_image_hash(img_path):
    '''
    Generates a hash for an image based on its pixel data
    '''
    with Image.open(img_path) as img:
        return hashlib.md5(img.tobytes()).hexdigest()

def merge_img_lists(known_imgs, new_imgs):
    '''
    Combines two lists of images while removing duplicates
    '''
    seen_hashes = {get_image_hash(img[0]) for img in known_imgs}
    unique_new_imgs = [img for img in new_imgs if get_image_hash(img[0]) not in seen_hashes]
    return known_imgs + unique_new_imgs

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

def st_log(content, color='#a6a6b3', font_family="'Courier New'"):
    st.markdown(
        f"""
        <p style="font-family: {font_family}; color: {color};">
            {content}
        </p>
        """,
        unsafe_allow_html=True
    )