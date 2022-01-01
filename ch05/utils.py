import mxnet as mx
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import DataLoader
import numpy as np
import os
import random
import shutil
from sklearn.model_selection import train_test_split
import zipfile

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
mx.random.seed(seed)

# Image Classification
KAGGLE_DOGS_CATS_FILE = "dogs-vs-cats.zip"
EXTRACT_DOGS_CATS_FOLDER = "dogs-vs-cats"
TOTAL_NUMBER_OF_IMAGES = 25000
TOTAL_NUMBER_OF_IMAGES_PER_CLASS = 12500


def preprocess_kaggle_cats_vs_dogs(path, split_weights, random_seed=42):
    # Function that from the path to the dogs-vs-cats.zip file
    # downloaded from: https://www.kaggle.com/c/dogs-vs-cats
    # Given the split percentages for training, validation and test sets
    # extracts images as a balanced split for train, val & test from the original training set
    
    # Assert there are 3 elements in split weights array
    assert len(split_weights) == 3
    
     # Extract source file
    file_path = os.path.join(path, KAGGLE_DOGS_CATS_FILE)
    extract_path = os.path.join(path, EXTRACT_DOGS_CATS_FOLDER)
    
    # Paths for later
    train_path = os.path.join(extract_path, "train")
    val_path = os.path.join(extract_path, "val")
    test_path = os.path.join(extract_path, "test")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Delete unused files
    os.remove(os.path.join(extract_path, "sampleSubmission.csv"))
    os.remove(os.path.join(extract_path, "test1.zip"))
    
    # Extract Images from original training set
    file_path = os.path.join(extract_path, "train.zip")
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    image_files = [f for f in os.listdir(train_path)]
    
    dog_files = [f for f in image_files if f.startswith("dog")]
    cat_files = [f for f in image_files if f.startswith("cat")]

    # Verify download and extraction has gone well
    assert len(image_files) == TOTAL_NUMBER_OF_IMAGES
    assert len(dog_files) == TOTAL_NUMBER_OF_IMAGES_PER_CLASS
    assert len(cat_files) == TOTAL_NUMBER_OF_IMAGES_PER_CLASS

    # Compute files to be moved to val and test folders, according to normalized split values
    # Need to do it separately for classes, as we want to keep the datasets balanced
    dog_train_val_files, dog_test_files = train_test_split(dog_files,
                                                           test_size=split_weights[2]/sum(split_weights),
                                                           random_state=random_seed)
    cat_train_val_files, cat_test_files = train_test_split(cat_files,
                                                           test_size=split_weights[2]/sum(split_weights),
                                                           random_state=random_seed)
    
    dog_train_files, dog_val_files = train_test_split(dog_train_val_files,
                                                      test_size=split_weights[1]/sum(split_weights[:2]),
                                                      random_state=random_seed)
    cat_train_files, cat_val_files = train_test_split(cat_train_val_files,
                                                      test_size=split_weights[1]/sum(split_weights[:2]),
                                                      random_state=random_seed)
    
    # Verify Number of files matches expectations
    assert len(dog_train_files + dog_val_files + dog_test_files) == TOTAL_NUMBER_OF_IMAGES_PER_CLASS
    assert len(cat_train_files + cat_val_files + cat_test_files) == TOTAL_NUMBER_OF_IMAGES_PER_CLASS
    
    # Create Destination Folders
    dog_train_path = os.path.join(extract_path, "train/dog")
    cat_train_path = os.path.join(extract_path, "train/cat")
    dog_val_path = os.path.join(extract_path, "val/dog")
    cat_val_path = os.path.join(extract_path, "val/cat")
    dog_test_path = os.path.join(extract_path, "test/dog")
    cat_test_path = os.path.join(extract_path, "test/cat")
    
    os.makedirs(dog_train_path, exist_ok=True)
    os.makedirs(cat_train_path, exist_ok=True)
    os.makedirs(dog_val_path, exist_ok=True)
    os.makedirs(cat_val_path, exist_ok=True)
    os.makedirs(dog_test_path, exist_ok=True)
    os.makedirs(cat_test_path, exist_ok=True)
    
    # Move files as split
    for f in dog_train_files:
        src_file = os.path.join(train_path, f)
        dst_file = os.path.join(dog_train_path, f)
        shutil.move(src_file, dst_file)
        
    for f in cat_train_files:
        src_file = os.path.join(train_path, f)
        dst_file = os.path.join(cat_train_path, f)
        shutil.move(src_file, dst_file)

    for f in dog_val_files:
        src_file = os.path.join(train_path, f)
        dst_file = os.path.join(dog_val_path, f)
        shutil.move(src_file, dst_file)
        
    for f in cat_val_files:
        src_file = os.path.join(train_path, f)
        dst_file = os.path.join(cat_val_path, f)
        shutil.move(src_file, dst_file)
    
    for f in dog_test_files:
        src_file = os.path.join(train_path, f)
        dst_file = os.path.join(dog_test_path, f)
        shutil.move(src_file, dst_file)
        
    for f in cat_test_files:
        src_file = os.path.join(train_path, f)
        dst_file = os.path.join(cat_test_path, f)
        shutil.move(src_file, dst_file)
        
def generate_cats_vs_dogs_datasets(path, imageNet=False, image_size=224) -> (
    ImageFolderDataset,
    ImageFolderDataset,
    ImageFolderDataset):
    # Function that from the path to the dogs-vs-cats.zip folder
    # (as generated from the function preprocess_kaggle_cats_vs_dogs)
    # Returns DataLoader for train, val & test datasets
    
    # Paths
    extract_path = os.path.join(path, EXTRACT_DOGS_CATS_FOLDER)
    train_path = os.path.join(extract_path, "train")
    val_path = os.path.join(extract_path, "val")
    test_path = os.path.join(extract_path, "test")
        
    # Create Datasets
    train_dataset = ImageFolderDataset(train_path)
    val_dataset = ImageFolderDataset(val_path)
    test_dataset = ImageFolderDataset(test_path)
    
    transform_fn = mx.gluon.data.vision.transforms.ToTensor()
    
    if imageNet:
        transform_fn = mx.gluon.data.vision.transforms.Compose([
            mx.gluon.data.vision.transforms.Resize(image_size, keep_ratio=True),
            mx.gluon.data.vision.transforms.CenterCrop(image_size),
            mx.gluon.data.vision.transforms.ToTensor(),
            mx.gluon.data.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_fn = mx.gluon.data.vision.transforms.Compose([
            mx.gluon.data.vision.transforms.Resize(image_size, keep_ratio=True),
            mx.gluon.data.vision.transforms.CenterCrop(image_size)
        ])

    train_dataset = train_dataset.transform_first(transform_fn)
    val_dataset = val_dataset.transform_first(transform_fn)
    test_dataset = test_dataset.transform_first(transform_fn)
    
    return train_dataset, val_dataset, test_dataset
    
def generate_cats_vs_dogs_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=128) -> (
    DataLoader,
    DataLoader,
    DataLoader):
    # Function that from given datasets and batch size, computes the dataloaders
    
    # Create DataLoaders
    train_dataldr = None
    if train_dataset:
        train_dataldr = mx.gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataldr = None
    if val_dataset:    
        val_dataldr = mx.gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataldr = None
    if test_dataset:       
        test_dataldr = mx.gluon.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
    
    return train_dataldr, val_dataldr, test_dataldr

def generate_class_dict_cats_vs_dogs_imagenet() -> dict:
    # Function that returns a dict to transform from ImageNet classes to:
    # 0: Cat
    # 1: Dog
    # 2: Other
    
    # From: https://github.com/noameshed/novelty-detection/blob/master/imagenet_categories.csv
    # MIT License: https://github.com/noameshed/novelty-detection/blob/master/LICENSE.md
    
    CAT_CLASSES = [281, 283, 284, 285]
    DOG_CLASSES = list(range(151, 276))
    
    classes_dict = {}
    
    for class_idx in range(1, 10001):
        if class_idx in CAT_CLASSES:
            classes_dict[class_idx] = 0
        elif class_idx in DOG_CLASSES:
            classes_dict[class_idx] = 1
        else:
            classes_dict[class_idx] = 2
    
    return classes_dict

def rgb_to_gray(rgb):
    # Function that transforms an RGB colour image to grayscale
    return np.dot(rgb[...,:3].asnumpy(), np.array([0.2989, 0.5870, 0.1140]))
