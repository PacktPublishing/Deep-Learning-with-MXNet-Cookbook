import mxnet as mx
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import DataLoader
import numpy as np
import os
import random
import shutil
from sklearn.model_selection import train_test_split
from typing import Tuple
import zipfile


# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
mx.random.seed(seed)

# Image Classification Kaggle Full
KAGGLE_DOGS_CATS_FILE = "dogs-vs-cats.zip"
EXTRACT_DOGS_CATS_FOLDER = "dogs-vs-cats"
TOTAL_NUMBER_OF_IMAGES = 25000
TOTAL_NUMBER_OF_IMAGES_PER_CLASS = 12500

# Image Classification Kaggle Light (Zenodo)
KAGGLE_DOGS_CATS_LIGHT_FILE = "cats_dogs_light.zip"
EXTRACT_DOGS_CATS_LIGHT_FOLDER = "cats_dogs_light"
TOTAL_NUMBER_OF_IMAGES_TRAIN_LIGHT = 1000
TOTAL_NUMBER_OF_IMAGES_DOGS_TRAIN_LIGHT = 545
TOTAL_NUMBER_OF_IMAGES_CATS_TRAIN_LIGHT = 455
TOTAL_NUMBER_OF_IMAGES_DOGS_TEST_LIGHT = 200
TOTAL_NUMBER_OF_IMAGES_CATS_TEST_LIGHT = 200

def preprocess_kaggle_cats_vs_dogs(path, split_weights, light=True, random_seed=42):
    # Helper function that provides access
    # to pre-processing functions for the full or light versions
    # of the cats vs dogs datasets
    if light:
        preprocess_kaggle_cats_vs_dogs_light(path, split_weights, random_seed)
    else:
        preprocess_kaggle_cats_vs_dogs_full(path, split_weights, random_seed)
        
    
def preprocess_kaggle_cats_vs_dogs_full(path, split_weights, random_seed=42):
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
    dog_train_path = os.path.join(train_path, "dog")
    cat_train_path = os.path.join(train_path, "cat")
    dog_val_path = os.path.join(val_path, "dog")
    cat_val_path = os.path.join(val_path, "cat")
    dog_test_path = os.path.join(test_path, "dog")
    cat_test_path = os.path.join(test_path, "cat")
    
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

def preprocess_kaggle_cats_vs_dogs_light(path, split_weights, random_seed=42):
    # Function that from the path to the cats_dogs_light.zip file
    # downloaded from: https://zenodo.org/record/5226945#.Y9ZYCezP3VZ
    # extracts images as a balanced split for train & val from the original training set
    # Train+Val come from 1000 samples (500 per class)
    # Test set fixed for 400 samples (200 per class)
    
    # Assert there are 2 elements in split weights array (train, val)
    assert len(split_weights) == 2
    
     # Extract source file
    file_path = os.path.join(path, KAGGLE_DOGS_CATS_LIGHT_FILE)
    extract_path = os.path.join(path)
    
    # Paths for later
    train_path = os.path.join(extract_path, EXTRACT_DOGS_CATS_LIGHT_FOLDER, "train")
    val_path = os.path.join(extract_path, EXTRACT_DOGS_CATS_LIGHT_FOLDER, "val")
    test_path = os.path.join(extract_path, EXTRACT_DOGS_CATS_LIGHT_FOLDER, "test")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # After this step, 2 folders have been generated inside "cats_dogs_light" folder, "train" and "test"
    # Inside each of these folders, there are other 2, "cats" and "dogs"

    # 1st step is to move part of the training set to the validation set, in the "val" folder
    train_val_image_files = [f for f in os.listdir(train_path)]
    
    dog_train_val_files = [f for f in train_val_image_files if f.startswith("dog")]
    cat_train_val_files = [f for f in train_val_image_files if f.startswith("cat")]

    # Verify download and extraction has gone well
    assert len(train_val_image_files) == TOTAL_NUMBER_OF_IMAGES_TRAIN_LIGHT
    assert len(dog_train_val_files) == TOTAL_NUMBER_OF_IMAGES_DOGS_TRAIN_LIGHT
    assert len(cat_train_val_files) == TOTAL_NUMBER_OF_IMAGES_CATS_TRAIN_LIGHT

    # Compute files to be moved to val folder, according to normalized split values
    # Done separately for classes
    dog_train_files, dog_val_files = train_test_split(dog_train_val_files,
                                                      test_size=split_weights[1],
                                                      random_state=random_seed)
    cat_train_files, cat_val_files = train_test_split(cat_train_val_files,
                                                      test_size=split_weights[1],
                                                      random_state=random_seed)
    
    # Verify Number of files matches expectations
    assert len(dog_train_files + dog_val_files) == TOTAL_NUMBER_OF_IMAGES_DOGS_TRAIN_LIGHT
    assert len(cat_train_files + cat_val_files) == TOTAL_NUMBER_OF_IMAGES_CATS_TRAIN_LIGHT

    # 2nd step is to re-order test set (for the right folders)
    test_image_files = [f for f in os.listdir(test_path)]
    
    dog_test_files = [f for f in test_image_files if f.startswith("dog")]
    cat_test_files = [f for f in test_image_files if f.startswith("cat")]

    assert len(dog_test_files) == TOTAL_NUMBER_OF_IMAGES_DOGS_TEST_LIGHT
    assert len(cat_test_files) == TOTAL_NUMBER_OF_IMAGES_CATS_TEST_LIGHT

    # Create Destination Folders
    dog_train_path = os.path.join(train_path, "dog")
    cat_train_path = os.path.join(train_path, "cat")
    dog_val_path = os.path.join(val_path, "dog")
    cat_val_path = os.path.join(val_path, "cat")
    dog_test_path = os.path.join(test_path, "dog")
    cat_test_path = os.path.join(test_path, "cat")
    
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
        src_file = os.path.join(test_path, f)
        dst_file = os.path.join(dog_test_path, f)
        shutil.move(src_file, dst_file)
        
    for f in cat_test_files:
        src_file = os.path.join(test_path, f)
        dst_file = os.path.join(cat_test_path, f)
        shutil.move(src_file, dst_file)

def generate_cats_vs_dogs_datasets(path, light=True, imageNet=False, image_size=224) -> Tuple[
    ImageFolderDataset,
    ImageFolderDataset,
    ImageFolderDataset]:
    # Function that from the path to the dogs-vs-cats.zip folder
    # (as generated from the function preprocess_kaggle_cats_vs_dogs)
    # Returns DataLoader for train, val & test datasets
    
    # Paths
    extract_path = os.path.join(path, EXTRACT_DOGS_CATS_FOLDER)
    if light:
        extract_path = os.path.join(path, EXTRACT_DOGS_CATS_LIGHT_FOLDER)

    train_path = os.path.join(extract_path, "train")
    val_path = os.path.join(extract_path, "val")
    test_path = os.path.join(extract_path, "test")
        
    # Create Datasets
    train_dataset = ImageFolderDataset(train_path)
    val_dataset = ImageFolderDataset(val_path)
    test_dataset = ImageFolderDataset(test_path)
    
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
    
def generate_cats_vs_dogs_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=128) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader]:
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
    return np.dot(rgb[..., :3].asnumpy(), np.array([0.2989, 0.5870, 0.1140]))
