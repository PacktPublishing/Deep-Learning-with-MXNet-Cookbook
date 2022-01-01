"""
Classes for Object Detector to be evaluated
on the Penn-Fudan Pedestrian Dataset
https://www.cis.upenn.edu/~jshi/ped_html/
"""

import gluoncv as gcv
import mxnet as mx
import numpy as np
import os
import random
import re
import zipfile

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
mx.random.seed(seed)

# Constants
PEDESTRIAN_FILE = "PennFudanPed.zip"
EXTRACT_PEDESTRIAN_FOLDER = "PennFudanPed"
IMAGE_FOLDER = "PNGImages"
ANNOTATION_FOLDER = "Annotation"
MASK_FOLDER = "PedMasks"
PEDESTRIAN_IMAGES = 170

COCO_PERSON_CLASS = 15


class PedestrianDataset(mx.gluon.data.Dataset):
    """A custom dataset class to load the pedestrian dataset."""
    def __init__(self, path, is_segmentation_task=False):
        self.path = path
        self.is_segmentation_task = is_segmentation_task
        self.features, self.labels, self.masks = self.process_pedestrian()
        number_pedestrians = sum([len(label) for label in self.labels])
        print("Read " + str(len(self.features)) + " images with " + str(number_pedestrians) + " pedestrians")

    def __getitem__(self, idx):
        if not self.is_segmentation_task:
            return (self.features[idx],
                    self.labels[idx])
        else:
            return (self.features[idx],
                    self.masks[idx])

    def __len__(self):
        return len(self.features)
                        
    def process_pedestrian(self):
        """
        Function that processes the downloaded dataset file
        pedestrian.zip to generate a dataset
        """
        if not os.path.isdir(os.path.join(self.path, EXTRACT_PEDESTRIAN_FOLDER)):
            self.preprocess_pedestrian()

        return self.generate_pedestrian_test_dataset()

    def preprocess_pedestrian(self):
        """
        Function that from the path to the pedestrian.zip file
        downloaded from: https://www.cis.upenn.edu/~jshi/ped_html/
        extracts images to be used as an evaluation dataset
        """

         # Extract source file
        file_path = os.path.join(self.path, PEDESTRIAN_FILE)
        extract_path = self.path

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path) 
    
    def generate_pedestrian_test_dataset(self):
        """
        Function that from the path to the pedestrian.zip folder
        (as generated from the function preprocess_pedestrian)
        Returns images, bboxes dataset.
        If boolean parameter from class is_segmentation_dataset is True,
        segmentation masks are also processed.
        """
        seg_masks = None

        # Paths
        extract_path = os.path.join(self.path, EXTRACT_PEDESTRIAN_FOLDER)
        annotation_folder = os.path.join(extract_path, ANNOTATION_FOLDER)
        image_folder = os.path.join(extract_path, IMAGE_FOLDER)
        image_files = [f for f in os.listdir(image_folder)]
        if self.is_segmentation_task:
            mask_folder = os.path.join(extract_path, MASK_FOLDER)
            seg_masks = []

        # Verify download and extraction has gone well
        assert len(image_files) == PEDESTRIAN_IMAGES
            
        images, bboxes = [], []
        for image_file in image_files:
            
            # Image processing
            image_path = os.path.join(image_folder, image_file)
            image = mx.image.imread(image_path)
            images.append(image)
            
            # BBoxes processing
            file_name_no_ext, _ = os.path.splitext(image_file)
            anno_file_name = file_name_no_ext + ".txt"
            anno_path = os.path.join(annotation_folder, anno_file_name)
            annotations = PedestrianDataset.extract_bboxes(anno_path)
            bboxes.append(annotations)
            
            # print("From file: " + anno_file_name + ": " + str(len(annotations)) + " found.")
            
            if not self.is_segmentation_task:
                continue
                
            # Masks Processing
            mask_file_name = file_name_no_ext + "_mask.png"
            mask_path = os.path.join(mask_folder, mask_file_name)
            # Gray-scale images: flag=0
            mask_image = mx.image.imread(mask_path, flag=0)
            
            assert mx.nd.max(mask_image) == len(annotations)
            
            seg_mask = PedestrianDataset.process_mask(mask_image)
            seg_masks.append(seg_mask)            

        return images, bboxes, seg_masks

    @staticmethod
    def extract_bboxes(anno_path):
        """
        Function that reads the annotation data (Bounding Boxes)
        from an Annotation txt file in the Penn-Fudan Pedestrian dataset
        in the format [x-min, y-min, x-max, y-max]
        """
        # Open file as text
        with open (anno_path, "r") as f:
            data = f.readlines()
        
        rows_read = 0
        for current_row in data:
            if current_row.startswith("Objects with ground truth"):
                break
            rows_read += 1
        
        bboxes_in_file = int(re.findall(r"\d+", current_row)[0])
        
        bboxes_read = 0
        bboxes = []
        while bboxes_read < bboxes_in_file:
            rows_read += 1
            current_row = data[rows_read]
            if current_row.startswith("Bounding box for object"):
                bbox = [int(dim) for dim in re.findall(r"\d+", current_row)]
                
                assert bbox[0] == (bboxes_read + 1)

                bbox = bbox[1:]
                bboxes_read += 1
                bboxes.append(bbox)

        return bboxes
    
    @staticmethod
    def process_mask(mask):
        """
        Function that from an instance segmentation mask
        read from disk, with shape (H, W, 1), with different ids,
        returns a semantic segmentation mask with shape (1, H, W)
        with 
        ready to use with GluonCV functions
        """
        seg_mask = mask.copy()
        seg_mask = (seg_mask != 0)
        seg_mask = mx.nd.moveaxis(seg_mask, -1, 0)
        return seg_mask
    
    @staticmethod
    def filter_person(model_output):
        """
        Pre-trained model outputs will detect several different objects,
        including persons.
        For this dataset, we are only interested in the class "person", and
        therefore, all other classes are filtered.
        """
        PERSON_CLASS = 0
        
        class_indices = model_output[0].asnumpy()
        class_probs = model_output[1].asnumpy()
        bboxes = model_output[2].asnumpy()
        
        objects_to_delete = []
        
        for row, object_class in enumerate(class_indices[0]):
            if object_class != PERSON_CLASS:
                objects_to_delete.append(row)
                
        filtered_indices = np.delete(class_indices, objects_to_delete, 1)
        filtered_probs = np.delete(class_probs, objects_to_delete, 1)
        filtered_bboxes = np.delete(bboxes, objects_to_delete, 1)
        
        return filtered_indices, filtered_probs, filtered_bboxes
    
    @staticmethod
    def scale_bboxes(bboxes, new_image_shape, original_image_shape):
        """
        When passing an image through an MXNet Object Detection Model,
        the image needs to be transformed (scale and others). GT BBoxes needs
        to be scaled accordingly.
        """
        ratio = new_image_shape[0] / original_image_shape[0]

        new_bboxes = mx.nd.array(bboxes) * mx.nd.array([ratio] * 4)
        
        return new_bboxes

    @staticmethod
    def process_model_mask(model_mask):
        """
        Function that from the output of a GluonCV pre-trained
        semantic segmentation model, with shape (1, N, H, W),
        with N being the number of objects detected.
        returns a semantic segmentation mask with shape (1, H, W),
        with only objects classified as PERSON,
        ready to use with GluonCV functions
        """
        output_mask = model_mask.copy()
        output_mask = mx.nd.argmax(output_mask, 1)
        output_mask = (output_mask == COCO_PERSON_CLASS) * 1
        
        return output_mask
    