from yaml_reader import YamlReader
import paths
from pathlib import Path
import random
import numpy

import cv2

class Annotater(YamlReader):
    _instance = None  # Class-level variable to store the single instance
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create the instance if it doesn't exist yet
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, yaml_path = f'{paths.ROOT}/data/v2_EGH455-3/data.yaml'):
        super().__init__(yaml_path)
        self.class_names = self.get_names()


    def annotate_image(self, image, labels):
        image = image.copy()
        if len(labels) == 0:
            return image
        
        for classification, (xc, yc, w, h) in labels:
        
            xc, yc, w, h = map(float, (xc, yc, w, h))

            H, W = image.shape[:2]
            x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
            x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, classification, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return image
    
    def convert_raw_label(self, raw_label):
        if len(raw_label) == 0:
            return []
        else:
            final_labels = []
            for id, p1, p2, p3, p4 in raw_label:
                final_labels.append((self.class_names[int(id)], (p1, p2, p3, p4)))
            return final_labels
        
    
    def read_labels_from_file(self, file_path):
        with open(file_path, 'r') as f:
            labels = [tuple(line.split()) for line in f if line.strip()]
            
        return self.convert_raw_label(labels)
    
    def get_test_sample(self, index = None) -> tuple[numpy.ndarray, numpy.ndarray, list]:
        IMAGE_DIR = Path("data/v2_EGH455-3/test/images")
        image_files = sorted([f for f in IMAGE_DIR.iterdir() if f.is_file()])
        labels_files = sorted([f for f in Path("data/v2_EGH455-3/test/labels").iterdir() if f.is_file()])
        
        if index:
            random_number = index
        else:
            random_number = random.randint(0, len(image_files) - 1)
            
        image_path = image_files[random_number]
        raw_image = cv2.imread(str(image_path))
        
        label_path = labels_files[random_number]
        labels = self.read_labels_from_file(label_path)
        
        annotated_image = self.annotate_image(raw_image, labels)
        
        return raw_image, annotated_image, labels

    def extract_only_labels(self, labels):
        # input: labels = [('openValve', (0.1, 0.3, 0.5, 0.1)), ('openValve', (0.1, 0.3, 0.5, 0.1))]
        # output: ['openValve', 'openValve'] 
        return [name for name, _ in labels]
            
        
        
        