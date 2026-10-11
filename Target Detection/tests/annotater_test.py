import pytest
import random
from pathlib import Path
from annotater import Annotater
import cv2


class TestAnnotater:
    def get_label_path(self, index):
        labels_files = sorted([f for f in Path("data/v2_EGH455-3/test/labels").iterdir() if f.is_file()])
        return labels_files[index]
        
    def test_single_label(self):
        label_path = self.get_label_path(1)
        labels = Annotater().read_labels_from_file(label_path)
        print(f"Labels: {labels}")
        assert labels == [('openValve', ('0.450575', '0.5246979865771813', '0.06550999999999992', '0.06583087248322142'))]
        
    def test_no_label(self):
        label_path = self.get_label_path(5)
        labels = Annotater().read_labels_from_file(label_path)
        print(f"Labels: {labels}")
        assert labels == []
        
    def test_multi_label(self):
        labels = Annotater().read_labels_from_file(Path("tests/assets/multiple_test.txt"))
        print(f"Labels: {labels}")
        assert labels == [('openValve', ('0.40063750000000004', '0.515124832214765', '0.07832500000000002', '0.0802362416107382')), ('openValve', ('0.40063750000000004', '0.515124832214765', '0.07832500000000002', '0.0802362416107382'))]
    
    def test_random_sampler(self):
        raw_image, annotated_image, labels = Annotater().get_test_sample(3)
        
        assert 1+1 == 2
        cv2.imshow(f"Raw Image", raw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow(f"{labels}", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                
        
    