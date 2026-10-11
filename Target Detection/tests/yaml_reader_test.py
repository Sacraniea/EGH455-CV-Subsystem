
import pytest
from yaml_reader import YamlReader

class TestExample:
    def test_demonstrate_cwd(self):
        # Check if the current working directory is correct
        import os
        cwd = os.getcwd()
        print(f"Current working directory: {cwd}")
        assert 1 + 1 == 2
        
    def test_yaml_singleton(self):
        # Test that YamlReader is a singleton
        reader1 = YamlReader()
        reader2 = YamlReader()
        assert reader1 is reader2, "YamlReader is not a singleton"
        
    def test_print_names(self):
        names = YamlReader().get_names()
        print(f"Names: {names}")
        assert names == ['closedValve', 'gauge', 'openValve'], "Class labels do not match expected values"