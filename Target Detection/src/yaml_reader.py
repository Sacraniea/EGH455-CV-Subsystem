import yaml

from paths import ROOT   
        
class YamlReader:
    _instance = None  # Class-level variable to store the single instance
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create the instance if it doesn't exist yet
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, yaml_path = f'{ROOT}/data/v2_EGH455-3/data.yaml'):
        self.yaml_path = yaml_path
        self.data = self.read_yaml()

    def read_yaml(self):
        with open(self.yaml_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file) or {}

    def get_names(self):
        return self.data.get('names', {})