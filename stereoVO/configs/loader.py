from ruamel.yaml import YAML
from pathlib import Path
from attrdict import AttrDict

__all__ = ['yaml_parser']

def yaml_parser(path):
    """
    :param path(str):path of the yaml file relative to the root directory
    
    Returns:
        contents (AttrDict): contains the data loaded  in mememorty
    """
    
    if isinstance(path, str):
        path = Path(path)

    with open(path, 'r') as stream:
        contents = (YAML().load(stream))
            
    return AttrDict(contents)
    
