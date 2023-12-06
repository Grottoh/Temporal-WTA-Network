from typing import Any, Optional, Union
import pickle
import torch
import json
import re

class Constants:
    
    PD_DATA: str = "../[00LF] data/"
    ND_CODE: str = "network/"
    ND_CODECOPY: str = "_code/"
    ND_CHECKPOINTS: str = "_checkpoints/"
    ND_VISUALS: str = "visuals/"

def save_pkl(obj: Any, filename: str) -> None:
    """ 
    Save a pickle file. 
    
    Parameters
    ----------
    obj : Any
        Object that one wants to dump
    filename:
        Filename of the created file
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    
def load_pkl(filename: str) -> Any:
    """ Load a pickle file. """
    with open(filename,'rb') as f:
        return pickle.load(f)

def get_seed(seed: Optional[int]):
    """ If no seed is given, select one at random. """
    if seed == None: # Return a random seed
        return torch.randint(int(1e9), (1,))[0].item()
    return seed # Return the given seed

def convert(name):
    """ Convert name with uppercases to one with lowercases and underscores.
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
def is_jsonable(x: Any):
    """ Return True if the given thing is JSON serializable. """
    try:
        json.dumps(x)
        return True
    except:
        return False

def where(tensor: torch.BoolTensor) -> list[ tuple[int, ...] ]:
    """ Returns indices of spots where <tensor==True>. """
    return list( zip( *[x.tolist() for x in torch.where(tensor)] ) )

def full_slice(shape: Union[tuple, torch.Size]) -> tuple:
    """ Return a tuple of slices covering the entire given shape. """
    return tuple([slice(s) for s in shape])

def list_indices(tensor: torch.Tensor) -> list[ tuple[int, ...] ]:
    """ Return a list of all indices (as tuples) of the given tensor. """

    """
    NOTE: perhaps not necessary, see:
    https://numpy.org/doc/stable/reference/generated/numpy.s_.html#numpy.s_
    https://numpy.org/doc/stable/reference/generated/numpy.ndindex.html#numpy.ndindex
    """

    def increment(index: tuple[int, ...], limit: tuple[int, ...]):
        """ Increment the index (prioritizing later dimensions). """
        if index[-1] == limit[-1]: # Increment an earlier dimension
            return increment(index=index[:-1], limit=limit[:-1]) + (0,)
        else: # Increment the final dimension
            return index[:-1] + (index[-1]+1,)
    
    # Determine the maximum index (i.e. the index of the final element)
    limit: tuple[int] = tuple( [s-1 for s in tensor.shape] )

    # Start with index (0, ...)
    indices: list[ tuple[int, ...] ] = [ tuple([0 for _ in limit]) ]

    # Keep incrementing and adding indices until the final index has been added
    while indices[-1] != limit:
        
        # Increment the index (prioritizing later dimensions), then add it
        index = increment(index=indices[-1], limit=limit)
        indices.append(index)

    return indices
