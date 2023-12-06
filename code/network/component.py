from __future__ import annotations
from typing import Optional, Any
import torch
from typing import Union
import random
import re
import os
import numpy as np
import json
import pickle

from .utils.util import get_seed, convert, is_jsonable

class Component:
    
    """ Contains basic attributes shared by most network components. """

    def __init__(
            self,
            idle_until: Union[int, float] = 0,
            idle_after: Union[int, float] = float('inf'),
            learn_until: Union[int, float] = float('inf'),
            learn_after: Union[int, float] = 0,
            name: Optional[str] = None,
            device: str = 'cpu',
            seed: Optional[int] = None,
            **kwargs,
            ) -> None:
        
        # Ignored by the component itself, but used by the network of which it
        # is a part to determine when during a run the component is active
        self.idle_until = idle_until # Inactive until <idle_until> stimuli
        self.idle_after = idle_after # Inactive after <idle_after> stimuli
        
        # TODO: ENSURE THIS ALL WORKS AS INTENDED
        # Ignored by the component itself, but used by the network of which it
        # is a part to determine when during a run the component is active
        self.learn_until = learn_until # learn until <learn_until> stimuli
        self.learn_after = learn_after # learn after <learn_after> stimuli
        
        # Specify which device (GPU/CPU) should be used to work with the data
        self.device = device
        
        # Create a random number generator
        self.seed = get_seed(seed=seed) # Set random seed if None is given
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.seed)

        # Determine the name of the component
        # NOTE: a network will add something if it has others of the same name
        self.name = convert( type(self).__name__ ) if name == None else name
    
    def activate(self):
        """ Ensure the component is active for the entirety of any run. """
        self.idle_until = 0
        self.idle_after = float('inf')
        [value.activate() for value in self.__dict__.values() if 
         issubclass(type(value), Trait)]
    
    def is_active(self, idx_stimulus: int) -> bool:
        """ True if component is active at the given point during the run. """
        return (idx_stimulus >= self.idle_until and 
                idx_stimulus <  self.idle_after)
    
    def is_learning(self, idx_stimulus: int) -> bool:
        """ True if component is active at the given point during the run. """
        return (idx_stimulus < self.learn_until and 
                idx_stimulus >=  self.learn_after)
    
    def _saveables(self, forgettables: list[Any]) -> dict[str, Any]:
        """<__dict__> exluding attributes that can be saved, but shouldn't. """

        # Filter the forgettables out of the component dict
        saveables = {key: value for 
                     key, value in self.__dict__.copy().items() 
                     if not any( [value is x for x in forgettables] )}
        
        # Return the remaining attributes, which might be saved
        return saveables
    
    @property
    def saveables(self) -> dict[str, Any]:
        """<__dict__> exluding attributes that can be saved, but shouldn't. """
        return self._saveables(forgettables=[]) # By default forget nothing
        
    def save(
            self, 
            pd_save: str,
            nd_component: str,
            en_save_tensors: bool = True
            ) -> None:
        """ Save the network. """
            
        # Determine where to save the component, and create the directory
        pd_component = pd_save + nd_component
        os.mkdir(pd_component)

        # Save the class type of this component
        with open(pd_component+type(self).__name__+".pkl", 'wb') as f:
            pickle.dump(type(self), f)
        
        # Put jsonables, tensors, arrays, comps in separate dictionaries
        jsonables, tensors, arrays, traits = dict(), dict(), dict(), dict()

        # Iterate over all component attributes that one might want to save
        for key, value in self.saveables.items():


            # Distinguish between saveables depending on how they are saved
            # Attribute must be saved via JSON
            if is_jsonable(value):
                jsonables[key] = value

            # Attribute must be saved via torch
            elif type(value) == torch.Tensor:
                if en_save_tensors: # Only save torch tensors if enabled
                    tensors[key] = value
            
            # Attribute must be saved via numpy
            elif type(value) == np.ndarray:
                if en_save_tensors: # Only save numpy arrays if enabled
                    arrays[key] = value
            
            # Attribute must be saved as a component trait
            elif issubclass(type(value), Trait):
                traits[key] = value
            
            # Attribute is not saved
            else:
                pass
        
        # Save all JSON serializable attributes to a single .json file
        json_string = json.dumps(jsonables, indent=4)
        with open(pd_component+nd_component[:-1]+".json", "w") as f:
            f.write(json_string)
        
        # Save each torch tensor to a separate .pt file
        for key, value in tensors.items():
            torch.save(value, pd_component+key+".pt")
        
        # Save each numpy array to a separate .npy file
        for key, value in arrays.items():
            np.save(pd_component+key+".npy", value)
        
        # Save each component trait to its own sub-directory
        for key, value in traits.items():
            value.save(
                pd_save=pd_component,
                nd_component=key+"/",
                en_save_tensors=en_save_tensors, 
                )
    
    @staticmethod
    def load(
        pd_component: str, 
        en_load_tensors: bool = True, 
        host: Optional[Component] = None,
        ) -> Component:
        """ Load a component from directory <pd_component>. """
        
        # Load all saved component attributes
        type_component = None
        jsonables, tensors, arrays, traits = dict(), dict(), dict(), dict()
        for filename in os.listdir(pd_component):

            # Load the class type of the component
            if filename.endswith(".pkl"):
                with open(pd_component+filename,'rb') as f:
                    type_component = pickle.load(f)
            
            # Load the attributes that were saved via JSON
            elif filename.endswith(".json"):
                with open(pd_component+filename, 'r') as f:
                    jsonables = json.load(f)
            
            # Load the attributes that were saved via torch
            elif filename.endswith(".pt") and en_load_tensors:
                key = filename.removesuffix(".pt")
                tensors[key] = torch.load(pd_component+filename)
            
            # Load the attributes that were saved via numpy
            elif filename.endswith(".npy") and en_load_tensors:
                key = filename.removesuffix(".npy")
                arrays[key] = np.load(pd_component+filename)
            
            # Load the attributes that were saved as component traits
            elif os.path.isdir(pd_component+filename):
                key = filename
                traits[key] = pd_component+filename+"/"
        
        # Set host of the component if it has any (if so it should be a trait)
        if not host == None:
            if not issubclass(type_component, Trait):
                raise TypeError("Only traits have a host component.")
            jsonables['host'] = host
        
        # Initialize the component according to loaded JSON attributes
        component = type_component(**jsonables)
        
        # Set network tensors/arrays according to loaded torch/numpy attributes
        for key, value in (tensors|arrays).items():
            setattr(component, key, value)
        
        # Set component traits according to the loaded component traits
        for key, value in traits.items():
            trait = Component.load(
                pd_component=pd_component+key+"/",
                en_load_tensors=en_load_tensors,
                host=component,
                )
            setattr(component, key, trait)
        
        return component
    
    def on_run(self):
        """ Called on starting a new run. """
        pass # Implemented by child class
    
    def step(self) -> None:
        """ Take one step in time and update the component accordingly. """
        pass # Implemented by child class

    def step_two(self) -> None:
        # TODO: think of a better system
        pass # Implemented by child class
    
    def learn(self) -> None:
        """ Update variables according to learning dynamics. """
        pass # Implemented by child class

    def learn_two(self):
        # TODO: think of a better system
        pass # Implemented by child class

    def next(self) -> None:
        """ Called upon transitioning to a new stimulus. """
        pass # Implemented by child class

class Trait(Component):

    """
    A trait is a component that is an attribute of another component. This 
    means it can be saved/loaded when saving/loading the component of which
    it is an attribute.
    """

    def __init__(self, host: Component, **kwargs) -> None:
        super().__init__(**kwargs)

        # The component of which this trait is an attribute
        self.host = host

# class Link(Component):

#     """
#     A link is a component that is connected to other link components. This 
#     allows communication with connected links, but requires care with certain
#     functions. For example when saving or loading, or when using recursive
#     functions should as <self.activate()> (which without care lead to 
#     infinite recursion).
#     """

#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)