from typing import Union, Optional, Any
import torch
import numpy as np
from contextlib import suppress

import network.utils.util as util
from ..component import Component
from ..synapses.synapse import Synapse
from ..axons.axon import Axon

class Layer(Component):
    
    """ Describes key characteristics and dynamics of a layer of neurons. """
    
    def __init__(
            self,
            shape: Union[ tuple[int, ...], list[int] ],
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        # TODO: do I want it to be listable?
        # Set the shape of the layer and the number of layer dimensions
        #self.shape = tuple(shape)

        self._set_shape(shape=shape)
    
    def _set_shape(
            self, 
            shape: Union[ tuple[int, ...], list[int] ],
            ) -> None:
        
        self.shape = tuple(shape)
        self.n_dims = len(self.shape)
        self.n_neurons = np.prod(self.shape)

        # Indicates which neurons spiked at the current timestep
        self.spikes = torch.zeros(
            self.shape, 
            dtype=torch.float32, 
            device=self.device
            )
        
        # List of all neuron indices
        self.KS = util.list_indices(tensor=self.spikes)
        
        # Contains indices of the neurons that spiked at the current timestep
        self.ks: list[ tuple[int, ...] ] = []

        # Indicates whether at least one neuron spiked at the current timestep
        self.spiked: bool = False

        # Keep track of time (resets at the onset of a new stimulus)
        self.t: int = -1
        
        # To specify synapse group that receives spikes from preceding layers
        self.synapse: Synapse = None
        
        # To contain sets of axons which transmit spikes to succeeding layers
        self.axons_b: list[Axon] = []
    
    def set_synapse(self, synapse: Synapse):
        """ Set the synapse group of the layer. """
        self.synapse = synapse
    
    def add_axon(self, axon: Axon):
        """ Add an outgoing axon group to the layer. """
        self.axons_b.append(axon)

    def next(self) -> None:
        """ Called upon transitioning to a new stimulus. """
        super().next()
        self.t = -1 # Reset internal timer
    
    def step(self) -> None:
        """ Take one step in time and update the layer accordingly. """
        super().step()
        self.t += 1 # Increment the internal timer of the sensory layer