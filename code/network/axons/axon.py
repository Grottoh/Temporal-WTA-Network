from typing import Optional, Callable, Any
import torch
from contextlib import suppress

from ..component import Component
from ..weights.weights import Weights

class Axon(Component):
    """ Passes excitation from Layers to Synapse. """

    def __init__(
            self,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
    
    def set_connections(
            self, 
            layers_a: list['Layer'], 
            layer_b: 'Layer',
            synapse: 'Synapse',
            ):

        # Set the layers from which the axons originate
        self.layers_a = layers_a

        # Set the layer to which the axons lead (via the synapses)
        self.layer_b = layer_b

        # Assert that all preceding layers have the same shape
        assert all(layer.shape == layers_a[0].shape for layer in layers_a)

        # Shape preceding layers (a), succeeding layer (b), and axon weights
        self.shape_a = (len(self.layers_a),) + self.layers_a[0].shape
        self.shape_b = layer_b.shape
        self.shape = self.shape_b + self.shape_a

        # The axon dimensions 'occupied' respectively by layer b and layers a
        self.dims_b = tuple( [d for d in range( len(self.shape_b) )] )
        self.dims_a = tuple( [d for d in range(len(self.shape_b), 
                                               len(self.shape))] )
        
        # Set the synapse group to which the axon group leads.
        self.synapse = synapse
    
    def set_weights(self, weights: Weights):
        self._weights = weights
    
    @property
    def weights(self) -> torch.FloatTensor:
        """ Return the actual axon weight values. """
        return self._weights.weights
    
    @property
    def saveables(self) -> dict[str, Any]:
        """<__dict__> exluding attributes that can be saved, but shouldn't. """
        forgettables = [] # Specifies the to be forgotten attributes
        with suppress(AttributeError): forgettables.append(self.spikes)
        with suppress(AttributeError): forgettables.append(self.excitation)
        return self._saveables(forgettables=forgettables)
        
    def next(self) -> None:
        """ Called upon transitioning to a new stimulus. """
        super().next()
        self._weights.next()
    
    def step(self) -> None:
        """ Take one step in time and update the axon accordingly. """
        super().step()

        # Determine through which axons a spike is transmitted
        self.spikes = torch.stack(
            [layer.spikes for layer in self.layers_a], 
            dim=0,
            )
        
        self.excitation = self.spikes * self.weights


    def learn(self) -> None:
        """ If applicable: update weights following a post-synaptic spike. """
        super().learn()
        self._weights.learn()