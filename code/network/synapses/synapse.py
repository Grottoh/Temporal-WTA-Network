from typing import Optional, Callable, Any
import torch
from contextlib import suppress

from ..component import Component
from ..weights.weights import Weights

class Synapse(Component):
    """ Passes excitation from Axon to Layer. """

    def __init__(
            self,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
    
    def set_connections(self, axons: list['Axon'], layer: 'Layer') -> None:
        self.axons = axons # The axons from which excitation is received
        self.layer = layer # The layer to which excitation is passed

        # Require that all preceding axon groups have the same shape
        if not all([axon.shape == self.axons[0].shape for axon in self.axons]):
            raise ValueError( "Incoming axons are not of the same shape:" + 
                             f" <{[axon.shape for axon in self.axons]}>.")

        # Shape preceding layers (a), succeeding layer (b), and axon weights
        self.shape_a = (len(self.axons),) + self.axons[0].shape_a
        self.shape_b = self.layer.shape
        self.shape = self.shape_b + self.shape_a

        # The axon dimensions 'occupied' respectively by layer b and layers a
        self.dims_b = tuple( [d for d in range( len(self.shape_b) )] )
        self.dims_a = tuple( [d for d in range(len(self.shape_b), 
                                               len(self.shape))] )

    def step(self) -> None:
        """ Take one step in time and update the axon accordingly. """
        super().step()
        self.excitation = torch.stack(
            [axon.excitation for axon in self.axons], 
            dim=self.dims_a[0],
            )
    

class SynapseDynamic(Synapse):
    
    def __init__(
            self,
            cdt_rest:float=0.01,
            cdt_max:float=10.0,
            cst_decay:float=0.15,
            cst_growth:float=1,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        self.cdt_rest = cdt_rest
        self.cdt_max = cdt_max
        self.cst_decay = cst_decay
        self.cst_growth = cst_growth # NOTE

    def set_connections(self, **kwargs) -> None:
        super().set_connections(**kwargs)

        self.conductances = torch.full(
            size=self.shape,
            fill_value=self.cdt_rest,
            dtype=torch.float32,
            device=self.device,
            )
    
    def set_weights(self, weights: Weights):
        self._weights = weights
    
    @property
    def weights(self) -> torch.FloatTensor:
        """ Return the actual synapse weight values. """
        return self._weights.weights
    
    @property
    def saveables(self) -> dict[str, Any]:
        """<__dict__> exluding attributes that can be saved, but shouldn't. """
        forgettables = [] # Specifies the to be forgotten attributes
        with suppress(AttributeError): forgettables.append(self.conductances)
        with suppress(AttributeError): forgettables.append(self.spikes)
        with suppress(AttributeError): forgettables.append(self.excitation)
        return self._saveables(forgettables=forgettables)
    
    def step(self) -> None:
        """ Take one step in time and update the axon accordingly. """
        super().step()

        # Determine through which axons a spike is transmitted
        self.spikes = torch.stack([axon.spikes for axon in self.axons], dim=0)
        
        self.excitation = self.excitation * self.conductances
        breakpoint
        
    def step_two(self) -> None:
        super().step_two()

        if self.layer.spiked:
            self.conductances = torch.full_like(
                self.conductances, 
                fill_value=self.cdt_rest,
                )
        else:
            
            # Determine by how much conductances decay
            decay = self.cst_decay / self._weights.iv_spike

            # Determine by how much conductances grow
            growth = self.cst_growth * torch.tensordot(
                self.spikes, 
                self.weights,
                dims=([d for d in range(self.spikes.ndim)], 
                      [d+len(self.dims_a) for d in self.dims_a]),
                )
            
            # Update the synapse conductances
            self.conductances = torch.clip(
                self.conductances - decay + growth, 
                min=self.cdt_rest, 
                max=self.cdt_max
                )
            #ccc = (torch.round(self.conductances, decimals=2)*100).to(torch.int32)
            breakpoint
    
    def learn(self) -> None:
        """ If applicable: update weights following a post-synaptic spike. """
        super().learn()
        self._weights.learn()
    
    def next(self):
        """ Called upon transitioning to a new stimulus. """
        super().next()
        self._weights.next() # NOTE: THIS WAS NOT DONE FOR LONG TIME (23-10-11)

        self.conductances = torch.full_like(
            self.conductances, 
            fill_value=self.cdt_rest,
            )