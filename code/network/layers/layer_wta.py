from __future__ import annotations
from typing import Union, Callable, Optional, Any
import torch
from contextlib import suppress

from .layer import Layer
from .traits.threshold import Threshold, ThresholdDependent
from .traits.genesis import Genesis
from ..axons.axon import Axon
from .traits.neuron import Neuron

class LayerWTA(Layer):
    """ 
    Winner-Take-All (WTA) layer: spike of each neuron inhibits the others.
    """

    def __init__(
            self,
            mps_min: Union[float, torch.FloatTensor],
            mps_rest: Union[float, torch.FloatTensor],
            n_max_simultaneous_spikes: int,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        # Specify the minimum, maximum, and resting potential
        self.mps_min = mps_min
        self.mps_rest = mps_rest
        
        # Determines how many simultaneous spikes are allowed
        self.n_max_simultaneous_spikes = n_max_simultaneous_spikes

        # Initialize the membrane potential of each neuron at resting potential
        self.mps = torch.full(
            size=self.shape, 
            fill_value=self.mps_rest, 
            dtype=torch.float32, 
            device=self.device
            )
    
    def set_traits(
            self,
            neuron: Optional[Neuron] = None,
            threshold: Optional[Threshold] = None,
            genesis: Optional[Genesis] = None, 
            ) -> None:
        """ Set traits of the layer. Passing <None> will skip the trait. """
        
        # Set the neuron trait, determining spiking behaviour of layer neurons
        if issubclass(type(neuron), Neuron): # Set specific neuron class
            self.neuron = neuron
        elif neuron != None: # Given value is invalid (non-neuron) type
            raise TypeError(
                f"Trait <neuron> must be of type <{Neuron}>," +
                f" but <{type(neuron)}> was given."
                )
        
        # Set the threshold trait, determining value and evolution of max mps
        if issubclass(type(threshold), Threshold): # Set threshold class
            self.threshold = threshold
        elif threshold != None: # Given value is invalid (non-threshold) type
            raise TypeError(
                f"Trait <threshold> must be of type <{Threshold}>," +
                f" but <{type(threshold)}> was given."
                )
        
        # Set the genesis trait, determining when neurons die and are reborn
        if issubclass(type(genesis), Genesis): # Set specific genesis class
            self.genesis = genesis
        elif genesis != None: # Given value is invalid (non-genesis) type
            raise TypeError(
                f"Trait <genesis> must be of type <{Genesis}>," +
                f" but <{type(genesis)}> was given."
                )
    
    @property
    def saveables(self) -> dict[str, Any]:
        """<__dict__> exluding attributes that can be saved, but shouldn't. """
        forgettables = [] # Specifies the to be forgotten attributes
        with suppress(AttributeError): forgettables.append(self.mps)
        with suppress(AttributeError): forgettables.append(self.spikes)
        with suppress(AttributeError): forgettables.append(self.ks)
        with suppress(AttributeError): forgettables.append(self.spiked)
        return self._saveables(forgettables=forgettables)
    
    @property
    def mps_max(self) -> Union[float, torch.FloatTensor]:
        """ Specifies the maximum membrane potential of each neuron. """
        return self.threshold.mps_max

    @property
    def axons_a(self) -> list[Axon]:
        """ The axon groups transmitting signals to this layer. """
        return self.synapse.axons
    
    def step(self) -> None:
        """ Take one step in time and update the layer accordingly. """
        super().step()


        # Update the membrane potentials according to incoming excitation
        self.excitation = self.synapse.excitation.sum(self.synapse.dims_a)
        self.mps += self.excitation

        # Clip value (separately because one may be scalar, other tensor)
        self.mps = self.mps.clip(min=self.mps_min)
        self.mps = self.mps.clip(max=self.mps_max)

        self.neuron.step() # Determine which neurons spike
        if self.spiked: # If at least one neuron spiked ...
            self.inhibit() # Perform lateral inhibition
    
    def learn(self):
        super().learn()
        self.threshold.learn()
        self.genesis.learn()
    
    def learn_two(self):
        super().learn()
        self.threshold.learn_two()
        self.genesis.learn_two()

    def next(self):
        """ Called upon transitioning to a new stimulus. """
        super().next()
        self.inhibit() # Reset membrane potentials to their resting state
        self.threshold.next()
    
    def inhibit(self):
        """ Simple form of lateral inhibition: the membrane potential of each
        neuron is reset to its resting potential. """
        if type(self.mps_rest) == torch.Tensor:
            self.mps = self.mps_rest.copy()
        else: # Assume <mps_rest> is of type <float> or <int>
            self.mps = torch.full_like(self.mps, fill_value=self.mps_rest)
        