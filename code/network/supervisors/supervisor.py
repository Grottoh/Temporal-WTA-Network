import torch
import math

from ..component import Component
from ..network import Network
from ..axons.axon import Axon
from ..layers.layer_sensory import LayerSensory


# TODO: decide whether I actually want this to be a component
class Supervisor(Component):

    """
    Ensure a specific group of neurons within a layer only responds to stimuli
    of a specific class.

    The supervisor is given an axon instead of the layer, such that it can
    inhibit axon excitation before it is picked up by the synapses/layer
    (code-wise). This means I can insert the supervisor in the network run loop
    instead of in the class of the layer itself.
    """
    
    def __init__(
            self,
            axon: Axon,
            layer_sensory: LayerSensory,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
        
        self.axon = axon
        self.layer_sensory = layer_sensory
    
    @property
    def layer(self):
        """ Retrieve the to be supervised layer. """
        return self.axon.layer_b
    
    def on_run(self):
        """ On starting a new run, create the supervision mask. """

        # Determine how many neurons are assigned to each class
        # TODO: allow specification of different amounts of neurons per class
        # TODO: allow n-dimensional layer shapes
        n_neurons_per_class = math.ceil(
            self.layer.shape[0] / self.layer_sensory.data.n_classes)
        
        # Create mask that allows activating neurons by the given class
        self.mask = torch.zeros(
            size=(self.layer_sensory.data.n_classes,)+self.layer.shape, 
            dtype=torch.bool, 
            device=self.device,
            )
        for label, mask in enumerate(self.mask):
            idx_start = int(label*n_neurons_per_class)
            idx_stop = idx_start + n_neurons_per_class
            mask[idx_start:idx_stop] = True
    
    def step(self):
        """ Set excitation of all neurons not corresponding to the current
        stimulus class to zero. """
        super().step()
        # TODO: take into account .learn of threshold etc.
        self.axon.excitation *= self.mask[self.layer_sensory.T_i]

class SupervisorSingleton(Supervisor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def on_run(self):
        """ On starting a new run, create the singleton supervision mask. """

        # TODO: allow n-dimensional layer shapes
        if self.layer.shape[0] != self.layer_sensory.data.n_samples:
            raise ValueError("Singleton supervisor requires that number of" +
                             f" samples <{self.layer_sensory.data.n_samples}>"+
                             f" equals its layer's number of neurons" +
                             f" <{self.layer.shape[0]}>.")
        

        # Singleton supervisor has exactly 1 neuron per stimulus
        n_neurons_per_stimulus = 1
        
        # Create mask that allows activating neurons by the given class
        self.mask = torch.zeros(
            size=(self.layer_sensory.data.n_samples,)+self.layer.shape, 
            dtype=torch.bool, 
            device=self.device,
            )
        for idx_stimulus, mask in enumerate(self.mask):
            mask[idx_stimulus] = True
    
    def step(self):
        """ Set excitation of all neurons not corresponding to the current
        stimulus class to zero. """
        #super().step()
        self.axon.excitation *= self.mask[self.layer_sensory.index]


        



