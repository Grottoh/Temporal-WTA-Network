from typing import Callable, Optional
import torch
import numpy as np

from .weights import Weights
from ..layers.layer import Layer
from ..axons.axon import Axon

class WeightsNorm(Weights):
    """ Weights that evolve according to a local STDP rule. """

    def __init__(
            self,
            hz_timestep: int,
            hz_layer: int,
            omega_init: float = 1.0,
            omega_growth: float = 1.0,
            **kwargs
            ) -> None:
        super().__init__(
            w_min=None, 
            w_max=None, 
            eta_init=None, 
            eta_decay=None, 
            **kwargs
            )
        
        self.hz_timestep = hz_timestep
        self.hz_layer = hz_layer

        if self.hz_layer > 1000:
            raise ValueError(
                f"Number of spikes per second (<{self.hz_layer}>) cannot" +
                 " exceed the number of timesteps per second" +
                f" (<{self.hz_timestep}>)."
                )
        
        # The faster a layer spikes, the quicker its trace decays
        self.cst_decay = 1 / (self.hz_timestep/self.hz_layer)
        
        self.omega_init = omega_init
        self.omega_growth = omega_growth

    def compute_weights(self):
        #self.weights = self.omega / self.omega.sum()
        # self.weights = (self.omega / self.omega.sum() + 
        #                 self.layer_b.n_neurons**2 / self.omega.sum())

        x = self.omega.sum(self.host.dims_a)
        for _ in range(len(self.host.dims_a)):
            x = x.unsqueeze(-1)
        self.weights = (self.omega / x + 
                        self.layer_b.n_neurons**2 / self.omega.sum())
        
        breakpoint
        # self.weights = (self.omega / self.omega.sum(self.host.dims_a) + 
        #                 self.layer_b.n_neurons**2 / self.omega.sum())

    def generate_weights(self, **kwargs) -> None:
        
        # NOTE: <eta> works differently in WeightsNorm!
        # TODO: provide more detailed comments
        # Higher <omega> means weights will change more slowly
        self.omega = torch.full(
            size=self.host.shape,
            fill_value=self.omega_init,
            dtype=torch.float32,
            device=self.device,
            )
        
        self.compute_weights()

        # Initialize the trace of each axon, tracing pre-synaptic activity
        self.trace = torch.zeros(
            size=self.axon.shape_a,
            dtype=torch.float32,
            device=self.device,
            )
    
    # TODO: don't save trace?

    @property
    def axon(self) -> Axon:
        """ Alternative reference to the host component, which is an axon. """
        return self.host

    @property
    def layer_b(self) -> Layer:
        """ The post-synaptic layer. """
        return self.axon.synapse.layer
        
    def learn(self) -> None:
        """ Trace pre-synaptic spike activity and update the weights following
        a post-synaptic spike. """
        super().learn()

        
        # Traces decay over time
        self.trace = torch.clip(
            self.trace * (1 - self.cst_decay) - self.cst_decay,
            min=0,
        )

        # Strengthen traces for axons that transmit spikes
        self.trace += ( self.axon.spikes * 1/(1 + self.trace) ) #* 4
        
        if self.layer_b.spiked: # If a neuron in the post-synaptic layer spiked
            for k_b in self.layer_b.ks: # Iterate over spiking neurons <ks_b>

                self.omega[k_b] += self.trace * self.omega_growth
                #self.omega[k_b] = torch.clip(self.omega[k_b] - 1, min=1)

                # Reset activity traces
                self.trace = torch.zeros_like(self.trace)

            self.compute_weights()
    
    def next(self):
        """ Called upon transitioning to a new stimulus. """
        super().next()

        # Reset activity traces
        self.trace = torch.zeros_like(self.trace)
