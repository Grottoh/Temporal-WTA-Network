from typing import Union, Callable, Optional
import torch
import numpy as np

from ...component import Trait
from ..layer import Layer
from .threshold import Threshold, ThresholdDependent

class Genesis(Trait):
    
    @property
    def layer(self) -> Layer:
        """ Alternative reference to host component, which is a WTA layer. """
        return self.host

# NOTE: Resetting neurons while gathering evaluation statistics will invalidate
# the evaluation!
class GenesisX(Genesis):
    """ Neurogenesis. """

    def __init__(
            self,
            cst_redundance: int = 3,
            max_redundance: int = 1500,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)

        self.cst_redundance = cst_redundance
        self.max_redundance = max_redundance

        self.spikes_since_birth = torch.zeros(
            size=self.layer.shape,
            dtype=torch.int32,
            device=self.device,
        )

        self.redundance = torch.zeros(
            size=self.layer.shape,
            dtype=torch.int32,
            device=self.device,
        )
    
    # def step(self):
    #     super().step()
    #     self.spikes_since_birth += self.layer.spikes

    def learn(self):
        super().step()
        
        if len(self.layer.ks) > 1: # If a neuron in the layer spiked
            # print(f"Spikes from: <{self.layer.ks.tolist()}>, redundances: " +
            #       f" <{[self.redundance[k].item() for k in self.layer.ks]}>")
            for k in self.layer.ks: # Iterate over spiking neurons <ks>
                self.redundance[k] += self.cst_redundance
                if self.redundance[k] > self.max_redundance:
                    
                    for axon in self.layer.axons_a:
                        axon.weights[k] = (axon.weights[k] * 0 + 
                                           1.1*torch.mean(
                                               axon.weights[axon.weights>0]))
                        axon._weights.eta[k] = axon._weights.eta_init
                    if (not type(self.layer.threshold) == Threshold or
                        type(self.layer.threshold) == ThresholdDependent):
                        self.layer.mps_max[k] = (1.2 * torch.mean(
                            self.layer.mps_max))
                    #self.spikes_since_birth[k] = 0
                    self.redundance[k] = 0
                    print(f"Killed neuron {k} of candidates" +
                          f" {self.layer.ks}.")

                    for _k in self.layer.ks:
                        self.redundance[_k] = self.redundance[_k] // 2
                        #self.redundance[_k] = 0
                    #self.redundance = self.redundance // 2
                    break
        else:
            for k in self.layer.ks:
                # print(f"Spike from: <{self.layer.ks.tolist()}>, redundance:" +
                #       f" <{self.redundance[k].item()}>")
                self.redundance[k] = max(0, self.redundance[k] - 4)
        pass
                
