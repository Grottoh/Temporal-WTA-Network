from __future__ import annotations
from typing import Optional, Union
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
import os

import network.probes.statistic as stat
from ..utils.util import Constants as C
from .visualizer import Visualizer
from ..layers.layer import Layer
from ..layers.layer_sensory import LayerSensory
from ..axons.axon import Axon
from ..layers.traits.threshold import Threshold


class VisualizerSpikes(Visualizer):

    # Base template of the probe's file name
    NFT_SAVE = "spikes_{:05d}.png" # {idx_source}_{ith_stimulus}

    def __init__(
            self,
            layer: Layer,
            n_timesteps: int,
            figsize: Optional[ tuple[int, int] ] = (6, 4),
            **kwargs
            ) -> None:
        
        # Initialize according to parent class
        super().__init__(component=layer, **kwargs)

        # The amount of timesteps per stimulus
        self.n_timesteps = n_timesteps
        
        # Count each neuron's spikes for each stimuli (resets)
        _spike_times = stat.SpikeTimes(
            layer=self.layer,
            n_timesteps=self.n_timesteps,
            device=self.device,
            )
        self.key_spike_times = _spike_times.key
        _spike_times = self.add_statistic(_spike_times)

        # Determine the size of the figure
        self.figsize = figsize
    
    @property
    def layer(self):
        """ Alternative reference to target (layer) component. """
        return self.component
    
    @property
    def spike_times(self) -> torch.ShortTensor:
        """ Return the present value of <_spikes_since_stimulus>. """
        return self.network.statistics[self.key_spike_times].value
    
    def _probe(self, ith_stimulus: int, n_stimuli: int) -> None:
            
        # If neither showing nor saving is enabled, do nothing
        if not self.en_show() and not self.en_save():
            return

        # Initialize the figure and set its title
        fig = plt.figure(figsize=self.figsize)
        fig.suptitle(
            f"Layer <{self.layer.name}> spiked a total of" +
            f" {self.spike_times.sum()} times for stimulus {ith_stimulus}", 
            y=0.99, 
            fontsize=np.sum(self.figsize)
            )
        
        # Define the main axis
        ax = fig.add_axes([0.13, 0.12, 0.8, 0.8])
        ax.set_xlim(left=-1, right=self.n_timesteps)
        ax.set_ylim(bottom=-1, top=self.layer.n_neurons)
        ax.set_xticks(np.arange(0, self.n_timesteps, 1))

        # ax.set_yticks(np.arange(0, self.layer.n_neurons, 1), 
        #               [str(k) for k in self.layer.KS])
        ax.set_yticks(np.arange(0, self.layer.n_neurons, 1))

        ax.grid(visible=True, axis='x', alpha=0.3)
        ax.grid(visible=True, axis='y', alpha=0.1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Neuron index")
        fig.gca().invert_yaxis()
        
        # Scatter the spike times
        where = torch.where( self.spike_times.flatten(start_dim=1) )
        plt.scatter(
            x=where[0].cpu().numpy(),
            y=where[1].cpu().numpy(),
            marker='|',
            )

        # If enabled, save the figure
        if self.en_save(): 

            # Determine file name based on indices of the source and stimulus
            nf_save = self.NFT_SAVE.format(ith_stimulus)

            # Save the figure, set dpi according to 'largest' input
            fig.savefig(
                self.pd_visuals+nf_save, 
                format='png', 
                dpi=512,
                )
       
        # If enabled, show the figure
        if self.en_show(): 
            plt.show()
        
        # Close the figure
        plt.close(fig)