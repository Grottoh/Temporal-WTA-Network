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

import network.utils.util as util
import network.probes.statistic as stat
from ..utils.util import Constants as C
from .visualizer import Visualizer
from ..layers.layer_sensory import LayerSensory
from ..synapses.synapse import Synapse
from ..layers.traits.threshold import Threshold
from .visualizer_weights import VisualizerWeights

class VisualizerWeightsSynapse(Visualizer):

    # Base template of the probe's file name
    NFT_SAVE = "weights-source_{:05d}.png" # _{ith_stimulus}

    def __init__(
            self,
            synapse: Synapse,
            sources: list[Union[LayerSensory, VisualizerWeights]],
            ks: list[ tuple[int, ...] ] = [],
            figdims: Optional[ tuple[int, int] ] = None,
            figsize: Optional[ tuple[int, int] ] = None,
            **kwargs
            ) -> None:
        
        # Initialize according to parent class
        super().__init__(
            component=synapse,
            **kwargs)
        
        # Visualizations are in reference to the given sources (which can be
        # traced back to the input data)
        self.sources = sources 
        
        # Count each neuron's spikes for each stimuli (resets)
        _spikes_since_stimulus = stat.SpikesSinceStimulus(
            layer=self.synapse.layer,
            device=self.device,
            )
        self.key_spikes_since_stimulus = _spikes_since_stimulus.key
        _spikes_since_stimulus = self.add_statistic(_spikes_since_stimulus)

        # Initialize the figure dimensions and size
        self._init_figmeasures(ks=ks, figdims=figdims, figsize=figsize)
    
    @property
    def synapse(self):
        """ Alternative reference to target (synapse) component. """
        return self.component

    def _init_figmeasures(
            self,
            ks: list[ tuple[int, ...] ],
            figdims: Optional[ tuple[int, int] ] = None,
            figsize: Optional[ tuple[int, int] ] = None,
            ) -> None:
        """ Initialize the figure dimensions and size. """
        
        # Indices of neurons of which the synapse weights will be visualized
        self.ks = self.synapse.layer.KS if ks == [] else ks
        n_neurons = len(self.ks)

        # Determine the dimensions of the multi-image plot
        # NOTE: assumption right now is that the number of sources does not
        # exceed the number of columns.
        if figdims == None: # Decide dimensions based on <n_plots>
            n_cols = max( 10, math.floor(np.sqrt(n_neurons)) ) # At least 10
            n_cols = min(n_neurons, n_cols) # Except at most <n_neurons>
            n_rows = math.ceil(n_neurons/n_cols) + 1 # + 1 for sources
        else: # Use the given dimensions
            n_rows, n_cols = figdims
        self.n_rows, self.n_cols = n_rows, n_cols
        
        # If no figure size is specified, determine based on figure dimensions
        if figsize == None:
            width = (n_cols-6)*0.45
            height = (n_rows+1)*0.25
            figsize = (width, height)
        self.figsize = figsize
    
    @property
    def spikes_since_stimulus(self) -> torch.ShortTensor:
        """ Return the present value of <_spikes_since_stimulus>. """
        return self.network.statistics[self.key_spikes_since_stimulus].value
    
    def _probe(self, ith_stimulus: int, n_stimuli: int) -> None:
        """ Convert synapse weights to pixels and visualize them. """

        weight_pixels = self.weights_to_pixels()
        
        # If enabled, show and save weight visualizations of combined sources
        self.visualize(
            weight_pixels=weight_pixels,
            en_show=self.en_show(),
            en_save=self.en_save(),
            ith_stimulus=ith_stimulus,
            n_stimuli=n_stimuli,
            )

    def weights_to_pixels(self) -> np.ndarray:
        """ For each source, as well as the combined sources, transform synapse
        weights into pixels which can be visualized in plots. """

        # Normalize the weights such that the highest value is one
        weights = self.synapse.weights.clone().cpu().numpy()
        weights = weights - np.min(weights) # Smallest value -> 0
        weight_pixels = weights / np.max(weights)

        # Ensure a square shape for each neuron's synapse weights
        weight_pixels = weight_pixels.reshape( 
            (self.synapse.shape_b + 
             (np.prod(self.synapse.shape_a),) + 
             (np.prod(self.synapse.shape_a),))
            )

        return weight_pixels

    def visualize(
            self, 
            weight_pixels: np.ndarray,
            en_show: bool,
            en_save: bool,
            ith_stimulus: int,
            n_stimuli: int,
            ) -> None:
        """ 
        Create a large plot that contains for a layer:
            1) A row of images displaying the most recently presented stimuli
            2) A grid of images visualizing the synapse weights of each 
               individual neurons.
            3) Border highlights indicating the the relative responsive
               (proportion of spikes) of each neuron to the most recently
               presented stimulus.
        
        It further shows and/or saves the plot.
        """
        
        def compute_proportions_spikes() -> np.ndarray:
            """ 
            For each neuron, determine the degree to which it responded to the
            most recent stimulus, relative to other neurons in the same layer.

            These spike proportions are used to highlight neurons in the
            visualization: greater relative response means a thicker and redder
            border around the sub-plot of the neuron's synapse weights.
            """

            # Retrieve the absolute response to the most recent stimulus
            sss = self.spikes_since_stimulus.clone().cpu().numpy()

            # If not a single neuron spiked, simply return the given zeros
            if not np.any(sss > 0): 
                return sss
            
            # If at least one neuron responded, normalize the response
            proportions_spikes = sss / np.max(sss) # Highest value is 1
            
            return proportions_spikes
        
        def init_fig(
                ith_stimulus: int, 
                n_stimuli: int,
                ) -> tuple[Figure, GridSpec]:
            """ Initialize the overall figure and its grid layout. """
            
            # Initialize the figure and set its title
            fig = plt.figure(figsize=self.figsize)
            fig.suptitle(
                f"Perceived {ith_stimulus:5d} out of {n_stimuli} stimuli." +
                "\nTotal number of spikes in response to most" +
                f" recent stimulus: {self.spikes_since_stimulus.sum()}" +
                f"\nWeights:" + 
                f" mean={torch.mean(self.synapse.weights):.2f}" + 
                f" min={torch.min(self.synapse.weights):.2f}" + 
                f" max={torch.max(self.synapse.weights):.2f}" + 
                f" stdv={torch.std(self.synapse.weights):.2f}",
                y=0.99,
                fontsize=4)
            
            # Initialize grid layout of the figure (similar to <plt.subplots>)
            gs = GridSpec(
                nrows=self.n_rows, 
                ncols=self.n_cols, 
                figure=fig, 
                left=0.01, 
                right=0.99,
                bottom=0.1 / self.figsize[1],
                top=self.figsize[1] / ( self.figsize[1]+0.6 ),
                hspace=0.8
                )

            return fig, gs

        def set_ax(
                image: np.ndarray, 
                ax: Axes,
                title: str,
                fontsize: float = 2.5,
                ) -> None:
            """ Assign an image to a specific sub-plot (ax). """

            # Assign the given image to the given sub-plot (ax)
            ax.imshow(
                image, 
                cmap=plt.get_cmap('gray'), 
                vmin=0, 
                vmax=255 if np.max(image) > 1 else 1,
                )
            
            # Set thin borders for sub-plots
            [x.set_linewidth(0.3) for x in ax.spines.values()]

            # Remove sub-plot ticks
            ax.set_xticks([]) # Remove xticks
            ax.set_yticks([]) # Remove yticks

            # Set the title of the subplot
            ax.set_title(title, fontsize=fontsize, pad=1.1)
    
        def visualize_stimulus(stimulus: torch.float32) -> np.ndarray:
            # TODO: make proper (more universal) implementation (elsewhere?)
            #       (this may become necessary for non-visual stimuli)
            return stimulus.cpu().numpy()
        
        def visualize_sources(
                sources: list[Union[LayerSensory, VisualizerWeights]],
                ) -> None:
            """ Visualize the source stimuli in the first row of the grid. """

            # Determine nr of columns occupied per image of a source stimulus
            assert len(sources) <= self.n_cols # Sources must fit on one row
            n_cols_source = self.n_cols // len(sources)

            # Plot the most recently presented source stimuli
            for i, source in enumerate(sources):
                
                # Title contains certain info about the source stimulus
                row_i = source.data.df.iloc[source.index].to_dict()
                title = (f"{row_i}")
                
                # Select which first-row columns are occupied by the stimulus
                ax1 = fig.add_subplot(gs[0, i*n_cols_source:
                                            (i+1)*n_cols_source])
                
                # Visualize the source stimulus
                visual = visualize_stimulus(source.X_i)

                # Assign visual of the source stimulus to its spot in the grid
                set_ax(image=visual, ax=ax1, title=title )
        
        def visualize_axons(weight_pixels: np.ndarray) -> None:

            # Compute relative response of each neuron for most recent stimulus
            proportions_spikes = compute_proportions_spikes()

            # TODO: think of some more elegant solution for this (update_k)
            def update_k(
                    k: list[int], 
                    shape: tuple[int, ...],
                    ) -> list[int]:
                
                if len(k) != len(shape):
                    raise ValueError(f"k=<{k}> and shape=<{shape}> must" +
                                      " be of the same length.")
                
                if k[-1] == shape[-1] - 1:
                    if len(k) == 1:
                        k[0] += 1
                        return k
                    else:
                        return update_k(k=k[:-1], shape=shape[:-1]) + [0]
                else:
                    k[-1] += 1
                    return k

            # Second, for each neuron, plot the synapse weights
            _k = 0 # Plotting the <_k>th neuron (with index <ks[_k]>)
            for row in range(1, self.n_rows):
                for col in range(self.n_cols):

                    # Break if <_k> exceeds the nr of neurons to be visualized
                    if _k >= len(self.ks): 
                        break
                    else: # Retrieve the neuron index
                        k = self.ks[_k]
                        _k += 1

                    # Retrieve the weight pixels of neuron <k>
                    wps = weight_pixels[k]

                    # Add empty dimension if <wps> forms one-dimensional image
                    # TODO: remove once <visualize_stimulus> is properly
                    # implemented? It will require more than that, given that
                    # <wps> is formed in <weights_to_pixels>.
                    if wps.ndim == 1: 
                        wps = np.expand_dims(wps, axis=1)

                    # if type(self.synapse.layer.threshold) == Threshold:
                    #     # Indicate neuron index and nr of spikes for stimulus
                    #     title = f"z_{k}: {self.spikes_since_stimulus[k]}"
                    # else:
                    #     # Indicate neuron index and nr of spikes for stimulus
                    #     title = (f"z_{k}: {self.spikes_since_stimulus[k]}\n" +
                    #             f"{self.synapse.layer.mps_max[k]:.0f}")

                    if type(self.synapse.layer.threshold) == Threshold:
                        # Indicate neuron index
                        title = f"z_{k[0]}"
                    else:
                        raise NotImplementedError


                    # Assign a the weight pixels of neuron <k> to a sub-plot
                    ax = fig.add_subplot(gs[row, col])
                    set_ax(image=wps, ax=ax, title=title)
                    
                    # # Highlight border to indicate responsiveness of neuron <k>
                    # if proportions_spikes[k] > 0: # If <k> spiked at all ...
                    #     for spine in ax.spines.values(): # For each border ...
                    #         redness = 0.5 + proportions_spikes[k]/2
                    #         thickness = max(0.4, 1.2*proportions_spikes[k])
                    #         spine.set_edgecolor( (redness, 0, 0) )
                    #         spine.set_linewidth(thickness)
                    
                    if all([_k+1==s for _k, s in 
                            zip(k, self.synapse.layer.shape)]):
                        break
            
        # If neither showing nor saving is enabled, do nothing
        if not en_show and not en_save:
            return

        # Initialize the overall figure and its grid layout.
        fig, gs = init_fig(ith_stimulus=ith_stimulus, n_stimuli=n_stimuli)
        
        # Visualize the source stimuli in the first row of the grid
        visualize_sources(sources=self.sources)

        # Visualize each neuron's synapse weights in the remaining rows
        visualize_axons(weight_pixels=weight_pixels)

        # If enabled, save the figure
        if en_save: 

            # Determine file name based on indices of the source and stimulus
            nf_save = self.NFT_SAVE.format(ith_stimulus)

            # Save the figure, set dpi according to 'largest' input
            fig.savefig(
                self.pd_visuals+nf_save, 
                format='png', 
                dpi=np.max( 
                    [max(np.prod(source.data.shape),512) for
                      source in self.sources] 
                    ),
                )
       
        # If enabled, show the figure
        if en_show: 
            plt.show()
        
        # Close the figure
        plt.close(fig)