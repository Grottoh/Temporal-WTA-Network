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
from ..axons.axon import Axon
from ..layers.traits.threshold import Threshold


class VisualizerWeights(Visualizer):

    # Base template of the probe's file name
    NFT_SAVE = "weights-source-{}_{:05d}.png" # {idx_source}_{ith_stimulus}

    def __init__(
            self,
            axon: Axon,
            sources: list[Union[LayerSensory, VisualizerWeights]],
            en_show: Union[bool, list[bool]],
            en_save: Union[bool, list[bool]],
            ks: list[ tuple[int, ...] ] = [],
            source_relation: str = "identical", # TODO: make easier to customize
            figdims: Optional[ tuple[int, int] ] = None,
            figsize: Optional[ tuple[int, int] ] = None,
            **kwargs
            ) -> None:

        # TODO: ENSURE THAT SOURCES MATCH ORDER OF <axon.weights>
        
        # Each source, as well as the combined sources, can be enabled
        if type(en_show) == bool: 
            en_show = [en_show for _ in range( len(sources)+1 )]
        if type(en_save) == bool: 
            en_save = [en_save for _ in range( len(sources)+1 )]

        # Ensure it is specified properly for each source whether to show/save
        if not len(en_show) == len(en_save) == len(sources)+1:
            raise ValueError(
                "The following does not hold: " + 
                f"len(en_show)={len(en_show)} ==" +
                f"len(en_save)={len(en_save)} ==" +
                f"len(sources)={len(sources)}."
                )
        
        # Initialize according to parent class
        super().__init__(
            component=axon,
            en_show=en_show,
            en_save=en_save, 
            **kwargs)
        
        # Visualizations are in reference to the given sources (which can be
        # traced back to the input data)
        self.sources = sources 
        
        # Count each neuron's spikes for each stimuli (resets)
        _spikes_since_stimulus = stat.SpikesSinceStimulus(
            layer=self.axon.synapse.layer,
            device=self.device,
            )
        self.key_spikes_since_stimulus = _spikes_since_stimulus.key
        _spikes_since_stimulus = self.add_statistic(_spikes_since_stimulus)

        # Relation between source layers (identical, convolutional, ...)
        self.source_relation = source_relation # TODO: make easier to customize

        # Initialize the figure dimensions and size
        self._init_figmeasures(ks=ks, figdims=figdims, figsize=figsize)
    
    @property
    def axon(self):
        """ Alternative reference to target (axon) component. """
        return self.component

    def _init_figmeasures(
            self,
            ks: list[ tuple[int, ...] ],
            figdims: Optional[ tuple[int, int] ] = None,
            figsize: Optional[ tuple[int, int] ] = None,
            ) -> None:
        """ Initialize the figure dimensions and size. """
        
        # Indices of neurons of which incoming axon weights will be visualized
        self.ks = self.axon.synapse.layer.KS if ks == [] else ks
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
        """ Convert axon weights to pixels and visualize them. """

        weight_pixels, wps_combined = self.weights_to_pixels()

        # For each source, if enabled, show and save axon weight visualizations
        for i, wps in enumerate(weight_pixels):
            self.visualize(
                weight_pixels=wps,
                idx_source=i if len(weight_pixels) > 0 else None,
                en_show=self.en_show(i), 
                en_save=self.en_save(i),
                ith_stimulus=ith_stimulus,
                n_stimuli=n_stimuli,
                )
        
        # If enabled, show and save weight visualizations of combined sources
        if not isinstance(wps_combined, type(None)): # Only if multiple sources
            self.visualize(
                weight_pixels=wps_combined,
                idx_source=None,
                en_show=self.en_show(-1),
                en_save=self.en_save(-1),
                ith_stimulus=ith_stimulus,
                n_stimuli=n_stimuli,
                )
        
        # In case this probe is a source, remember the combined weight pixels
        self.wps_combined = (weight_pixels[0] if
                             isinstance(wps_combined, type(None)) else 
                             wps_combined)

    def weights_to_pixels(self) -> tuple[ list[np.ndarray], np.ndarray ]:
        """ For each source, as well as the combined sources, transform axon
        weights into pixels which can be visualized in plots. """

        # Gather weight pixels for each source
        weight_pixels = []
        for i in range(self.axon.shape_a[0]):

            # Select subset of the weights corresponding to the selected source
            weights = torch.select(self.axon.weights, self.axon.dims_a[0], i)
            source = self.sources[i]

            # Normalize the weights such that the highest value is one
            weights = weights.clone().cpu().numpy()
            # weights = weights - np.min(weights) # Make smallest value zero
            # weight_pixels_i = weights / np.max(weights)
            weights = weights - np.min(
                self.axon.weights.clone().cpu().numpy()) # Smallest value -> 0
            weight_pixels_i = (weights / np.max(
                self.axon.weights.clone().cpu().numpy() - 
                np.min( self.axon.weights.clone().cpu().numpy() ) ))

            # If applicable, weigh&shape <weight_pixels_i> with preceding ones
            #if isinstance(source, type(self)):
            if issubclass(type(source), Visualizer):
                # NOTE: skip first of <dims_a> as this concerns <len(layers_a)>
                a = weight_pixels_i
                b = source.wps_combined
                self.weight_pixels_i = torch.tensordot(
                    a=weight_pixels_i,
                    #b=source.weight_pixels,
                    b=source.wps_combined,
                    dims=(self.axon.dims_a[1:], 
                          [d for d in range(len(self.axon.dims_a))]),
                    )
                # self.weight_pixels_i = np.tensordot(
                #     a=weight_pixels_i,
                #     #b=source.weight_pixels,
                #     b=source.wps_combined,
                #     axes=([1], 
                #           [0]),
                #     )
            
            # Hold on to the weight pixels with respect to <sources[i]>
            weight_pixels.append(weight_pixels_i)
        
        if len(weight_pixels) > 1: # If there are multiple sources, ...

            # Determine how to combine elements of <weight_pixels>
            # TODO: make easier to customize
            if self.source_relation == "identical": # Root sources identical

                # Take the mean of the sources
                wps_combined = np.array(weight_pixels).mean(0)
            
            elif self.source_relation == "inverted":

                # If src spike prob is inverted, plot black for high weights
                _weight_pixels = weight_pixels.copy()
                for i, source in enumerate(self.sources):
                    if (isinstance(source, LayerSensory) and
                        source.spike_generator.inverted):
                        _weight_pixels[i] = np.abs(_weight_pixels[i] - 1)

                # Take the mean of the sources
                wps_combined = np.array(_weight_pixels).mean(0)

            elif self.source_relation == "convolutional": # Pieced together
                raise NotImplementedError # TODO: implement
            else:
                raise ValueError(f"Source relation <'{self.source_relation}'>" +
                                f" is not known.")
            
            return weight_pixels, wps_combined
        
        # There is only one source, so weight pixels cannot be combined
        return weight_pixels, None

    def visualize(
            self, 
            weight_pixels: np.ndarray,
            idx_source: Optional[int],
            en_show: bool,
            en_save: bool,
            ith_stimulus: int,
            n_stimuli: int,
            ) -> None:
        """ 
        Create a large plot that contains for the selected axon:
            1) A row of images displaying the most recently presented stimuli
            2) A grid of images visualizing the axon weights coming into 
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
            border around the sub-plot of the neuron's incoming axon weights.
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
                (f"\nSource: {idx_source}" if idx_source != None else 
                f"\nCombined: {self.source_relation}") +
                f"\nWeights:" + 
                f" mean={torch.mean(self.axon.weights):.2f}" + 
                f" min={torch.min(self.axon.weights):.2f}" + 
                f" max={torch.max(self.axon.weights):.2f}" + 
                f" stdv={torch.std(self.axon.weights):.2f}",
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

            # Second, for each neuron, plot the incoming axon weights
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

                    # if type(self.axon.layer_b.threshold) == Threshold:
                    #     # Indicate neuron index and nr of spikes for stimulus
                    #     title = f"z_{k}: {self.spikes_since_stimulus[k]}"
                    # else:
                    #     # Indicate neuron index and nr of spikes for stimulus
                    #     title = (f"z_{k}: {self.spikes_since_stimulus[k]}\n" +
                    #             f"{self.axon.layer_b.mps_max[k]:.0f}")

                    if type(self.axon.layer_b.threshold) == Threshold:
                        # Indicate neuron index
                        title = f"z_{k[0]}"
                    else:
                        raise NotImplementedError


                    # Assign a the weight pixels of neuron <k> to a sub-plot
                    ax = fig.add_subplot(gs[row, col])
                    set_ax(image=wps, ax=ax, title=title)
                    
                    # Highlight border to indicate responsiveness of neuron <k>
                    # if proportions_spikes[k] > 0: # If <k> spiked at all ...
                    #     for spine in ax.spines.values(): # For each border ...
                    #         redness = 0.5 + proportions_spikes[k]/2
                    #         thickness = max(0.4, 1.2*proportions_spikes[k])
                    #         spine.set_edgecolor( (redness, 0, 0) )
                    #         spine.set_linewidth(thickness)
                    
                    if all([_k+1==s for _k, s in 
                            zip(k, self.axon.layer_b.shape)]):
                        break
            
        # If neither showing nor saving is enabled, do nothing
        if not en_show and not en_save:
            return
        
        # Select the appropriate source(s); <None> means to select all sources
        sources = ([self.sources[idx_source]] if 
                   idx_source != None else 
                   self.sources)

        # Initialize the overall figure and its grid layout.
        fig, gs = init_fig(ith_stimulus=ith_stimulus, n_stimuli=n_stimuli)
        
        # Visualize the source stimuli in the first row of the grid
        visualize_sources(sources=sources)

        # Visualize each neuron's incoming axon weights in the remaining rows
        visualize_axons(weight_pixels=weight_pixels)

        # If enabled, save the figure
        if en_save: 

            # Determine file name based on indices of the source and stimulus
            nf_save = self.NFT_SAVE.format(
                idx_source if idx_source != None else 'c',
                ith_stimulus,
            )

            # Save the figure, set dpi according to 'largest' input
            fig.savefig(
                self.pd_visuals+nf_save, 
                format='png', 
                dpi=np.max( 
                    [max(np.prod(source.data.shape),512) for source in sources] 
                    ),
                )
       
        # If enabled, show the figure
        if en_show: 
            plt.show()
        
        # Close the figure
        plt.close(fig)