from __future__ import annotations
from typing import Any, Optional, Union

import numpy as np
from network.data_loaders.data_tidigits import DataTidigits
from network.data_loaders.data_toy_temporal import DataToyTemporal
from network.probes.logger_weights import LoggerWeights
from network.probes.probe_experiment import ProbeExperiment
from network.probes.visualizer_weights import VisualizerWeights
from network.probes.visualizer_weights_synapse import VisualizerWeightsSynapse
from network.spike_generators.sg_binned import SGBinned
from network.spike_generators.sg_static import SGStatic
from network.spike_generators.sg_static_sequence import SGStaticSequence
from network.spike_generators.spike_generator import SpikeGenerator
from network.weights.weights_stdp import WeightsStdp

from network.weights.weights_synapse import WeightsSynapse

from ...network import Network
from ...layers.layer_sensory import LayerSensory
from ...layers.layer_wta import LayerWTA
from ...layers.traits.neuron import Neuron, NeuronSoftmax, NeuronStochastic
from ...weights.weights import Weights, WeightGenerator
from ...layers.traits.threshold import Threshold
from ...axons.axon import Axon
from ...synapses.synapse import Synapse, SynapseDynamic
from ...probes.logger_evaluate import LoggerEvaluate
from network.probes.visualizer_spikes import VisualizerSpikes
from ...probes.logger_spikes import LoggerSpikes
from ..arrangement import Node, Arrangement
from ...layers.traits.genesis import Genesis

class Arch1lTidigits(Network):

    def __init__(
            self,

            en_variable_duration: bool = True,

            n_timebins_max: int = 70,
            data_form: str = 'mfcc&delta',
            coefficients:np.ndarray = np.array(range(0, DataTidigits.N_MFCCS)),
            label_by: list[str] = ['utterance'],
            whitelists: list[ dict[str, list[Any]] ] = [{
                'utterance': ['o', 'z', '1', '2', '3', '4', 
                              '5', '6', '7', '8', '9'],
                }],
            blacklists: list[ dict[str, list[Any]] ] = [dict()],
            shuffle: bool = False,
            n_bins: int = 8,

            axon_w_init_min: float = 0.9990,
            axon_w_init_max: float = 0.9999,
            axon_en_weights: bool = True,
            axon_en_learn: bool = False,

            K: tuple[int, ...] = (100,),
            s_hz: float = 20.0,
            s_w_init_min: float = 0.60,
            s_w_init_max: float = 0.80,

            eta_init: float = 1.0,
            eta_decay: float = 0.60,
            eta_star: int = 25,

            en_visualize_weights: bool = True,

            pd_networks: str = Network.PD_NETWORKS + "arch1l-tidigits/",
            ndt_network: str = "arch1l-tidigits{:04d}/", # _{idx_network}

            **kwargs
            ) -> None :
        super().__init__(en_variable_duration=en_variable_duration, **kwargs)

        # Layer-0 parameters (sensory layer, data, and spike generator)
        # ---------------------------------------------------------------------
        # The maximum number of timebins (after which a stimulus is cut off)
        self.n_timebins_max = n_timebins_max

        # Determine whether to use MFCCs, their deltas, or both
        self.data_form = data_form

        # Determine which MFCC coefficients to use
        self.coefficients = coefficients

        # Determine the total number of coefficients
        self.n_coefficients = (2*len(coefficients) if 
                          self.data_form == 'mfcc&delta' else 
                          len(coefficients))
        
        # Create unique labels based on the selected data columns
        self.label_by = label_by

        # Select only data rows that satisfy the conditions of the whitelists
        self.whitelists = whitelists

        # Remove any data row that satisfies the conditions of the blacklists
        self.blacklists = blacklists

        # Whether or not the data is shuffled at the start of the run
        self.shuffle = shuffle

        # Determine the number of bins to divide MFCCs in
        self.n_bins = n_bins
        # ---------------------------------------------------------------------

        # Layer-1 parameters (single circuit WTA layer and softmax neurons)
        # ---------------------------------------------------------------------

        # The number of neurons in the WTA circuit
        self.K = K

        # Frequency at which the single WTA circuit of layer-2 produces a spike
        self.l_hz = Network.HZ_TIMESTEP / self.n_timebins_max
        # ---------------------------------------------------------------------

        # Connection parameters (axons and synapses)
        # ---------------------------------------------------------------------
        # Determine whether axons have (diverging) weights
        self.axon_en_weights = axon_en_weights

        # Minimum and maximum initial axon weights (iff <axon_en_weights>)
        self.axon_w_init_min = axon_w_init_min
        self.axon_w_init_max = axon_w_init_max

        # Whether or not axon weights change (according to STDP dynamics)
        self.axon_en_learn = axon_en_learn

        # Minimum and maximum initial weight of the synapse
        self.s_w_init_min = s_w_init_min
        self.s_w_init_max = s_w_init_max

        # Timescale at which the synapse operates
        self.s_hz = s_hz

        # Constants influencing learning rate <eta>
        self.eta_init = eta_init # Initial value of <eta>
        self.eta_decay = eta_decay # Higher means <eta> diminishes more quickly
        self.eta_star = eta_star # Learning is repeated <eta_star> times
        # ---------------------------------------------------------------------

        # Whether or not to visualize axon and synapse weights
        self.en_visualize_weights = en_visualize_weights

        # Network directories
        self.pd_networks = pd_networks # Path to where all networks are stored
        self.ndt_network = ndt_network# Directory name template of this network

    def _init_components(self):
        """ Initialize network components. """
        self._init_layer0() # Initialize sensory layer-0
        self._init_layer1() # Initialize the WTA circuit of layer-1
        self._init_connections() # Initialize connections betweens the layers
    
    def _init_layer0(self):
        """ Initialize sensory layer, including data and spike generator. """

        # Input layer is a sensory layer
        self.layer0 = LayerSensory(
            name="layer-0",
            device=self.device,
            seed=self.seed,
        )

        # Set the data and the encoding method of the input layer
        self.layer0.set_traits(

            # Data is all TIDIGITS utterances of 30 to 40 timebins
            data=DataTidigits(
                
                form=self.data_form,
                coefficients=self.coefficients,
                normalization=None,

                n_timebins_max=self.n_timebins_max,
                label_by=self.label_by,
                whitelists=self.whitelists,
                blacklists=self.blacklists,
                shuffle=self.shuffle,

                host=self.layer0,

                device=self.device,
                seed=self.seed,
            ),
            
            # MFCC data is binned and converted to spikes
            spike_generator=SGBinned(
                dim_time=1,
                n_bins=self.n_bins,

                sepzero=False,
                omit_radius=None,

                en_repeat=True,
                
                mode=SGBinned.FREQUENCY,
                host=self.layer0,

                device=self.device,
                seed=self.seed,
            ),
        )
    def _init_layer1(self):
        """ Initialize layer-1 consisting of a single WTA circuit. """

        # Layer-1 makes a final 'classification' based on input from layer-0
        self.layer1 = LayerWTA(
            mps_min=0,
            mps_rest=0,
            n_max_simultaneous_spikes=1,

            shape=self.K,
            
            idle_until=0,
            name="layer-1",
            device=self.device,
            seed=self.seed,
            )
        
        # Set the traits of WTA layer-2
        self.layer1.set_traits(

            # Layer-2 uses softmax neurons
            neuron=NeuronSoftmax(
                hz_timestep=self.HZ_TIMESTEP,
                hz_layer=self.l_hz,
                en_variable_duration=self.en_variable_duration,

                host=self.layer1,

                device=self.device,
                seed=self.seed,
                ),
            threshold=Threshold(
                mps_max=float('inf'),

                host=self.layer1,
                
                device=self.device,
                seed=self.seed,
                ),
            genesis= Genesis(
                host=self.layer1,

                device=self.device,
                seed=self.seed,
                ),
            )
    
    def _init_connections(self):
        """ Initialize the components that connect layer-0 with layer-1. """

        # Set dynamic synapses for layer-1
        self.synapse1 = SynapseDynamic(
            cdt_rest=0.00,
            cdt_max=float('inf'),
            cst_decay=0.0,
            cst_growth=1.0,

            idle_until=0,
            name="synapse-1",
            device=self.device,
            seed=self.seed,
            )
        
        # Axon from all circuits in layer-0 to the single circuit of layer-1
        self.axon0to1 = Axon(
            learn_after=(0 if self.axon_en_learn else float('inf')),
            idle_until=0,
            name="axon-0to1",
            device=self.device, 
            seed=self.seed
            )
    
    def _connect_components(self):
        """ Ensure each component has access to connected components. """
        
        # Layer-0 has access to axon-0to1 leading from layer-0 to layer-1
        self.layer0.add_axon(axon=self.axon0to1)

        # Layer1 has access to synapse1
        self.layer1.set_synapse(synapse=self.synapse1)
        
        # Axon0to1 has access to layer0, layer1, and synapse-1
        self.axon0to1.set_connections(
            layers_a=[self.layer0],
            layer_b=self.layer1,
            synapse=self.synapse1,
            )
        
        # Synapse-0 has access to axon0to1 and layer-1
        self.synapse1.set_connections(
            axons=[self.axon0to1],
            layer=self.layer1,
            )
    
    def _set_traits(self):
        """ Set the traits of network components """
        
        # Set weight trait of axon-0to1
        # ---------------------------------------------------------------------
        # Initialize axon weights trait if it isn't attached already
        if not hasattr(self.axon0to1, "_weights"):

            # Allow diverging and changing axon weights
            if self.axon_en_weights:
                weights = WeightsStdp(
                    hz_timestep=self.HZ_TIMESTEP,
                    hz_layer=self.s_hz,
                    decay_slow=0.99,
                    decay_fast=0.50,
                    cst_curve=1.0,
                    cst_beta=0.0,
                    
                    eta_init=1.0,
                    eta_decay=self.eta_decay,
                    eta_star=self.eta_star,

                    shape=self.axon0to1.shape,
                    w_min=0.0,
                    
                    host=self.axon0to1,
                    device=self.device,
                    seed=self.seed,
                    )
            
            # Use singleton axon weight of 1.0 that does not change
            else:
                weights = Weights(
                    shape=(1,),
                    w_min=1.0,
                    w_max=1.0,
                    eta_init=None,
                    eta_decay=None,
                    eta_star=1,
                    
                    host=self.axon0to1,
                    device=self.device,
                    seed=self.seed,
                )
                # NOTE: may not work with saving and loading
                weights.weights=1.0
            
            # Attach the weight trait to axon-0to1i
            self.axon0to1.set_weights(weights)
        
        # Generate new weights for the weights trait of axons-0to1
        if self.axon_en_weights:
            self.axon0to1._weights.generate_weights(
                weight_generator=lambda **kwargs: WeightGenerator.uniform(
                    init_min=self.axon_w_init_min,
                    init_max=self.axon_w_init_max,
                    relative=True,
                    **kwargs
                    ),
                )
        # ---------------------------------------------------------------------

        # Set weight trait of synapse in synapse-1
        # ---------------------------------------------------------------------
        # Initialize synapse weights trait if it isn't attached already
        if not hasattr(self.synapse1, "_weights"):
            weights = WeightsSynapse(
                hz_timestep=self.HZ_TIMESTEP,
                hz_layer=self.s_hz,
                cst_curve=1.0,

                shape=(self.synapse1.shape_b+
                       self.synapse1.shape_a+
                       self.synapse1.shape_a),
                eta_init=self.eta_init,
                eta_decay=self.eta_decay,
                eta_star=self.eta_star,
                
                name=f"weights_{self.synapse1.name}",
                host=self.synapse1,
                device=self.device,
                seed=self.seed,
                )
            self.synapse1.set_weights(weights)
        
        # Generate new weights for the weights trait of synapse-1i
        self.synapse1._weights.generate_weights(
            weight_generator=lambda **kwargs: WeightGenerator.uniform(
                init_min=self.s_w_init_min,
                init_max=self.s_w_init_max,
                relative=True,
                **kwargs
                ),
            )
        # ---------------------------------------------------------------------
    
    def _init_arrangement(self):
        """ Determine the manner in which network components are arranged. """

        # First input spike trains are produced by sensory layer-0
        node_0_0 = Node(position=(0, 0), component=self.layer0)
        
        # Axon-0to1 transmits spikes from sensory layer-0 to WTA layer-1
        node_1_0 = Node(position=(1, 0), component=self.axon0to1)

        # Synapse-1 additionally weighs spikes travelling to WTA layer-1
        node_2_0 = Node(position=(2, 0), component=self.synapse1) 

        # WTA layer-1 reponds to incoming spike patterns from sensory layer-0
        node_3_0 = Node(position=(3, 0), component=self.layer1)
        
        # The arrangement of network components determines processing order
        self.arrangement = Arrangement(
            nodes=([node_0_0, node_1_0, node_2_0, node_3_0]),
            )

    def _init_probes(self) -> None :
        """ Initialize probes that monitor network activity. """
        super()._init_probes()

        # Create a list to contain the probes
        self.probes = []

        # Log spike counts
        # ---------------------------------------------------------------------
        # Log spike-count statistics of WTA layer 1
        logger_spikes_layer1 = LoggerSpikes(
            layer=self.layer1,
            aggregation_period=min(100, self.iv_probes),

            loggers=[self.logger],
            en_local_logging=True,
            
            network=self,
            en_show=True,
            en_save=True,
            iv_probe=self.iv_probes,
            device=self.device,
            )
        self.probes.append(logger_spikes_layer1)
        # ---------------------------------------------------------------------
        
        # Track the count of (non-)zero weights of the synapses
        # ---------------------------------------------------------------------        
        # Log weights of synapse-1
        logger_weights_synapse1 = LoggerWeights(
            weights=self.synapse1._weights,

            loggers=[self.logger],
            en_local_logging=True,
            
            network=self,
            en_show=True,
            en_save=True,
            iv_probe=(float('inf') if 
                      self.mode_run in [self.TEST, 
                                        self.MAP, 
                                        self.EVALUATE] else 
                      self.iv_probes),
            device=self.device,
            )
        self.probes.append(logger_weights_synapse1)
        # ---------------------------------------------------------------------

        # Visualize spikes
        # ---------------------------------------------------------------------
        # Visualize spikes of sensory layer-0
        visualizer_spikes_layer0 = VisualizerSpikes(
            layer=self.layer0,
            n_timesteps=self.layer0.spike_generator.n_timesteps,

            en_show=False,
            en_save=True,

            network=self,
            iv_probe=self.iv_probes,
            device=self.device,
            )
        self.probes.append(visualizer_spikes_layer0)

        # Visualize spikes of WTA layer-1
        visualizer_spikes_layer1 = VisualizerSpikes(
            layer=self.layer1,
            n_timesteps=self.layer0.spike_generator.n_timesteps,

            en_show=False,
            en_save=True,

            network=self,
            iv_probe=self.iv_probes,
            device=self.device,
            )
        self.probes.append(visualizer_spikes_layer1)
        # ---------------------------------------------------------------------

        # If diverging axon weights are enabled, visualize them
        # ---------------------------------------------------------------------
        if self.axon_en_learn and self.en_visualize_weights: # If enabled ...
                visualizer_weights_axon0to1 = VisualizerWeights(
                    axon=self.axon0to1,
                    sources=self.axon0to1.layers_a,
                    en_show=False,
                    en_save=True,

                    network=self,
                    iv_probe=self.iv_probes,
                    device=self.device,
                    
                    figsize=(5, 2),
                    )
                self.probes.append(visualizer_weights_axon0to1)
        # ---------------------------------------------------------------------

        # Visualize synapse weights
        # ---------------------------------------------------------------------
        if self.en_visualize_weights: # If enabled ...
            visualizer_weights_synapse1 = VisualizerWeightsSynapse(
                synapse=self.synapse1,
                sources=self.axon0to1.layers_a, # TODO: what are true sources?
                en_show=False,
                en_save=True,

                network=self,
                iv_probe=self.iv_probes,
                device=self.device,
                
                figsize=(5, 2),
                )
            self.probes.append(visualizer_weights_synapse1)
        # ---------------------------------------------------------------------

        # Evaluate performance
        # ---------------------------------------------------------------------
        # If it is a <TEST> or <MAP> run, evaluate WTA layer-1
        if self.mode_run in [self.TEST, self.MAP]:
            self.logger_evaluate_layer1 = LoggerEvaluate(
                layer=self.layer1,
                layer_sensory=self.layer0,
                n_past=self.n_stimuli,

                loggers=[self.logger],
                en_local_logging=True,
                
                network=self,
                en_show=True,
                en_save=True,
                iv_probe=self.iv_probes,
                device=self.device,
            )
            self.probes.append(self.logger_evaluate_layer1)
        
        # If this is an <EVALUATE> run, use the evaluator of which the mapping
        # has been decided during the preceding <MAP> run
        if self.mode_run == self.EVALUATE:
            for evaluator in self.evaluators:
                evaluator.loggers.append(self.logger)
            self.probes = self.probes + self.evaluators
        # ---------------------------------------------------------------------
        
        if self.mode_run in [self.TEST, self.EVALUATE]:
            # Summarize and save some results
            self.probe_experiment = ProbeExperiment(
                probes=self.probes,

                component=self,
                network=self,
                en_show=True,
                en_save=True,
                iv_probe=float('inf'),
                device=self.device,
                )
            self.probes.append(self.probe_experiment)
        
    @classmethod
    def run_network(
        cls,
        network: Arch1lTidigits,

        n_unique_trn: int,
        n_cycles_trn: int,

        n_unique_tst: int,
        n_unique_map: int,
        n_cycles_tst: int,
        
        en_train: bool = True,
        data_subsets_trn: list[str] = ['train'],
        n_probings_trn: int = 10,
        
        en_test: bool = True,
        data_subsets_map: list[str] = ['train'],
        data_subsets_tst: list[str] = ['test'],
        n_probings_tst: int = 10,

        en_save: bool = True,
        en_save_logs: bool = True,
        en_save_visuals: bool = True,
        en_save_network: bool = True,
        en_save_tensors: bool = False,
        ) -> None:
        Network.run_network(
            network=network,

            en_train=en_train,
            data_subsets_trn=data_subsets_trn,
            n_unique_trn=n_unique_trn,
            n_cycles_trn=n_cycles_trn,
            n_probings_trn=n_probings_trn,

            en_test=en_test,
            data_subsets_map=data_subsets_map,
            data_subsets_tst=data_subsets_tst,
            n_unique_map=n_unique_map,
            n_unique_tst=n_unique_tst,
            n_cycles_tst=n_cycles_tst,
            n_probings_tst=n_probings_tst,

            en_save=en_save,
            en_save_logs=en_save_logs,
            en_save_visuals=en_save_visuals,
            en_save_network=en_save_network,
            en_save_tensors=en_save_tensors,
            )
    
    @classmethod
    def run_all_default(cls, device='cpu', seed=2023):
        """ 
        Create and run a network on all single-digit utterances. 
        """

        # Number of unique train and test stimuli
        N_UNIQUE_TRAIN = 3586
        N_UNIQUE_TEST = 3586

        # How many times to observe the entire train set
        N_CYCLES_TRN = 5

        # Create the network
        network = cls._create(
            ndt_network="all-default_{:04d}/",

            device=device,
            seed=seed,
            )
        
        # Train and test the network
        cls.run_network(
            network=network,

            n_unique_trn=N_UNIQUE_TRAIN,
            n_cycles_trn=N_CYCLES_TRN,

            n_unique_map=N_UNIQUE_TRAIN,
            n_unique_tst=N_UNIQUE_TEST,
            n_cycles_tst=1,
            )

    @classmethod
    def run_all_custom(cls):
        """ 
        Create and run a network on all single-digit utterances. 
        """

        # Number of unique train and test stimuli
        N_UNIQUE_TRAIN = 3586
        N_UNIQUE_TEST = 3586

        # How many times to observe the entire train set
        N_CYCLES_TRN = 10

        # Create the network
        network = cls._create(

            n_timebins_max=70,
            data_form='mfcc&delta',
            #label_by=['dialect'],
            shuffle=False,
            n_bins=16,

            axon_w_init_min=0.9990,
            axon_w_init_max=0.9999,
            axon_en_weights=True,
            axon_en_learn=False,

            K=(100,),
            s_hz=20.0,
            s_w_init_min=0.90,
            s_w_init_max=0.99,

            eta_init=1.0,
            eta_decay=0.60,
            eta_star=25,

            en_visualize_weights=True,

            ndt_network="all-custom_{:04d}/",

            device='cpu',
            seed=2023,
            )
        
        # Train and test the network
        cls.run_network(
            network=network,

            n_unique_trn=N_UNIQUE_TRAIN,
            n_cycles_trn=N_CYCLES_TRN,
            n_probings_trn=10,

            n_unique_map=N_UNIQUE_TRAIN,
            n_unique_tst=N_UNIQUE_TEST,
            n_cycles_tst=1,
            )



        




        


    

        

