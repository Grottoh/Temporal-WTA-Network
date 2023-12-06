from __future__ import annotations
from typing import Optional
from network.data_loaders.data_tidigits import DataTidigits
from network.data_loaders.data_toy_repeat import DataToyRepeat
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

class Arch1lRepeat(Network):

    # Determines the interval at which softmax neurons spike
    HZ_SOFTMAX = Network.HZ_TIMESTEP/20 # (Iff type_neuron==NeuronSoftmax)

    def __init__(
            self,
            
            inverted: bool = False,
            shuffle: bool = False,

            K: tuple[int, ...] = (4,),

            en_axon_weights: bool = True,
            eta_init: float = 1.0,
            eta_decay: float = 1.0,
            eta_star: float = 25,

            pd_networks: str = Network.PD_NETWORKS + "arch1l-repeat/",
            ndt_network: str = "arch1l-repeat_{:04d}/", # _{idx_network}

            **kwargs
            ) -> None :
        super().__init__(**kwargs)

        # Layer-0 parameters (sensory layer, data, and spike generator)
        # ---------------------------------------------------------------------

        # If True, the data is inverted (0->1 and 1->0)
        self.inverted = inverted

        # Whether or not the data is shuffled at the start of the run
        self.shuffle = shuffle
        # ---------------------------------------------------------------------

        # Layer-1 parameters (WTA layer and neurons)
        # ---------------------------------------------------------------------

        # The number of neurons in the WTA circuit
        self.K = K
        # ---------------------------------------------------------------------

        # Connection parameters (axons and synapses)
        # ---------------------------------------------------------------------

        # Determine whether axons have (diverging) weights
        self.en_axon_weights = en_axon_weights

        # Constants influencing learning rate <eta>
        self.eta_init = eta_init # Initial value of <eta>
        self.eta_decay = eta_decay # Higher means <eta> diminishes more quickly
        self.eta_star = eta_star # Learning is repeated <eta_star> times
        # ---------------------------------------------------------------------

        # Network directories
        self.pd_networks = pd_networks # Path to where all networks are stored
        self.ndt_network = ndt_network# Directory name template of this network

    def _init_components(self):
        """ Initialize network components. """
        self._init_layer0() # Initialize sensory layer-0
        self._init_layer1() # Initialize WTA layer-1
        self._init_connections() # Initialize connections betweens the layers
    
    def _init_layer0(self):
        """ Initialize sensory layer, including data and spike generator. """

        # Input layer is a sensory layer
        self.layer0 = LayerSensory(
            name="layer-0",
            device=self.device,
            seed=self.seed,
        )

        # Use sequential encoding where one dimension is folded out over time
        spike_generator = SGStaticSequence.instant(
            dim_time=1,

            inverted=self.inverted,

            host=self.layer0,

            device=self.device,
            seed=self.seed,
        )

        # Assign the data and spike generator to the sensory layer
        self.layer0.set_traits(

            # Data is all 4 temporal toy patterns
            data=DataToyRepeat(
                
                whitelists=[],
                blacklists=[],
                shuffle=self.shuffle,

                host=self.layer0,

                device=self.device,
                seed=self.seed,
            ),

            spike_generator=spike_generator,
        )
    
    def _init_layer1(self):
        """ Initialize WTA layer-1. """

        # Output layer-1 is a single WTA circuit
        self.layer1 = LayerWTA(
            mps_min=0,
            mps_rest=0,
            n_max_simultaneous_spikes=1,

            shape=self.K,
            
            name="layer-1",
            device=self.device,
            seed=self.seed,
            )
        
        # Layer-1 uses softmax neurons
        neuron = NeuronSoftmax(
            hz_timestep=self.HZ_TIMESTEP,
            hz_layer=self.HZ_SOFTMAX,

            host=self.layer1,

            device=self.device,
            seed=self.seed,
            )
        threshold = Threshold(
            mps_max=float('inf'),
            
            host=self.layer1,
            
            device=self.device,
            seed=self.seed,
            )
        
        # Set the traits of WTA layer-1
        self.layer1.set_traits(
            neuron=neuron,
            threshold=threshold,
            genesis= Genesis(
                host=self.layer1,

                device=self.device,
                seed=self.seed,
                ),
        )
    
    def _init_connections(self):
        """ Initialize the components that connect layer-0 with layer-1. """

        # Layer-1 uses dynamic synapses (which learn temporal relations)
        self.synapse1 = SynapseDynamic(
            cdt_rest=0.0,
            cdt_max=float('inf'),
            cst_decay=0.0,
            cst_growth=1.0,

            name="synapse-1",
            device=self.device,
            seed=self.seed,
            )
        
        # Initialize the axon connecting layer-0 with layer-1
        self.axon0to1 = Axon(
            name="axon-0to1",

            device=self.device, 
            seed=self.seed
            )
    
    def _connect_components(self):
        """ Ensure each component has access to connected components. """
        
        # Layer-0 has access to axon-0to1
        self.layer0.add_axon(axon=self.axon0to1)

        # Layer-1 has access to synapse-1
        self.layer1.set_synapse(synapse=self.synapse1)
        
        # Axon-0to1 has access to layer-0 and layer-1
        self.axon0to1.set_connections(
            layers_a=[self.layer0],
            layer_b=self.layer1,
            synapse=self.synapse1,
            )
        
        # Synapse-1 has access to axon-0to1
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
            if self.en_axon_weights:
                weights = WeightsStdp(
                    hz_timestep=self.HZ_TIMESTEP,
                    hz_layer=self.HZ_SOFTMAX,
                    decay_slow=0.99,
                    decay_fast=0.50,
                    cst_curve=1.0,
                    cst_beta=0.25,
                    
                    eta_init=1.0,
                    eta_decay=1.0,

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
                weights.weights=1.0# NOTE: may not work with saving and loading
            
            # Attach the weight trait to axon-0to1
            self.axon0to1.set_weights(weights)
        
        # Generate new weights for the axon weights trait
        if self.en_axon_weights: 
            self.axon0to1._weights.generate_weights(
                weight_generator=lambda **kwargs: WeightGenerator.uniform(
                    init_min=0.9,
                    init_max=0.99,
                    relative=True,
                    **kwargs
                    ),
                )
        # ---------------------------------------------------------------------

        # Initialize synapse weights trait if it isn't attached already
        if not hasattr(self.synapse1, "_weights"):
            weights = WeightsSynapse(
                hz_timestep=self.HZ_TIMESTEP,
                hz_layer=self.HZ_SOFTMAX,
                cst_decay_trace=1.0,
                cst_curve=1.0,

                shape=(self.synapse1.shape_b+
                        self.synapse1.shape_a+
                        self.synapse1.shape_a),
                eta_init=self.eta_init,
                eta_decay=self.eta_decay,
                eta_star=self.eta_star,
                
                host=self.synapse1,
                device=self.device,
                seed=self.seed,
                )
            self.synapse1.set_weights(weights)
        
        # Generate new weights for the axon weights trait
        self.synapse1._weights.generate_weights(
            weight_generator=lambda **kwargs: WeightGenerator.uniform(
                init_min=0.95,
                init_max=0.99,
                relative=True,
                **kwargs
                ),
            )
    
    def _init_arrangement(self):
        """ Determine the manner in which network components are arranged. """

        # First input spike trains are produced by sensory layer-0
        node_0_0 = Node(position=(0, 0), component=self.layer0)

        # Axon-0to1 transmits spikes from the sensory layer to WTA synapse-1
        node_1_0 = Node(position=(1, 0), component=self.axon0to1)

        # Synapse-1 (may) additionally weigh spikes travelling to WTA layer-1
        node_2_0 = Node(position=(2, 0), component=self.synapse1) 

        # WTA layer-1 reponds to incoming spike patterns
        node_3_0 = Node(position=(3, 0), component=self.layer1)
        
        # The arrangement of network components determines processing order
        self.arrangement = Arrangement(
            nodes=[node_0_0, node_1_0, node_2_0, node_3_0],
            )

    def _init_probes(self) -> None :
        """ Initialize probes that monitor network activity. """
        super()._init_probes()

        # Create a list to contain the probes
        self.probes = []

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

        # If diverging axon weights are enabled, visualize them
        if self.en_axon_weights:
            visualizer_weights_axon0to1 = VisualizerWeights(
                axon=self.axon0to1,
                sources=self.axon0to1.layers_a,
                en_show=False,
                en_save=True,

                network=self,
                iv_probe=self.iv_probes,
                device=self.device,
                
                figsize=(3, 2),
                )
            self.probes.append(visualizer_weights_axon0to1)

        # Visualize the dynamic synapse weights
        visualizer_weights_synapse1 = VisualizerWeightsSynapse(
            synapse=self.synapse1,
            sources=self.axon0to1.layers_a, # TODO: what are true sources?
            en_show=False,
            en_save=True,

            network=self,
            iv_probe=self.iv_probes,
            device=self.device,

            figsize=(3, 2),
            )
        self.probes.append(visualizer_weights_synapse1)
        
        # Track the count of (non-)zero weights of the dynamic synapse
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
        
        # If it is a <TEST> or <MAP> run, evaluate WTA layer-1
        if self.mode_run in [self.TEST, self.MAP]:
            logger_evaluate_layer1 = LoggerEvaluate(
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
            self.probes.append(logger_evaluate_layer1)
        
        # If this is an <EVALUATE> run, use the evaluator of which the mapping
        # has been decided during the preceding <MAP> run
        if self.mode_run == self.EVALUATE:
            for evaluator in self.evaluators:
                evaluator.loggers.append(self.logger)
            self.probes = self.probes + self.evaluators
            
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
        network: Arch1lRepeat,
        
        en_train: bool = True,
        n_unique_trn: int = 4,
        n_cycles_trn: int = 10,
        n_probings_trn: int = 8,
        
        en_test: bool = True,
        n_unique_tst: int = 4,
        n_cycles_tst: int = 10,
        n_probings_tst: int = 8,

        en_save: bool = True,
        en_save_logs: bool = True,
        en_save_visuals: bool = True,
        en_save_network: bool = True,
        en_save_tensors: bool = True,
        ) -> None:
        Network.run_network(
            network=network,

            en_train=en_train,
            data_subsets_trn=['train'],
            n_unique_trn=n_unique_trn,
            n_cycles_trn=n_cycles_trn,
            n_probings_trn=n_probings_trn,

            en_test=en_test,
            data_subsets_map=['train'],
            data_subsets_tst=['train'], # train==test
            n_unique_map=n_unique_tst, # map==test
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
    def run_default(cls, device='cpu', seed=2023):
        """ 
        Create and run a network with temporal encoding and default settings. 
        """
        network = cls._create(
            device=device,
            seed=seed,
            )
        cls.run_network(network=network)