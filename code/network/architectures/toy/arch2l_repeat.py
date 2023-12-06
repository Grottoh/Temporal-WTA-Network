from __future__ import annotations
from typing import Optional, Union
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

class Arch2lRepeat(Network):

    # Determines the interval at which softmax neurons spike
    HZ_SOFTMAX = Network.HZ_TIMESTEP/20 # (Iff type_neuron==NeuronSoftmax)

    def __init__(
            self,

            l1_type_neuron: type[Neuron] = NeuronSoftmax,
            
            inverted: bool = False,
            shuffle: bool = False,

            l1_n_circuits: int = 5,
            l1_K: tuple[int, ...] = (4,),
            l1_hz: Optional[ Union[float, list[float]] ] = 1000/10,
            l1_n_max_simultaneous_spikes: int = 1,
            l1_cst_idle: int = 0,
            l1_learn_until: Union[float, int] = float('inf'),
            l1_cst_p_spike: Optional[float] = 30.0,
            l1_mps_max: Optional[float] = 10.0,

            l2_K: tuple[int, ...] = (4,),
            l2_idle_until: int = 0,

            axon_w_init_min: float = 0.9990,
            axon_w_init_max: float = 0.9999,
            axon_en_weights: bool = True,
            axon_en_learn: bool = False,

            s1_w_init_min: float = 0.60,
            s1_w_init_max: float = 0.80,

            s2_w_init_min: float = 0.60,
            s2_w_init_max: float = 0.80,
            s2_hz: float = 1000/20,

            eta_init: float = 1.0,
            eta_decay: float = 0.60,
            eta_star: int = 25,

            en_visualize_weights: bool = True,

            pd_networks: str = Network.PD_NETWORKS + "arch2l-repeat/",
            ndt_network: str = "arch2l-repeat_{:04d}/", # _{idx_network}

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

        # Layer-1 parameters (<l1_n_circuits> circuit WTA layer and neurons)
        # ---------------------------------------------------------------------
        # The type of neuron WTA layer-1 consists of
        self.l1_type_neuron = l1_type_neuron
        
        # The number of circuits in layer-1
        self.l1_n_circuits = l1_n_circuits

        # The number of neurons in each WTA circuit of layer-1
        self.l1_K = l1_K

        # Approximate frequency at which stochastic neurons in layer-1 operate
        if type(l1_hz) == float: # Repeat value for each circuit
            self.l1_hz = [l1_hz for _ in range(self.l1_n_circuits)]
        elif len(l1_hz) != self.l1_n_circuits: # Length must fit
            raise ValueError(
                f"Length of <l1_hz={l1_hz}> must equal the number of" +
                f" circuits <{self.l1_n_circuits}> in layer-1."
                 )
        else: # Assign the list of frequencies
            self.l1_hz = l1_hz

        # Determine how many neurons in circuits b may spike simultaneously
        self.l1_n_max_simultaneous_spikes = l1_n_max_simultaneous_spikes

        # Each circuit in layer-1 is idle for <l1_cst_idle*idx_layer> stimuli
        self.l1_cst_idle = l1_cst_idle

        # For how many stimuli layer-1 (and incoming axons/synapses) learns
        self.l1_learn_until = l1_learn_until

        # Determines slope of a stochastic neuron's spike-probability-curve
        # (Iff type_neuron==NeuronStochastic)
        self.l1_cst_p_spike = l1_cst_p_spike

        # Spiking probability is 1 if membrane potential is at <mps_max>
        self.l1_mps_max = l1_mps_max # (Iff l1_type_neuron==NeuronStochastic)
        # ---------------------------------------------------------------------

        # Layer-2 parameters (single circuit WTA layer and softmax neurons)
        # ---------------------------------------------------------------------
        # The number of neurons in the single WTA circuit of layer-2
        self.l2_K = l2_K

        # Layer-2 is idle for <l2_cst_idle> cycles
        self.l2_idle_until = l2_idle_until

        # Frequency at which the single WTA circuit of layer-2 produces a spike
        self.l2_hz = 1000/20

        # Connection parameters (axons and synapses)
        # ---------------------------------------------------------------------
        # Determine whether axons have (diverging) weights
        self.axon_en_weights = axon_en_weights

        # Minimum and maximum initial axon weights (iff <axon_en_weights>)
        self.axon_w_init_min = axon_w_init_min
        self.axon_w_init_max = axon_w_init_max

        # Whether or not axon weights change (according to STDP dynamics)
        self.axon_en_learn = axon_en_learn

        # Minimum and maximum initial weight of synapse-1
        self.s1_w_init_min = s1_w_init_min
        self.s1_w_init_max = s1_w_init_max

        # Minimum and maximum initial weight of synapse-2
        self.s2_w_init_min = s2_w_init_min
        self.s2_w_init_max = s2_w_init_max

        # Timescale at which synapse-2 operates
        self.s2_hz = s2_hz

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
        self._init_layer1() # Initialize the WTA circuits of layer-1
        self._init_layer2() # Initialize the single WTA circuit of layer-2
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
        """ Initialize layer-1 consisting of <l1_n_circuits> WTA circuits. """

        # Initialize the list that holds the circuits of layer-1
        self.layer1: list[LayerWTA] = []

        # Initialize each circuit in layer-1
        for idx_circuit, hz in enumerate(self.l1_hz):

            # Prevent identical randomness
            seed_i = self.seed + idx_circuit 

            # Initialize a WTA circuit
            circuit_i = LayerWTA(
                mps_min=0,
                mps_rest=0,
                n_max_simultaneous_spikes=self.l1_n_max_simultaneous_spikes,

                idle_until=idx_circuit*self.l1_cst_idle,
                learn_until=self.l1_learn_until,
                shape=self.l1_K,
                
                name=f"layer-1{chr(97+idx_circuit)}",
                device=self.device,
                seed=seed_i,
                )
            setattr(self, circuit_i.name, circuit_i)

            # Determine what type of neuron layer-1 circuits consists of
            # -----------------------------------------------------------------
            # Layer-1 uses stochastic neurons
            if self.l1_type_neuron == NeuronStochastic:
                neuron_b = NeuronStochastic(
                    cst_p_spike=self.l1_cst_p_spike,

                    host=circuit_i,

                    device=self.device,
                    seed=seed_i,
                    )
                threshold_b = Threshold(
                    mps_max=self.l1_mps_max,

                    host=circuit_i,
                    
                    device=self.device,
                    seed=self.seed,
                    )
            
            # Layer-1 uses softmax neurons
            elif self.l1_type_neuron == NeuronSoftmax:
                neuron_b = NeuronSoftmax(
                    hz_timestep=self.HZ_TIMESTEP,
                    hz_layer=hz,

                    host=circuit_i,

                    device=self.device,
                    seed=self.seed,
                    )
                threshold_b = Threshold(
                    mps_max=float('inf'),
                    
                    host=circuit_i,
                    
                    device=self.device,
                    seed=self.seed,
                    )
                
            # There is no implementation for the specified neuron
            else:
                raise TypeError(
                    f"Neuron of type {self.l1_type_neuron} is not known.")
            # -----------------------------------------------------------------

            # Set circuit traits
            circuit_i.set_traits(
                neuron=neuron_b,
                threshold=threshold_b,
                genesis= Genesis(
                    host=circuit_i,

                    device=self.device,
                    seed=seed_i, 
                    ),
                )
            
            # Add the circuit to layer-1
            self.layer1.append(circuit_i)
    
    def _init_layer2(self):
        """ Initialize layer-2 consisting of a single WTA circuit. """

        # Layer-2 makes a final 'classification' based on input from layer-1
        self.layer2 = LayerWTA(
            mps_min=0,
            mps_rest=0,
            n_max_simultaneous_spikes=1,

            shape=self.l2_K,
            
            idle_until=self.l2_idle_until,
            name="layer-2",
            device=self.device,
            seed=self.seed,
            )
        
        # Set the traits of WTA layer-2
        self.layer2.set_traits(

            # Layer-2 uses softmax neurons
            neuron=NeuronSoftmax(
                hz_timestep=self.HZ_TIMESTEP,
                hz_layer=self.l2_hz,
                en_variable_duration=self.en_variable_duration,

                host=self.layer2,

                device=self.device,
                seed=self.seed,
                ),
            threshold=Threshold(
                mps_max=float('inf'),

                host=self.layer2,
                
                device=self.device,
                seed=self.seed,
                ),
            genesis= Genesis(
                host=self.layer2,

                device=self.device,
                seed=self.seed,
                ),
            )
    
    def _init_connections(self):
        """ Initialize the components that connect layer-0 with layer-1. """

        # Set synapses-1 of layer-1
        # ---------------------------------------------------------------------
        # Initialize the list that holds the synapses of layer-1
        self.synapses1: list[SynapseDynamic] = []

        # Set dynamic synapses for each circuit in layer-1
        for idx_circuit, circuit_i in enumerate(self.layer1):
            synapse1i = SynapseDynamic(
                cdt_rest=0.0,
                cdt_max=float('inf'),
                cst_decay=0.0,
                cst_growth=1.0,

                learn_until=self.l1_learn_until,
                name=f"synapse-1{chr(97+idx_circuit)}",
                device=self.device,
                seed=circuit_i.seed,
                )
            setattr(self, synapse1i.name, synapse1i)
            self.synapses1.append(synapse1i)
        # ---------------------------------------------------------------------


        # Set axons-0to1 between layer-0 and layer-1
        # ---------------------------------------------------------------------
        # Initialize the list that holds axons0to1
        self.axons0to1: list[Axon] = []

        # Set axon from layer-0 to each circuit in layer-1
        for idx_circuit, circuit_i in enumerate(self.layer1):
            axon0to1i = Axon(
                learn_after=0 if self.axon_en_learn else float('inf'),
                learn_until=self.l1_cst_idle,

                name=f"axon-0to1{chr(97+idx_circuit)}",
                device=self.device, 
                seed=circuit_i.seed,
                )
            setattr(self, axon0to1i.name, axon0to1i)
            self.axons0to1.append(axon0to1i)
        # ---------------------------------------------------------------------
        
        # Set dynamic synapses for layer-2
        self.synapse2 = SynapseDynamic(
            cdt_rest=0.00,
            cdt_max=float('inf'),
            cst_decay=0.0,
            cst_growth=1.0,

            idle_until=self.l2_idle_until,
            name="synapse-2",
            device=self.device,
            seed=self.seed,
            )
        
        # Axon from all circuits in layer-1 to the single circuit of layer-2
        self.axon1to2 = Axon(
            learn_after=(self.l2_idle_until if 
                         self.axon_en_learn else 
                         float('inf')),
            idle_until=self.l2_idle_until,
            name="axon-1to2",
            device=self.device, 
            seed=self.seed
            )
    
    def _connect_components(self):
        """ Ensure each component has access to connected components. """

        # For each circuit in layer-1 ...
        for layer1i, synapse1i, axon0to1i in zip(
            self.layer1, self.synapses1, self.axons0to1):

            # Layer-0 has access to axon-0to1i leading from layer-0 to layer-1i
            self.layer0.add_axon(axon=axon0to1i)

            # Layer-1i has access to synapse-1i
            layer1i.set_synapse(synapse=synapse1i)

            # Axon-0to1 has access to layer-0 and layer-1i and synapse-1i
            axon0to1i.set_connections(
                layers_a=[self.layer0],
                layer_b=layer1i,
                synapse=synapse1i,
                )
            
            # Synapse-1i has access to axon-0to1i and layer-1i
            synapse1i.set_connections(
                axons=[axon0to1i],
                layer=layer1i,
                )
            
            # Layer-1i has access to axon-1to2
            # TODO: check whether this (<add_axon>) is even necessary
            layer1i.add_axon(self.axon1to2)
        
        # Layer2 has access to synapse2
        self.layer2.set_synapse(synapse=self.synapse2)
        
        # Axon1to2 has access to layer1, layer2, and synapse-2
        self.axon1to2.set_connections(
            layers_a=self.layer1,
            layer_b=self.layer2,
            synapse=self.synapse2,
            )
        
        # Synapse-2 has access to axon1to2 and layer-2
        self.synapse2.set_connections(
            axons=[self.axon1to2],
            layer=self.layer2,
            )
    
    def _set_traits(self):
        """ Set the traits of network components """

        # Set weight trait of each axon in axons-0to1
        # ---------------------------------------------------------------------
        for hz, axon0to1i in zip(self.l1_hz, self.axons0to1):
            axon0to1i: Axon

            # Initialize axon weights trait if it isn't attached already
            if not hasattr(axon0to1i, "_weights"):

                # Allow diverging and changing axon weights
                if self.axon_en_weights:
                    weights = WeightsStdp(
                        hz_timestep=self.HZ_TIMESTEP,
                        hz_layer=hz,
                        decay_slow=0.99,
                        decay_fast=0.50,
                        cst_curve=1.0,
                        cst_beta=0.0,
                        
                        eta_init=1.0,
                        eta_decay=self.eta_decay,
                        eta_star=self.eta_star,

                        shape=axon0to1i.shape,
                        w_min=0.0,
                        
                        host=axon0to1i,
                        device=self.device,
                        seed=axon0to1i.seed,
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
                        
                        host=axon0to1i,
                        device=self.device,
                        seed=axon0to1i.seed,
                    )
                    # NOTE: may not work with saving and loading
                    weights.weights=1.0
                
                # Attach the weight trait to axon-0to1i
                axon0to1i.set_weights(weights)
        
        # Generate new weights for the weights trait of axons-0to1
        if self.axon_en_weights: 
            for axon0to1i in self.axons0to1:
                axon0to1i._weights.generate_weights(
                    weight_generator=lambda **kwargs: WeightGenerator.uniform(
                        init_min=self.axon_w_init_min,
                        init_max=self.axon_w_init_max,
                        relative=True,
                        **kwargs
                        ),
                    )
        # ---------------------------------------------------------------------

        # Set weight trait of each synapse in synapses-1
        # ---------------------------------------------------------------------
        for hz, synapse1i in zip(self.l1_hz, self.synapses1):
            synapse1i: SynapseDynamic
            
            # Initialize synapse weights trait if it isn't attached already
            if not hasattr(synapse1i, "_weights"):
                weights = WeightsSynapse(
                    hz_timestep=self.HZ_TIMESTEP,
                    hz_layer=hz,
                    cst_curve=1.0,

                    shape=(synapse1i.shape_b+
                           synapse1i.shape_a+
                           synapse1i.shape_a),
                    eta_init=self.eta_init,
                    eta_decay=self.eta_decay,
                    eta_star=self.eta_star,
                    
                    name=f"weights_{synapse1i.name}",
                    host=synapse1i,
                    device=self.device,
                    seed=synapse1i.seed,
                    )
                synapse1i.set_weights(weights)
            
            # Generate new weights for the weights trait of synapse-1i
            synapse1i._weights.generate_weights(
                weight_generator=lambda **kwargs: WeightGenerator.uniform(
                    init_min=self.s1_w_init_min,
                    init_max=self.s1_w_init_max,
                    relative=True,
                    **kwargs
                    ),
                )
        # ---------------------------------------------------------------------
        
        # Set weight trait of axon-1to2
        # ---------------------------------------------------------------------
        # Initialize axon weights trait if it isn't attached already
        if not hasattr(self.axon1to2, "_weights"):

            # Allow diverging and changing axon weights
            if self.axon_en_weights:
                weights = WeightsStdp(
                    hz_timestep=self.HZ_TIMESTEP,
                    hz_layer=self.s2_hz,
                    decay_slow=0.99,
                    decay_fast=0.50,
                    cst_curve=1.0,
                    cst_beta=0.0,
                    
                    eta_init=1.0,
                    eta_decay=self.eta_decay,
                    eta_star=self.eta_star,

                    shape=self.axon1to2.shape,
                    w_min=0.0,
                    
                    host=self.axon1to2,
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
                    
                    host=self.axon1to2,
                    device=self.device,
                    seed=self.seed,
                )
                # NOTE: may not work with saving and loading
                weights.weights=1.0
            
            # Attach the weight trait to axon-0to1i
            self.axon1to2.set_weights(weights)
        
        # Generate new weights for the weights trait of axons-1to2
        if self.axon_en_weights:
            self.axon1to2._weights.generate_weights(
                weight_generator=lambda **kwargs: WeightGenerator.uniform(
                    init_min=self.axon_w_init_min,
                    init_max=self.axon_w_init_max,
                    relative=True,
                    **kwargs
                    ),
                )
        # ---------------------------------------------------------------------

        # Set weight trait of synapse in synapse-2
        # ---------------------------------------------------------------------
        # Initialize synapse weights trait if it isn't attached already
        if not hasattr(self.synapse2, "_weights"):
            weights = WeightsSynapse(
                hz_timestep=self.HZ_TIMESTEP,
                hz_layer=self.s2_hz,
                cst_curve=1.0,

                shape=(self.synapse2.shape_b+
                       self.synapse2.shape_a+
                       self.synapse2.shape_a),
                eta_init=self.eta_init,
                eta_decay=self.eta_decay,
                eta_star=self.eta_star,
                
                name=f"weights_{self.synapse2.name}",
                host=self.synapse2,
                device=self.device,
                seed=self.seed,
                )
            self.synapse2.set_weights(weights)
        
        # Generate new weights for the weights trait of synapse-1i
        self.synapse2._weights.generate_weights(
            weight_generator=lambda **kwargs: WeightGenerator.uniform(
                init_min=self.s2_w_init_min,
                init_max=self.s2_w_init_max,
                relative=True,
                **kwargs
                ),
            )
        # ---------------------------------------------------------------------
    
    def _init_arrangement(self):
        """ Determine the manner in which network components are arranged. """

        # First input spike trains are produced by sensory layer-0
        node_0_0 = Node(position=(0, 0), component=self.layer0)

        # Axons-0to1 transmits spikes from the sensory layer to WTA synapses-1
        nodes_1_i = []
        for i, axon0to1i in enumerate(self.axons0to1):
            nodes_1_i.append( Node(position=(1, i), component=axon0to1i) )

        # Synapses-1 additionally weigh spikes travelling to WTA layer-1
        nodes_2_i = []
        for i, synapse1i in enumerate(self.synapses1):
            nodes_2_i.append( Node(position=(2, i), component=synapse1i) )

        # Each WTA circuit in layer-1 reponds to incoming spike patterns
        nodes_3_i = []
        for i, layer1i in enumerate(self.layer1):
            nodes_3_i.append( Node(position=(3, i), component=layer1i) )
        
        # Axon-1to2 transmits spikes from the circuits of layer-1 to synapse-2
        node_4_0 = Node(position=(4, 0), component=self.axon1to2)

        # Synapse-2 additionally weighs spikes travelling to WTA layer-2
        node_5_0 = Node(position=(5, 0), component=self.synapse2) 

        # WTA layer-2 reponds to incoming spike patterns from WTA layer-1
        node_6_0 = Node(position=(6, 0), component=self.layer2)
        
        # The arrangement of network components determines processing order
        self.arrangement = Arrangement(
            nodes=(
                [node_0_0] +
                nodes_1_i + nodes_2_i + nodes_3_i + 
                [node_4_0, node_5_0, node_6_0]
                ),
            )

    def _init_probes(self) -> None :
        """ Initialize probes that monitor network activity. """
        super()._init_probes()

        # Create a list to contain the probes
        self.probes = []

        # Log spike counts
        # ---------------------------------------------------------------------
        # Log spike-count statistics of each WTA circuit in layer 1
        for layer1i in self.layer1:
            logger_spikes_layer1i = LoggerSpikes(
                layer=layer1i,
                aggregation_period=min(100, self.iv_probes),

                loggers=[self.logger],
                en_local_logging=True,
                
                network=self,
                en_show=True,
                en_save=True,
                iv_probe=self.iv_probes,
                device=self.device,
                )
            self.probes.append(logger_spikes_layer1i)

        # Log spike-count statistics of WTA layer 2
        logger_spikes_layer2 = LoggerSpikes(
            layer=self.layer2,
            aggregation_period=min(100, self.iv_probes),

            loggers=[self.logger],
            en_local_logging=True,
            
            network=self,
            en_show=True,
            en_save=True,
            iv_probe=self.iv_probes,
            device=self.device,
            )
        self.probes.append(logger_spikes_layer2)
        # ---------------------------------------------------------------------
        
        # Track the count of (non-)zero weights of the synapses
        # ---------------------------------------------------------------------
        # Log weights of each synapse in synapses-1
        for synapse1i in self.synapses1:
            synapse1i: SynapseDynamic
            logger_weights_synapse1i = LoggerWeights(
                weights=synapse1i._weights,

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
            self.probes.append(logger_weights_synapse1i)
        
        # Log weights of synapse-2
        logger_weights_synapse2 = LoggerWeights(
            weights=self.synapse2._weights,

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
        self.probes.append(logger_weights_synapse2)
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

        # Visualize spikes of each WTA circuit in layer-1
        for layer1i in self.layer1:
            visualizer_spikes_layer1i = VisualizerSpikes(
                layer=layer1i,
                n_timesteps=self.layer0.spike_generator.n_timesteps,

                en_show=False,
                en_save=True,

                network=self,
                iv_probe=self.iv_probes,
                device=self.device,
                )
            self.probes.append(visualizer_spikes_layer1i)

        # Visualize spikes of WTA layer-2
        visualizer_spikes_layer2 = VisualizerSpikes(
            layer=self.layer2,
            n_timesteps=self.layer0.spike_generator.n_timesteps,

            en_show=False,
            en_save=True,

            network=self,
            iv_probe=self.iv_probes,
            device=self.device,
            )
        self.probes.append(visualizer_spikes_layer2)
        # ---------------------------------------------------------------------

        # If diverging axon weights are enabled, visualize them
        # ---------------------------------------------------------------------
        if self.axon_en_learn and self.en_visualize_weights: # If enabled ...
            for axon0to1i in self.axons0to1:
                    visualizer_weights_axon0to1i = VisualizerWeights(
                        axon=axon0to1i,
                        sources=axon0to1i.layers_a,
                        en_show=False,
                        en_save=True,

                        network=self,
                        iv_probe=self.iv_probes,
                        device=self.device,
                        
                        figsize=(3, 2),
                        )
                    self.probes.append(visualizer_weights_axon0to1i)
        # ---------------------------------------------------------------------

        # Visualize synapse weights
        # ---------------------------------------------------------------------
        if self.en_visualize_weights: # If enabled ...
            for axon0to1i, synapse1i in zip(self.axons0to1, self.synapses1):
                axon0to1i: Axon; synapse1i: SynapseDynamic
                visualizer_weights_synapse1i = VisualizerWeightsSynapse(
                    synapse=synapse1i,
                    sources=axon0to1i.layers_a, # TODO: what are true sources?
                    en_show=False,
                    en_save=True,

                    network=self,
                    iv_probe=self.iv_probes,
                    device=self.device,
                    
                    figsize=(3, 2),
                    )
                self.probes.append(visualizer_weights_synapse1i)
        # ---------------------------------------------------------------------

        # Evaluate performance
        # ---------------------------------------------------------------------
        # If it is a <TEST> or <MAP> run, evaluate WTA layer-2
        if self.mode_run in [self.TEST, self.MAP]:
            self.logger_evaluate_layer2 = LoggerEvaluate(
                layer=self.layer2,
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
            self.probes.append(self.logger_evaluate_layer2)
        
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
        network: Arch2lRepeat,
        
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
    def run_softmax(cls, device='cpu', seed=2023):
        """ 
        Create and run a network with temporal encoding and softmax neurons. 
        """
        network = cls._create(
            l1_type_neuron=NeuronSoftmax,

            device=device,
            seed=seed,
            )
        cls.run_network(network=network)
    
    @classmethod
    def run_stochastic(cls, device='cpu', seed=2023):
        """ 
        Create and run a network with temporal encoding and stochastic neurons. 
        """
        network = cls._create(
            l1_type_neuron=NeuronStochastic,
            l1_K=(16,),

            device=device,
            seed=seed,
            )
        cls.run_network(network=network, n_cycles_trn=40)