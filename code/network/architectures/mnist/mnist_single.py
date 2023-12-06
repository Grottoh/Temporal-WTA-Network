from __future__ import annotations
from typing import Optional

from network.weights.weights_norm import WeightsNorm

#from ...utils.util import Constants as C
from ...network import Network
from ...data_loaders.data_mnist import DataMNIST
from ...spike_generators.sg_static import SGStatic
from ...layers.layer_sensory import LayerSensory
from ...layers.layer_wta import LayerWTA
#from ...layers.traits.neuron import NeuronCallbacks
from ...layers.traits.neuron import Neuron, NeuronSoftmax, NeuronStochastic
from ...weights.weights import Weights, WeightGenerator
from ...layers.traits.threshold import Threshold, ThresholdAdaptive
from ...axons.axon import Axon
from ...weights.weights_stdp import WeightsStdp
from ...synapses.synapse import Synapse
from ...probes.logger import Logger
from ...probes.logger_evaluate import LoggerEvaluate
from ...probes.visualizer_weights import VisualizerWeights
from ...probes.logger_spikes import LoggerSpikes
from ..arrangement import Node, Arrangement
from ...data_loaders.data import Data
from ...supervisors.supervisor import Supervisor
from ...layers.traits.genesis import Genesis, GenesisX
from ...layers.traits.threshold import ThresholdDependent
from ...utils.util import Constants as C

class MnistSingle(Network):

    # Path to directory where MNIST networks are saved
    PD_NETWORKS = Network.PD_NETWORKS + "mnist/"

    # Base template of the network's directory name
    # {mode_sg}-{inverted-}k{K}_{idx_network}
    NDT_NETWORK = "{}-{}k{}_{}/"

    def __init__(
            self,
            mode_sg: str = SGStatic.LINEAR,
            inverted: bool = False,
            K: tuple[int, ...] = (100,),
            **kwargs
            ) -> None :
        super().__init__(**kwargs)
        
        # Customizable network parameters
        self.mode_sg = mode_sg
        self.inverted = inverted
        self.K = K
    
    @classmethod
    def create(cls, **kwargs) -> MnistSingle:
        network = cls(**kwargs)

        # Initialize network components
        network._init_components()

        # Initialize network components
        network._connect_components()

        # Initialize certain trait attributes
        # TODO: naming doesn't really match in purpose with <_init_components>
        network._init_traits() 
        
        # # Initialize the generic network structure
        network._init_arrangement()

        return network
    
    @property
    def ndt_network(self):
        """ Template of network's directory name, <{:4d}> is placeholder for
        network index. """    
        ndt_network = self.NDT_NETWORK.format(
            self.mode_sg.lower(),
            "inv-" if self.inverted else "",
            self.K,
            '{:04d}',
            )
        return ndt_network
        
    def _init_components(self):

        # One timestep every <1/hz_timestep> seconds
        self.hz_timestep = 1000
        self.hz_sensory = 200
        self.hz_wta = 50

        # Input layer is a static amplitude-sensitive SpikeGenerator
        self.layer_a = LayerSensory(
            name="layer_a",
            device=self.device,
            seed=self.seed,
        )
        self.layer_a.set_traits(
            data=DataMNIST(
                binary=False,
                
                whitelists=[],
                blacklists=[],
                shuffle=False,

                host=self.layer_a,

                device=self.device,
                seed=self.seed,
            ),

            spike_generator=SGStatic(
                mode=self.mode_sg,
                threshold=0.0,
                hz_spike=self.hz_sensory,
                hz_timestep=self.hz_timestep,
                d_seconds=200/1000,
                inverted=self.inverted,

                host=self.layer_a,

                device=self.device,
                seed=self.seed,
            )
        )

        # Output layer is a WTA layer
        self.layer_b = LayerWTA(
            mps_min=0,
            mps_rest=0,
            n_max_simultaneous_spikes=1,

            shape=self.K,
            
            name="layer_b",
            device=self.device,
            seed=self.seed,
            )
        self.layer_b.set_traits(
            neuron=NeuronStochastic(
                host=self.layer_b,

                device=self.device,
                seed=self.seed,
                ),
            
            # threshold=Threshold(
            #     mps_max=self.hz_sensory/self.hz_wta,

            #     host=self.layer_b,
                
            #     device=self.device,
            #     seed=self.seed,
            #     ),
            # threshold=ThresholdDependent(
            #     hz_a=self.hz_sensory,
            #     hz_b=self.hz_wta,

            #     host=self.layer_b,
                
            #     device=self.device,
            #     seed=self.seed,
            #     ),
            
            genesis= Genesis(
                host=self.layer_b,

                device=self.device,
                seed=self.seed,
                ),
            # genesis= GenesisX(
            #     cst_redundance=3,
            #     max_redundance=1500,

            #     host=self.layer_b,

            #     device=self.device,
            #     seed=self.seed,
            #     ),
            )
        
        self.synapse_b = Synapse(
            name="synapse_b",
            device=self.device,
            seed=self.seed,
            )
        
        self.axon_ab = Axon(
            name="axon_ab",
            device=self.device, 
            seed=self.seed
            )
    
    def _connect_components(self):
        
        self.layer_a.add_axon(axon=self.axon_ab)

        self.layer_b.set_synapse(synapse=self.synapse_b)
        
        self.axon_ab.set_connections(
            layers_a=[self.layer_a],
            layer_b=self.layer_b,
            synapse=self.synapse_b,
            )

        self.synapse_b.set_connections(
            axons=[self.axon_ab],
            layer=self.layer_b,
            )
    
    def _init_traits(self):

        # Initialize axon weights trait if it isn't attached already
        if not hasattr(self.axon_ab, "_weights"):
            weights = WeightsStdp(
                hz_timestep=self.hz_timestep,
                hz_layer=self.hz_wta,
                decay_slow=0.99,
                decay_fast=0.50,
                cst_curve=1.0,
                cst_beta=0.25,
                
                eta_init=1.0,
                eta_decay=1.0,

                shape=self.axon_ab.shape,
                w_min=-0.2,
                
                host=self.axon_ab,
                device=self.device,
                seed=self.seed,
                )
            self.axon_ab.set_weights(weights)
        
        # Generate new weights for the axon weights trait
        self.axon_ab._weights.generate_weights(
            weight_generator=lambda **kwargs: WeightGenerator.uniform(
                init_min=0.9,
                init_max=0.99,
                relative=True,
                **kwargs
                ),
            )
        
        if not hasattr(self.layer_b, "threshold"):
            threshold = ThresholdAdaptive(
                hz_timestep=self.hz_timestep,
                hz_layer=self.hz_wta,
                cst_decay=1 - 1e-6,
                cst_growth=0.20,

                host=self.layer_b,
                
                device=self.device,
                seed=self.seed,
                )
            threshold.generate_weights(
                weight_generator=lambda **kwargs: WeightGenerator.gaussian(
                    init_mean=0.5,
                    init_stdv=0.1,
                    w_min=self.hz_sensory/self.hz_wta*0.5 *10,
                    w_max=self.hz_sensory/self.hz_wta*1.5 *10,
                    relative=True,
                    **kwargs
                    ),
                )
            self.layer_b.set_traits(threshold=threshold)
    
    def _init_arrangement(self):
        """ Determine the manner in which network components are arranged. """

        # Sensory layer
        node_0_0 = Node(position=(0, 0), component=self.layer_a)

        # Axon transmitting spikes from the sensory layer to the WTA synapses
        node_1_0 = Node(position=(1, 0), component=self.axon_ab)

        # Supervisor that ensures certain neurons responds to certain classes
        #node_1_1 = Node(position=(1, 1), component=self.supervisor)

        # Synapses of WTA layer
        node_2_0 = Node(position=(2, 0), component=self.synapse_b) 

        # WTA layer
        node_3_0 = Node(position=(3, 0), component=self.layer_b)
        
        # The arrangement of network components determines processing order
        self.arrangement = Arrangement(
            nodes=[node_0_0, node_1_0, node_2_0, node_3_0],
            )

    def _init_probes(self) -> None :
        """ Initialize probes that monitor network activity. """
        super()._init_probes()

        self.probes = []

        self.logger_spikes_layer_b = LoggerSpikes(
            layer=self.layer_b,
            aggregation_period=min(1000, self.iv_probes),

            loggers=[self.logger],
            en_local_logging=True,
            
            network=self,
            en_show=True,
            en_save=True,
            iv_probe=max(100, self.iv_probes),
            device=self.device,
            )
        self.probes.append(self.logger_spikes_layer_b)

        self.visualizer_weights_axon_ab = VisualizerWeights(
            axon=self.axon_ab,
            sources=self.axon_ab.layers_a,
            en_show=False,
            en_save=True,

            network=self,
            iv_probe=self.iv_probes,
            #iv_probe=1,
            device=self.device,

            #figdims=(2, 10),
            #figsize=(5, 2),
            )
        self.probes.append(self.visualizer_weights_axon_ab)

        # NOTE: includes <self.TRAIN>
        if self.mode_run in [self.TEST, self.MAP, self.TRAIN]: 
            self.logger_evaluate_layer_b = LoggerEvaluate(
                layer=self.layer_b,
                layer_sensory=self.layer_a,
                n_past=min(10_000, self.n_stimuli),

                loggers=[self.logger],
                en_local_logging=True,
                
                network=self,
                en_show=True,
                en_save=True,
                iv_probe=500,
                device=self.device,
            )
            self.probes.append(self.logger_evaluate_layer_b)
        
        # <EVALUATE> is preceded by <MAP>, we want to use the evaluators used
        # during the <MAP> run
        if self.mode_run == self.EVALUATE:
            for evaluator in self.evaluators:
                evaluator.loggers.append(self.logger)
            self.probes = self.probes + self.evaluators

    @classmethod
    def run_default(cls):
        """ Convenience function for training/testing MnistSingle nets. """

        # Determine whether to create or load the network
        create = bool(1)
        if create: # Determine a few network parameters
            K = (100,)
            device = 'cpu'
            seed = 2023
            print( "\nCreating new network with " + 
                  f"K={K}, device={device}, seed={seed}.\n")
        else: # Determine a few loading parameters
            pd_load = cls.PD_NETWORKS + "linear-k(500,)_0000/"
            #pd_load = "[03LF] networks (servers)/" + "linear-k(500,)_0002/"
            en_load_tensors = bool(1)
            print(f"\nLoading network <{pd_load}>.\n")

        # Determine whether to train and/or to test the network
        train = bool(1)
        test  = bool(1)
        evaluate = bool(1)
        
        # Determine for how many stimuli to train, and when to load/save
        n_probings_trn = 60 # How many times to probe during testing
        n_stimuli_trn = 60_000
        iv_probes_trn = n_stimuli_trn // n_probings_trn
        iv_save_trn = n_stimuli_trn // n_probings_trn
        
        # Determine for how many stimuli to test, and when to load/save
        n_probings_tst = 10 # How many times to probe during testing
        n_stimuli_tst = 10_000
        iv_probes_tst = n_stimuli_tst // n_probings_tst
        iv_save_tst = float('inf') #n_stimuli_tst // n_probings_tst
        
        # Determine for how many stimuli to test, and when to load/save
        n_probings_evl = 10 # How many times to probe during testing
        n_stimuli_evl = 10_000
        iv_probes_evl = n_stimuli_evl // n_probings_evl
        iv_save_evl = float('inf') #n_stimuli_tst // n_probings_tst

        if create: # Create a new network
            network = cls.create(
                mode_sg=SGStatic.LINEAR,
                inverted=False,
                K=K,
                device=device,
                seed=seed,
                )
        else: # Load a saved network
            network = Network.load_by_ids(
                pd_network=pd_load,
                idx_run=-1,
                idx_checkpoint=-1,
                en_load_tensors=en_load_tensors,
                )
        
        # Train the network
        if train:
            
            # Do a run with train data
            network.run(
                mode_run=cls.TRAIN,
                data_subsets=['train'],
                n_stimuli=n_stimuli_trn,
                en_learn=True,

                en_show=True,
                en_show_logs=True,
                en_show_visuals=True,

                en_save=True,
                en_save_logs=True,
                en_save_visuals=True,
                en_save_network=True,
                en_save_tensors=True,

                iv_probes=iv_probes_trn,
                iv_save_network=iv_save_trn,
                )
            
        # Test the network
        if test:
            
            # Ensure the network is active for the entirety of the test run
            network.activate()

            # Do a run with test data
            network.run(
                mode_run=cls.TEST,
                data_subsets=['test'],
                n_stimuli=n_stimuli_tst,
                en_learn=False,

                en_show=True,
                en_show_logs=True,
                en_show_visuals=True,

                en_save=True,
                en_save_logs=True,
                en_save_visuals=True,
                en_save_network=True,
                en_save_tensors=True,

                iv_probes=iv_probes_tst,
                iv_save_network=iv_save_tst,
                )
            
        # Evaluate the network
        if evaluate:
            
            # Ensure the network is active for the entirety of the test run
            network.activate()

            # Do a run with test data
            network.evaluate(
                data_subsets_map=['train'], 
                data_subsets_eval=['test'],
                n_stimuli=n_stimuli_evl,
                en_learn=False,

                en_show=True,
                en_show_logs=True,
                en_show_visuals=True,

                en_save=True,
                en_save_logs=True,
                en_save_visuals=True,
                en_save_network=True,
                en_save_tensors=True,

                iv_probes=iv_probes_evl,
                iv_save_network=iv_save_evl,
                )
