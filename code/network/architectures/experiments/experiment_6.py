from network.architectures.experiments.experiment import Experiment
from network.architectures.toy.arch1l_repeat import Arch1lRepeat
from network.architectures.toy.arch1l_toy import Arch1lToy
from network.architectures.toy.arch2l_repeat import Arch2lRepeat
from network.layers.traits.neuron import NeuronSoftmax, NeuronStochastic
from network.spike_generators.sg_static import SGStatic
from network.spike_generators.sg_static_sequence import SGStaticSequence
from network.synapses.synapse import Synapse, SynapseDynamic

class Experiment6(Experiment):

    @classmethod
    def run_1(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(10)]),
        device='cpu',
        ndt_network: str = "network-6.1{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-6.1{}_{}/",
        ) -> None:
        """
        Temporal WTA network with static and temporal encodings of toy data.
        """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))
        
        for seed in sorted(set(seeds)):

            if idf == 'a':
                network = Arch1lRepeat._create(
                    pd_networks=pd_experiment,
                    ndt_network=ndt_network.format(idf, seed, '{:02d}'),
                    
                    device=device,
                    seed=seed,
                    )
            elif idf == 'b':
                network = Arch2lRepeat._create(
                    l1_type_neuron=NeuronSoftmax,

                    l1_n_circuits=1,
                    l2_idle_until=6*4,

                    pd_networks=pd_experiment,
                    ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                    device=device,
                    seed=seed,
                    )
            elif idf == 'c':
                network = Arch2lRepeat._create(
                    l1_type_neuron=NeuronStochastic,

                    l1_n_circuits=1,
                    l2_idle_until=6*4,
                    
                    pd_networks=pd_experiment,
                    ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                    device=device,
                    seed=seed,
                    )
            elif idf == 'd':
                network = Arch2lRepeat._create(
                    l1_type_neuron=NeuronStochastic,

                    l1_n_circuits=1,
                    l1_K=(16,),
                    l2_idle_until=6*4,
                    
                    pd_networks=pd_experiment,
                    ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                    device=device,
                    seed=seed,
                    )
            elif idf == 'e':
                network = Arch2lRepeat._create(
                    l1_type_neuron=NeuronStochastic,

                    l1_n_circuits=5,
                    l1_K=(16,),
                    l2_idle_until=6*4,
                    
                    pd_networks=pd_experiment,
                    ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                    device=device,
                    seed=seed,
                    )
            else:
                raise NotImplementedError
            
            network.run_network(
                network=network, 
                
                n_cycles_trn=10,
                n_probings_trn=4,

                n_cycles_tst=10,
                n_probings_tst=8,
                )