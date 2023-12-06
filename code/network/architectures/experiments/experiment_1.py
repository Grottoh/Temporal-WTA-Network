from network.architectures.experiments.experiment import Experiment
from network.architectures.toy.arch1l_toy import Arch1lToy
from network.layers.traits.neuron import NeuronSoftmax
from network.spike_generators.sg_static import SGStatic
from network.spike_generators.sg_static_sequence import SGStaticSequence
from network.synapses.synapse import Synapse, SynapseDynamic

class Experiment1(Experiment):

    @classmethod
    def run_1(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(10)]),
        device='cpu',
        ndt_network: str = "network-1.1{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-1.1{}_{}/",
        ) -> None:
        """
        Static WTA network with static and temporal encodings of toy data.
        """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            type_sg=SGStatic # Static encoding
            n_cycles_trn=1
        elif idf == 'b':
            type_sg=SGStaticSequence # Temporal encoding
            n_cycles_trn=1
        elif idf == 'c':
            type_sg=SGStaticSequence # Temporal encoding
            n_cycles_trn=10
        else:
            raise NotImplementedError
        
        for seed in sorted(set(seeds)):
            network = Arch1lToy._create(
                type_sg=type_sg,
                type_neuron=NeuronSoftmax,
                type_synapse=Synapse,

                pd_networks=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),
                
                device=device,
                seed=seed,
                )
            network.run_network(
                network=network, 
                
                n_cycles_trn=n_cycles_trn,
                n_probings_trn=4,

                n_cycles_tst=10,
                n_probings_tst=8,
                )

    @classmethod
    def run_2(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(10)]),
        device='cpu',
        ndt_network: str = "network-1.2{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-1.2{}_{}/",
        ) -> None:
        """
        Temporal WTA network with static and temporal encodings of toy data.
        """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            type_sg=SGStatic # Static encoding
            n_cycles_trn=1
            en_axon_weights = True
        elif idf == 'b':
            type_sg=SGStaticSequence # Temporal encoding
            n_cycles_trn=1
            en_axon_weights = True
        elif idf == 'c':
            type_sg=SGStaticSequence # Temporal encoding
            n_cycles_trn=1
            en_axon_weights = False
        else:
            raise NotImplementedError
        
        for seed in sorted(set(seeds)):
            network = Arch1lToy._create(
                type_sg=type_sg,
                type_neuron=NeuronSoftmax,
                type_synapse=SynapseDynamic,

                en_axon_weights=en_axon_weights,

                pd_networks=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),
                
                device=device,
                seed=seed,
                )
            network.run_network(
                network=network, 
                
                n_cycles_trn=n_cycles_trn,
                n_probings_trn=4,

                n_cycles_tst=10,
                n_probings_tst=8,
                )