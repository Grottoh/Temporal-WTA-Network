import math

from network.architectures.experiments.experiment import Experiment
from network.architectures.tidigits.arch2l_tidigits import Arch2lTidigits
from network.layers.traits.neuron import NeuronStochastic

class Experiment3(Experiment):
    """
    Temporal WTA network with single-speaker TIDIGITS data. The number of
    output neurons (l2_K) is smaller than the number of unique stimuli, and
    equal to the number of digit classes (11).
    """

    @classmethod
    def run_1(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(5)]),
        device='cpu',
        ndt_network: str = "network-3.1{}_{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-3.1{}_{}/",
        ) -> None:
        """ Varying number of train cycles. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            n_cycles_trn=1
        elif idf == 'b':
            n_cycles_trn=5
        elif idf == 'c':
            n_cycles_trn=10
        elif idf == 'd':
            n_cycles_trn=30
        else:
            raise NotImplementedError
        
        for _, id_speaker in cls.BGMW:
            for seed in sorted(set(seeds)):
                cls._run(
                    id_speaker=id_speaker,
                    pd_experiment=pd_experiment,
                    ndt_network=ndt_network.format(
                        idf, id_speaker, seed, '{:02d}'),

                    device=device,
                    seed=seed,

                    n_cycles_trn=n_cycles_trn
                    )

    @classmethod
    def run_2(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(5)]),
        device='cpu',
        ndt_network: str = "network-3.2{}_{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-3.2{}_{}/",
        ) -> None:
        """ Varying number circuits in layer-1. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            l1_n_circuits=1
        elif idf == 'b':
            l1_n_circuits=10
        else:
            raise NotImplementedError
        
        for _, id_speaker in cls.BGMW:
            for seed in sorted(set(seeds)):
                cls._run(
                    id_speaker=id_speaker,
                    pd_experiment=pd_experiment,
                    ndt_network=ndt_network.format(
                        idf, id_speaker, seed, '{:02d}'),

                    device=device,
                    seed=seed,

                    l1_n_circuits=l1_n_circuits,
                    )

    @classmethod
    def run_3(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(5)]),
        device='cpu',
        ndt_network: str = "network-3.3{}_{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-3.3{}_{}/",
        ) -> None:
        """ Varying number of neurons in layer-1. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            l1_K=(10,)
        elif idf == 'b':
            l1_K=(50,)
        elif idf == 'c':
            l1_K=(150,)
        else:
            raise NotImplementedError
        
        for _, id_speaker in cls.BGMW:
            for seed in sorted(set(seeds)):
                cls._run(
                    id_speaker=id_speaker,
                    pd_experiment=pd_experiment,
                    ndt_network=ndt_network.format(
                        idf, id_speaker, seed, '{:02d}'),

                    device=device,
                    seed=seed,

                    l1_K=l1_K,
                    )
    
    @classmethod
    def _run(
        cls,

        id_speaker: str,
        pd_experiment: str,
        ndt_network: str,

        device: str,
        seed: int,

        l1_n_circuits: int = 5,
        l1_K: tuple[int] = (100,),

        n_cycles_trn: int = 30,
        ) -> None:
        
        network = Arch2lTidigits._create(
            l1_type_neuron=NeuronStochastic,
            
            label_by = ['utterance'],
            whitelists=[{
                'utterance': ['o', 'z', '1', '2', '3', '4', 
                            '5', '6', '7', '8', '9'],
                'speaker-id': [id_speaker],
                }],

            l1_n_circuits=l1_n_circuits,
            l1_K=l1_K,
            l1_cst_idle=0,

            l2_K=(11,),
            l2_idle_until=math.floor(0.6*n_cycles_trn)*22,

            pd_networks=pd_experiment,
            ndt_network=ndt_network,

            en_visualize_weights=False,

            device=device,
            seed=seed,
            )
        Arch2lTidigits.run_network(
            network=network,

            n_unique_trn=22,
            n_cycles_trn=n_cycles_trn,

            n_unique_tst=22,
            n_unique_map=22,
            n_cycles_tst=10,

            data_subsets_trn=['train'],
            n_probings_trn=1,

            data_subsets_map=['train'],
            data_subsets_tst=['train'],
            n_probings_tst=1,
            )
