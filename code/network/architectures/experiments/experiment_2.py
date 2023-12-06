import math

from network.architectures.experiments.experiment import Experiment
from network.architectures.tidigits.arch2l_tidigits import Arch2lTidigits
from network.layers.traits.neuron import NeuronStochastic

class Experiment2(Experiment):
    """
    Temporal WTA network with single-speaker TIDIGITS data. The number of
    output neurons (l2_K) is greater than or equal to the number of unique
    stimuli.
    """

    @classmethod
    def run_1(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(5)]),
        device='cpu',
        ndt_network: str = "network-2.1{}_{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-2.1{}_{}/",
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
        ndt_network: str = "network-2.2{}_{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-2.2{}_{}/",
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
        ndt_network: str = "network-2.3{}_{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-2.3{}_{}/",
        ) -> None:
        """ Varying number of neurons. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            l1_K=(10,)
            l2_K=(22,)
            n_cycles_tst=10
        elif idf == 'b':
            l1_K=(50,)
            l2_K=(22,)
            n_cycles_tst=10
        elif idf == 'c':
            l1_K=(100,)
            l2_K=(50,)
            n_cycles_tst=10
        elif idf == 'd':
            l1_K=(100,)
            l2_K=(50,)
            n_cycles_tst=30
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
                    l2_K=l2_K,

                    n_cycles_tst=n_cycles_tst,
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
        l2_K: tuple[int] = (22,),

        n_cycles_trn: int = 10,
        n_cycles_tst: int = 10,
        ) -> None:
        
        network = Arch2lTidigits._create(
            l1_type_neuron=NeuronStochastic,
            
            label_by = ['index'],
            whitelists=[{
                'utterance': ['o', 'z', '1', '2', '3', '4', 
                            '5', '6', '7', '8', '9'],
                'speaker-id': [id_speaker],
                }],

            l1_n_circuits=l1_n_circuits,
            l1_K=l1_K,
            l1_cst_idle=0,

            l2_K=l2_K,
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
            n_cycles_tst=n_cycles_tst,

            data_subsets_trn=['train'],
            n_probings_trn=1,

            data_subsets_map=['train'],
            data_subsets_tst=['train'],
            n_probings_tst=1,
            )