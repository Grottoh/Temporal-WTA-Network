import math

from network.architectures.experiments.experiment import Experiment
from network.architectures.tidigits.arch2l_tidigits import Arch2lTidigits
from network.layers.traits.neuron import NeuronStochastic

class Experiment4(Experiment):
    """
    Temporal WTA network with all single-digit TIDIGITS data.
    """

    @classmethod
    def run_1(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(3)]),
        device='cpu',
        ndt_network: str = "network-4.1{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-4.1{}_{}/",
        ) -> None:
        """ Varying number of train cycles. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            en_train = False
            n_cycles_trn = 0
        elif idf == 'b':
            en_train = True
            n_cycles_trn=3
        elif idf == 'c':
            en_train = True
            n_cycles_trn=5
        elif idf == 'd':
            en_train = True
            n_cycles_trn=7
        else:
            raise NotImplementedError
        
        for seed in sorted(set(seeds)):
            cls._run(
                pd_experiment=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                device=device,
                seed=seed,

                en_train=en_train,
                n_cycles_trn=n_cycles_trn
                )

    @classmethod
    def run_2(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(3)]),
        device='cpu',
        ndt_network: str = "network-4.2{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-4.2{}_{}/",
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
        
        for seed in sorted(set(seeds)):
            cls._run(
                pd_experiment=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                device=device,
                seed=seed,

                l1_n_circuits=l1_n_circuits,
                )

    @classmethod
    def run_3(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(3)]),
        device='cpu',
        ndt_network: str = "network-4.3{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-4.3{}_{}/",
        ) -> None:
        """ Varying number of neurons. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            l1_K=(50,)
            l2_K=(50,)
        elif idf == 'b':
            l1_K=(150,)
            l2_K=(150,)
        else:
            raise NotImplementedError
        
        for seed in sorted(set(seeds)):
            cls._run(
                pd_experiment=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                device=device,
                seed=seed,

                l1_K=l1_K,
                l2_K=l2_K,
                )

    @classmethod
    def run_4(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(3)]),
        device='cpu',
        ndt_network: str = "network-4.4{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-4.4{}_{}/",
        ) -> None:
        """ Varying learning rate decay. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            eta_decay = 0.80
        elif idf == 'b':
            eta_decay = 1.00
        else:
            raise NotImplementedError
        
        for seed in sorted(set(seeds)):
            cls._run(
                pd_experiment=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                device=device,
                seed=seed,

                eta_decay=eta_decay,
                )

    @classmethod
    def run_5(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(3)]),
        device='cpu',
        ndt_network: str = "network-4.5{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-4.5{}_{}/",
        ) -> None:
        """ Varying maximum number of simultaneous spikes in layer-1. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            l1_n_max_simultaneous_spikes=1
        elif idf == 'b':
            l1_n_max_simultaneous_spikes=5
        else:
            raise NotImplementedError
        
        for seed in sorted(set(seeds)):
            cls._run(
                pd_experiment=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                device=device,
                seed=seed,

                l1_n_max_simultaneous_spikes=l1_n_max_simultaneous_spikes,
                )

    @classmethod
    def run_6(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(3)]),
        device='cpu',
        ndt_network: str = "network-4.6{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-4.6{}_{}/",
        ) -> None:
        """ Changes to data encoding. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            data_form = 'mfcc'
            n_bins = 8
            l1_mps_max = 500.0
        elif idf == 'b':
            data_form = 'delta'
            n_bins = 8
            l1_mps_max = 500.0
        elif idf == 'c':
            data_form = 'mfcc&delta'
            n_bins = 6
            l1_mps_max = 1500.0
        elif idf == 'd':
            data_form = 'mfcc&delta'
            n_bins = 10
            l1_mps_max = 1500.0
        else:
            raise NotImplementedError
        
        for seed in sorted(set(seeds)):
            cls._run(
                pd_experiment=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                device=device,
                seed=seed,

                data_form=data_form,
                n_bins=n_bins,
                l1_mps_max=l1_mps_max,
                )

    @classmethod
    def run_7(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(3)]),
        device='cpu',
        ndt_network: str = "network-4.7{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-4.7{}_{}/",
        ) -> None:
        """ Miscellaneous. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            label_by=['utterance']
            shuffle = True
            axon_en_learn = False
            n_cycles_trn=5
            l1_n_circuits=5
            l1_K=l2_K=(100,)
        elif idf == 'b':
            label_by=['utterance']
            shuffle = False
            axon_en_learn = True
            n_cycles_trn=5
            l1_n_circuits=5
            l1_K=l2_K=(100,)
        elif idf == 'c':
            label_by=['utterance']
            shuffle = False
            axon_en_learn = False
            n_cycles_trn=7
            l1_n_circuits=10
            l1_K=l2_K=(150,)
        elif idf == 'd':
            label_by=['development', 'sex']
            shuffle = False
            axon_en_learn = False
            n_cycles_trn=5
            l1_n_circuits=5
            l1_K=l2_K=(100,)
        else:
            raise NotImplementedError
        
        for seed in sorted(set(seeds)):
            cls._run(
                pd_experiment=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                device=device,
                seed=seed,
                
                label_by=label_by,
                shuffle=shuffle,
                axon_en_learn=axon_en_learn,

                l1_n_circuits=l1_n_circuits,

                l1_K=l1_K,
                l2_K=l2_K,
                
                n_cycles_trn=n_cycles_trn
                )
    
    @classmethod
    def _run(
        cls,

        pd_experiment: str,
        ndt_network: str,

        device: str,
        seed: int,

        data_form: str = 'mfcc&delta',
        label_by: list[str] = ['utterance'],
        shuffle: bool = False,
        n_bins = 8,

        l1_n_circuits: int = 5,
        l1_K: tuple[int] = (100,),
        l1_n_max_simultaneous_spikes: int = 3,
        l1_mps_max: float = 1500.0,

        l2_K: tuple[int] = (100,),

        axon_en_learn: bool = False,

        eta_decay: float = 0.60,

        en_train: bool = True,
        n_cycles_trn: int = 5,
        ) -> None:
        
        network = Arch2lTidigits._create(
            l1_type_neuron=NeuronStochastic,
            
            data_form=data_form,
            label_by = label_by,
            whitelists=[{
                'utterance': ['o', 'z', '1', '2', '3', '4', 
                              '5', '6', '7', '8', '9'],
                }],
            shuffle=shuffle,
            n_bins=n_bins,

            l1_n_circuits=l1_n_circuits,
            l1_K=l1_K,
            l1_n_max_simultaneous_spikes=l1_n_max_simultaneous_spikes,
            l1_cst_idle=0,
            l1_mps_max=l1_mps_max,

            l2_K=l2_K,
            l2_idle_until=math.floor(0.6*n_cycles_trn)*3586,

            axon_en_learn=axon_en_learn,

            eta_decay=eta_decay,

            pd_networks=pd_experiment,
            ndt_network=ndt_network,

            en_visualize_weights=False,

            device=device,
            seed=seed,
            )
        Arch2lTidigits.run_network(
            network=network,

            en_train=en_train,
            n_unique_trn=3586,
            n_cycles_trn=n_cycles_trn,

            n_unique_tst=3586,
            n_unique_map=3586,
            n_cycles_tst=1,

            data_subsets_trn=['train'],
            n_probings_trn=10,

            data_subsets_map=['train'],
            data_subsets_tst=['test'],
            n_probings_tst=10,
            )