from network.architectures.experiments.experiment import Experiment
from network.architectures.tidigits.arch1l_tidigits import Arch1lTidigits

class Experiment5(Experiment):
    """
    Single layer temporal WTA network with all single-digit TIDIGITS data.
    """

    @classmethod
    def run_1(
        cls, 
        idf: str, 
        seeds: set = set([(i+1)*100 for i in range(3)]),
        device='cpu',
        ndt_network: str = "network-5.1{}_s{:04d}_{}/",
        ndt_experiment: str = "experiment-5.1{}_{}/",
        ) -> None:
        """ Varying number neurons and bins. """

        # Determine the path to the experiment directory
        pd_experiment = cls.get_pd_experiment(
            ndt_experiment=ndt_experiment.format(idf, '{:02d}'))

        if idf == 'a':
            K = (100,)
            n_bins = 8
        elif idf == 'b':
            K = (200,)
            n_bins = 8
        elif idf == 'c':
            K = (100,)
            n_bins = 16
        else:
            raise NotImplementedError
        
        for seed in sorted(set(seeds)):
            cls._run(
                pd_experiment=pd_experiment,
                ndt_network=ndt_network.format(idf, seed, '{:02d}'),

                device=device,
                seed=seed,

                K=K,
                n_bins=n_bins,
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

        K: tuple[int] = (100,),

        en_train: bool = True,
        n_cycles_trn: int = 5,
        ) -> None:
        
        network = Arch1lTidigits._create(            
            data_form=data_form,
            label_by = label_by,
            whitelists=[{
                'utterance': ['o', 'z', '1', '2', '3', '4', 
                              '5', '6', '7', '8', '9'],
                }],
            shuffle=shuffle,
            n_bins=n_bins,

            K=K,

            pd_networks=pd_experiment,
            ndt_network=ndt_network,

            en_visualize_weights=False,

            device=device,
            seed=seed,
            )
        Arch1lTidigits.run_network(
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