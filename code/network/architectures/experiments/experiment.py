import json
import os
import numpy as np

class Experiment:

    # Path to the directory where experiments are saved
    PD_EXPERIMENTS = "../[03LF] experiments/"

    # Speakers of single-speaker experiments (first two by alphabetical order)
    BOYS  = ['am', 'bb']#, 'ci', 'cw', 'da']
    GIRLS = ['aa', 'ab']#, 'as', 'bp', 'ch']
    MEN   = ['ae', 'aj']#, 'al', 'aw', 'bd']
    WOMEN = ['ac', 'ag']#, 'ai', 'an', 'bh']
    BGMW = ([('b', id_boy)   for id_boy   in BOYS] + 
            [('g', id_girl)  for id_girl  in GIRLS] + 
            [('m', id_man)   for id_man   in MEN] + 
            [('w', id_woman) for id_woman in WOMEN])

    @classmethod
    def get_pd_experiment(cls, ndt_experiment: str):
        """ 
        Experiment index is the number of already saved experiments of same
        name. 
        """
        idx_experiment = len(
            [name for name in os.listdir(cls.PD_EXPERIMENTS) if 
                os.path.isdir(cls.PD_EXPERIMENTS+name) and 
                name.split('_')[0]==ndt_experiment.split('_')[0]]
            )
        return cls.PD_EXPERIMENTS + ndt_experiment.format(idx_experiment)
    
    @classmethod
    def evaluate(
            cls,
            idf_experiment: str = '1.1a_00',
            ndt_experiment: str = "experiment-{}/",
            ) -> None:
        
        # Determine the path to the experiment directory
        nd_experiment = ndt_experiment.format(idf_experiment)
        pd_experiment = cls.PD_EXPERIMENTS + nd_experiment
        
        accuracies = dict()
        
        l1_nonzero = []
        l1_zero = []
        l1_total = []

        l2_nonzero = []
        l2_zero = []
        l2_total = []

        # Iterate over network directories of the given experiment
        for nd_network in os.listdir(pd_experiment):

            # Determine path to network directory
            if os.path.isdir(pd_experiment+nd_network):
                pd_network = pd_experiment + nd_network + '/'
            else: # Skip if it is not a directory
                continue

            # Extract path to evaluation run (assumes there is exactly one)
            for nd_run in os.listdir(pd_network):
                if 'evaluate' in nd_run:
                    pd_run = pd_network + nd_run + '/'
            
            # Read the results file of the run
            pf_results = pd_run+'results.json'
            with open(pf_results, 'r') as f:
                results = json.load(f)

            # ------------- Accuracy -------------
            if 'LoggerEvaluate-layer-1' in results:
                results_accuracy = results['LoggerEvaluate-layer-1']
            elif 'LoggerEvaluate-layer-2' in results:
                results_accuracy = results['LoggerEvaluate-layer-2']
            
            for key, value in results_accuracy.items():
                if key == 'accuracy':
                    continue
                label = key.split('_')[1]
                if not label in accuracies:
                    accuracies[label] = [float(value)]
                else:
                    accuracies[label].append(float(value))
                        
            # ------------- Weight counts layer 1 -------------
            for key, value in results.items():
                
                if 'LoggerWeights-weights_synapse-2' in key:
                    l2_zero.append(value['0.0'])
                    l2_nonzero.append(value['nonzero'])
                    l2_total.append(value['total'])
                elif 'LoggerWeights-weights_synapse' in key:
                    l1_zero.append(value['0.0'])
                    l1_nonzero.append(value['nonzero'])
                    l1_total.append(value['total'])

        msg = ""    
        msg += f"{nd_experiment} - accuracies:\n"
        for key, value in accuracies.items():
            n_runs = len(value)
            mean = np.mean(value)
            stdv = np.std(value)
            minv = np.min(value)
            maxv = np.max(value)
            msg += f" > {key}:\n"
            msg += f" >>>             n_runs = {n_runs}\n"
            msg += f" >>>               mean = {mean:.2f} %\n"
            msg += f" >>> standard deviation = {stdv:.2f}\n"
            msg += f" >>>            minimum = {minv:.2f} %\n"
            msg += f" >>>            maximum = {maxv:.2f} %\n"
        
        if len(l1_nonzero) > 0:
            msg += "\n"
            msg += (f"{nd_experiment} - synapse weight counts layer-1:\n")
            msg += (f" > Total synapse weight count: {l1_total[0]:,d}\n")
            msg += (f" > Non-zero synapse weight count:\n")
            mean = np.mean(l1_nonzero)
            stdv = np.std(l1_nonzero)
            minv = np.min(l1_nonzero)
            maxv = np.max(l1_nonzero)
            prcnt = mean/l1_total[0] * 100
            msg += (f" >>>             n_runs = {n_runs}\n")
            msg += (f" >>>               mean = {mean:,.0f} ({prcnt:.2f}%)\n")
            msg += (f" >>> standard deviation = {stdv:,.0f}\n")
            msg += (f" >>>            minimum = {minv:,.0f}\n")
            msg += (f" >>>            maximum = {maxv:,.0f}\n")

        if len(l2_nonzero) > 0:
            msg += "\n"
            msg += (f"{nd_experiment} - synapse weight counts layer-2:\n")
            msg += (f" > Total synapse weight count: {l2_total[0]:,d}\n")
            msg += (f" > Non-zero synapse weight count:\n")
            mean = np.mean(l2_nonzero)
            stdv = np.std(l2_nonzero)
            minv = np.min(l2_nonzero)
            maxv = np.max(l2_nonzero)
            prcnt = mean/l2_total[0] * 100
            msg += (f" >>>             n_runs = {n_runs}\n")
            msg += (f" >>>               mean = {mean:,.0f} ({prcnt:.2f}%)\n")
            msg += (f" >>> standard deviation = {stdv:,.0f}\n")
            msg += (f" >>>            minimum = {minv:,.0f}\n")
            msg += (f" >>>            maximum = {maxv:,.0f}\n")

        print(msg)
        with open(pd_experiment+f"results-{idf_experiment}.txt", 'w') as f:
            f.write(msg)
