from typing import Optional
import torch
import numpy as np

import network.probes.statistic as stat
from .logger import Logger
from ..layers.layer import Layer
from ..layers.layer_sensory import LayerSensory

class LoggerEvaluate(Logger):

    def __init__(
            self,
            layer: Layer,
            layer_sensory: LayerSensory, 
            n_past: int,
            **kwargs
            ) -> None:
        super().__init__(component=layer, **kwargs)

        # Evaluation is in reference to data of sensory layer <layer_sensory>
        self.layer_sensory = layer_sensory

        # Evaluate up to <n_past> of the most recently perceived stimuli
        self.n_past = n_past
        
        # Count each neuron's spikes for each stimuli (resets)
        _spikes_since_stimulus = stat.SpikesSinceStimulus(
            layer=self.layer,
            device=self.device,
            )
        self.key_spikes_since_stimulus = _spikes_since_stimulus.key
        _spikes_since_stimulus = self.add_statistic(_spikes_since_stimulus)
        
        # Count the number of times each neuron spiked for each label (stable)
        _count_neuron_label = stat.CountNeuronLabel(
            layer=self.layer,
            layer_sensory=self.layer_sensory,
            spikes_since_stimulus=_spikes_since_stimulus,
            device=self.device,
            )
        self.key_count_neuron_label = _count_neuron_label.key
        _count_neuron_label = self.add_statistic(_count_neuron_label)
        
        # For each stimulus, count the number of times each neuron spiked
        _count_stimulus_neuron = stat.CountStimulusNeuron(
            layer=self.layer,
            n_past=self.n_past,
            spikes_since_stimulus=_spikes_since_stimulus,
            device=self.device,
            )
        self.key_count_stimulus_neuron = _count_stimulus_neuron.key
        _count_stimulus_neuron = self.add_statistic(_count_stimulus_neuron)

        # Whether or not the neuron-to-label mapping is fixed (unchanging)
        self.fixed = False
    
    @property
    def layer(self):
        """ Alternative reference to target (layer) component. """
        return self.component

    @property
    def spikes_since_stimulus(self) -> torch.ShortTensor:
        """ Return the present value of <_spikes_since_stimulus>. """
        return self.network.statistics[self.key_spikes_since_stimulus].value
    
    @property
    def count_neuron_label(self) -> torch.ShortTensor:
        """ Return the present value of <_count_neuron_label>. """
        return self.network.statistics[self.key_count_neuron_label].value
    
    def map_neurons(self) -> None:
        """ Determine for each neuron to which label it most frequently
        responds. """
        self.neuron2label: dict[ tuple[int, ...], int ] = dict()
        for k in self.layer.KS:
            self.neuron2label[k] = torch.argmax(self.count_neuron_label[k])
    
    def fix_mapping(
            self, 
            mapping: Optional[ dict[ tuple[int, ...], int ] ] = None,
            ) -> None:
        """ Indicate that the neuron-to-label mapping should remain fixed. """
        # Set a new neuron-to-label mapping if it is given
        if not isinstance(mapping, type(None)):
            self.neuron2label = mapping
        self.fixed = True

    def _probe(self, ith_stimulus: int, n_stimuli: int) -> None:
        """ ... """

        # Do nothing if none of the loggers are active
        if not self.active:
            return
        
        # Determine for each neuron to which label it most frequently responds
        if not self.fixed:
            self.map_neurons()
        
        n_samples = self.layer_sensory.data.n_samples
        n_classes = self.layer_sensory.data.n_classes
        T = self.layer_sensory.data.T
        n_evaluate = min(ith_stimulus, self.n_past) # Nr of stimuli to evaluate

        # Get up to <n_past> stimulus-neuron counts
        _count_stimulus_neuron = self.network.statistics[
            self.key_count_stimulus_neuron]
        count_stimulus_neuron = _count_stimulus_neuron.get(n_evaluate)
        
        # Determine the indices of the to be evaluated stimuli
        indices = np.arange(
            start=_count_stimulus_neuron.index - n_evaluate, 
            stop =_count_stimulus_neuron.index, 
            dtype=np.uint32,
            ) % n_samples
        
        # For each class, count the number of right and wrong classifications
        count_correct = np.zeros(n_classes)
        count_wrong = np.zeros(n_classes)
        for idx_stimulus, csn in zip(indices, count_stimulus_neuron):
                    
            # Count for each label how often its corresponding neurons spike
            count_labels = np.zeros(n_classes, dtype=np.uint16)

            # Determine for which label the spikes are interpreted as evidence
            for k in self.layer.KS:

                # Get spike count of neuron <k> for stimulus <idx_stimulus>
                spike_count = csn[k]

                # Determine the label associated with neuron <k>
                label_k = self.neuron2label[k]

                # Spikes of neuron <k> are evidence for the corresponding label
                count_labels[label_k] += spike_count
                        
            # Label with most spikes is considered the prediction
            Y_i = np.argmax(count_labels) # The prediction
                    
            # Compare the prediction to the true label
            T_i = T[idx_stimulus%n_samples] # The true label
            if Y_i == T_i: # Prediction was correct
                count_correct[T_i] += 1
            else: # Prediction was wrong
                count_wrong[T_i] += 1
        
        # Determine the overall accuracy of the predictions
        accuracy = (
            100 *
            count_correct.sum() / 
            ( count_correct.sum() + count_wrong.sum() )
            )
        self.results['accuracy_overall'] = accuracy
        
        # TODO: compute a weighted average of the accuracy per class, ensure it
        # equals the overall accuracy as computed above.
        # Determine the accuracy per class
        accuracy_per_class = []
        for i in range(n_classes):
            accuracy_per_class.append(
                100 *
                count_correct[i].sum() / 
                ( count_correct[i].sum() + count_wrong[i].sum() )
                )
        
        # TODO: make this compatible with classes that aren't digits (I think
        # it is by now?)
        msg = ""
        msg += f"[{Logger.now()}] - {ith_stimulus:5d} / {n_stimuli} -"
        msg += f" {self.layer.name}\n"
        msg += f"Evaluating with respect to the {n_evaluate}"
        msg +=  " most recently perceived stimuli.\n"
        msg += f"Achieved an overall accuracy of {accuracy:.2f}%\n"
        msg += "Accuracy per class:\n  "
        for i, accuracy_i in enumerate(accuracy_per_class):
            label_name = self.layer_sensory.data.T_[i]
            msg += f"{i}:{label_name}={accuracy_i:.2f}%, "
            self.results[f'accuracy_{label_name}'] = accuracy_i
        msg = msg[:-2] + ".\n\n"
        
        # Log the message
        self.produce(msg=msg)
