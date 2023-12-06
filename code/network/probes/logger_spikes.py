import torch

import network.probes.statistic as stat
from ..layers.layer import Layer
from .logger import Logger

class LoggerSpikes(Logger):
    
    """ Tracks the spiking behaviour of the given layer over the past specified
    period of time. """

    def __init__(
            self,
            layer: Layer,
            aggregation_period: int,
            **kwargs
            ) -> None:
        super().__init__(component=layer, **kwargs)
        
        # Count each neuron's spikes for each stimuli (resets)
        _spikes_since_stimulus = stat.SpikesSinceStimulus(
            layer=self.layer,
            device=self.device,
            )
        self.key_spikes_since_stimulus = _spikes_since_stimulus.key
        _spikes_since_stimulus= self.add_statistic(_spikes_since_stimulus)
        
        # Remember for the past <n_past> stimuli what the spike response was
        _spikes_history = stat.SpikesHistory(
            layer=self.layer,
            n_past=aggregation_period,
            spikes_since_stimulus=_spikes_since_stimulus,
            device=self.device,
        )
        self.key_spikes_history = _spikes_history.key
        _spikes_history = self.add_statistic(_spikes_history)

        # Probe and process spike data of the last <aggregation period>
        # stimuli. True aggregation period may be limited by the number of
        # stimuli processed thus far.
        self._aggregation_period = aggregation_period
    
    @property
    def layer(self):
        """ Alternative reference to target (layer) component. """
        return self.component
    
    @property
    def aggregation_period(self) -> int:
        """ Aggregation period is not greater than nr of processed stimuli. """
        _spikes_history = self.network.statistics[self.key_spikes_history]
        return min(_spikes_history.index, self._aggregation_period)

    @property
    def spikes_history(self) -> tuple[torch.ShortTensor, int]:
        """ Return spikes history of the last <aggregation_period> stimuli. """
        _spikes_history = self.network.statistics[self.key_spikes_history]
        return _spikes_history.get(m_past=self.aggregation_period)

    def _probe(self, ith_stimulus: int, n_stimuli: int) -> None:
        """ Print and/or log statistics regarding recent spiking behaviour. """

        # Do nothing if none of the loggers are active
        if not self.active:
            return

        # Retrieve the aggregation period and spikes history
        aggregation_period = self.aggregation_period
        spikes_history = self.spikes_history

        # Determine spike statistics over <aggregation_period> past stimuli
        dims_layer = tuple( [d+1 for d in range(self.layer.n_dims)] )
        spikes_sum = spikes_history.sum(dims_layer)
        spikes_mean = spikes_sum.mean() # Mean nr of spikes per stimulus
        spikes_stdv = spikes_sum.std() # Stdv spikes over stimuli
        spikes_min = spikes_sum.min() # Minimum nr of spikes for a stimulus
        spikes_max = spikes_sum.max() # Maximum nr of spikes for a stimulus

        # Write the message to be logged
        msg = (
            f"[{Logger.now()}] - {ith_stimulus:5d} / {n_stimuli} -" +
            f" {self.layer.name}\n" +
            f"Spike statistics for the past {aggregation_period} stimuli:\n" +
             " >             Average number of spikes per stimulus:" + 
            f" {spikes_mean:5.2f}\n" 
             " > Standard deviation wrt number spikes per stimulus:" + 
            f" {spikes_stdv:5.2f}\n"
             " > Minimum and maximum amount of spikes respectively:" +
            f" {spikes_min:.0f}, {spikes_max:.0f}\n\n"
            )
        
        # Log the message
        self.produce(msg=msg)

    
