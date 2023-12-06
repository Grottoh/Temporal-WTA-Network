import datetime
import json
import os

from network.probes.logger import Logger

from .probe import Probe

class ProbeExperiment(Probe):
    
    def __init__(
            self,
            probes: list[Probe],
            **kwargs,
            ) -> None :
        super().__init__(**kwargs)
        
        self.probes = probes


    def _probe(self, ith_stimulus: int, n_stimuli: int) -> None:
        """ ... """

        for probe in self.probes:
            if probe != self and probe.results != {}:
                idf_probe = (f"{probe.__class__.__name__}"+
                            f"-{probe.component.name}")
                self.results[idf_probe] = probe.results
        
        msg = ""
        msg +=f"[{Logger.now()}] - {ith_stimulus:5d} / {n_stimuli} - results\n"
        for idf_probe, probe_results in self.results.items():
            msg += f"{idf_probe}:\n"
            for key, result in probe_results.items():
                msg += f" > {key:>20}: {result}\n"
        msg += "\n"

        self.network.logger.produce(msg)

        if self.en_save(): # Save all results to a single .json file
            json_string = json.dumps(self.results, indent=4)
            with open(self.network.pd_run+"results.json", "w") as f:
                f.write(json_string)
        
        
    