from __future__ import annotations
from typing import Optional, Union
import datetime
import os

from .probe import Probe

class Logger(Probe):

    """ Log (to a terminal and/or to a text file) data/diagnostics regarding
    (parts of) the network. """
    
    @staticmethod
    def now() -> str:
        """ Return the present <year-month-day hour:minute:second>. """
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # NOTE: in case I want to be oblivious to time
        #return datetime.datetime.now().timestamp()
    
    def __init__(
            self,
            loggers: list[Logger],
            en_local_logging: bool,
            nf_log: Optional[str] = None,
            **kwargs,
            ) -> None :
        super().__init__(**kwargs)

        # A (possibly empty) list of Loggers which save what this log logs
        self.loggers = loggers

        # Also log apart from the other <loggers>
        self.en_local_logging = en_local_logging

        # Logger must have something to log to
        if self.loggers == [] and not self.en_local_logging:
            raise ValueError("Logger must have something to log to.")

        # The name of the log file
        self.nf_log = self.component.name+".log" if nf_log == None else nf_log
        
        # Keep track of a (initially empty) message to be logged or printed
        self.msg = ""
    
    def init_directories(
            self, 
            pd_log: Optional[str] = None, 
            nf_log: Optional[str] = None
            ):
        """ If local logging is enabled, determine and create the directory of
        the log file and optionally set a new filename. If saving is enabled,
        create/overwrite the log file with an initial message. """

        # If local logging is not enabled, no directories/files need be created
        if not self.en_local_logging:
            return
        
        # Determine directory for the logger component
        if pd_log == None: # Create directory according to component name
            super().init_directories(en_save=self.network.en_save_logs)
        else: # Use the given path to determine the directory
            self.pd_component = pd_log

        # Set the path to the log file (only update filename one is given)
        self.nf_log = self.nf_log if nf_log == None else nf_log # File name
        self.pf_log = self.pd_component + self.nf_log # File path

        # If saving is enabled ...
        if self.network.en_save_logs and any(self._en_save): 
            
            # If necessary, create the directory for the logger component
            if not os.path.isdir(self.pd_component):
                os.mkdir(self.pd_component)
            
            # Create/overwrite log file with initial message
            with open(self.pf_log, 'w') as f:
                now = datetime.datetime.now()
                msg = (f"Initialized log <{self.pf_log}> on" +
                           f" {now.strftime('%Y-%m-%d')} at" +
                           f" {now.strftime('%H:%M:%S')}.")
                self.produce( Logger.frame(msg=msg, char='#', s_padding=3) )
    
    @property
    def active(self):
        """ Return True if any of the loggers have showing/saving enabled. """

        # Active if any of the external loggers has showing enabled
        if self.network.en_show_logs:
            for logger in self.loggers:
                if any(logger._en_show):
                    return True
                
        # Active if any of the external loggers has saving enabled
        if self.network.en_save_logs:
            for logger in self.loggers:
                if any(logger._en_save):
                    return True
        
        # Active if the local logger has either showing or saving enabled
        return (( self.network.en_show_logs and any(self._en_show) ) or 
                ( self.network.en_save_logs and any(self._en_save) ))
    
    def en_show(self, i: Optional[int] = None):
        """ Determine whether to show something according to given index. """
        return self.network.en_show_logs and super().en_show(i=i)
    
    def en_save(self, i: Optional[int] = None):
        """ Determine whether to save something according to given index. """
        return self.network.en_save_logs and super().en_save(i=i)

    def frame(msg: str, char: str = '#', s_padding: int = 5):
        """ Frame the given message in <s_padding> <char>s. """
        msg = (char*s_padding + " " + msg + " " + char*s_padding + "\n")
        msg = char*(len(msg)-1)+"\n" + msg + char*(len(msg)-1)+"\n\n"
        return msg
        
    def append(self, msg: str) -> None:
        """ Append the given message to the logger's existing message. """
        self.msg += msg
    
    def overwrite(self, msg: str) -> None:
        """ Overwrite the logger's existing message with the given message. """
        self.msg = msg
    
    def produce(self, msg: str = "", en_show: bool = True) -> None:
        """ Print and/or log the logger's message plus the given message. """

        # Append to the existing message
        self.msg += msg
        
        # If enabled, log the complete message
        if self.en_save():

            # If enabled, save to its own (local) log file
            if self.en_local_logging: 
                with open(self.pf_log, 'a') as f:
                    f.write(self.msg)

            # If enabled, have designated loggers save message of this logger
            for logger in self.loggers:
                logger.produce(msg, en_show=False)
        
        # If enabled, print the complete message
        if self.en_show() and en_show:
            print(self.msg, end="")
        
        # Reset the logger's message to the empty string
        self.msg = ""