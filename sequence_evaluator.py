from typing import List, Union
import numpy as np
from config import SequenceConfig
import os, ray, torch
from sequence_design import SequenceDesign
from surrogate import Ensemble, ConvolutionalNetworkModel
from objectives.utils import TAPELandscape, AAVLandscape


@ray.remote
class PredictorWorker:
    def __init__(self, config: SequenceConfig, device: torch.device):

        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
        self.device = device
        self.config = config

class SequenceEvaluator:
    def __init__(self, config: SequenceConfig, device: torch.device = None):
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.task = self.config.tasks_configs['task']
        self.predictor_workers = [PredictorWorker.remote(self.config, self.device) for _ in range(self.config.num_predictor_workers)] 
        
        if self.task !='AAV':
            self.oracle = TAPELandscape(self.config, self.config.oracle_path)
        elif self.task == 'AAV':
            self.oracle = AAVLandscape(oracle_path=self.config.oracle_path)

        self.proxy = Ensemble([ConvolutionalNetworkModel(len(config.tasks_configs['wt_sequences'][config.tasks_configs['task']]), config) for _ in range(config.proxy_config['ensemble_size'])]
                              ,self.config)

    def calculate_scores(self, sequences:List[Union[SequenceDesign, str]], if_wt: bool = None):
        scores = self.oracle.get_fitness(sequences)
        return scores






    
