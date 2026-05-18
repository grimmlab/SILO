import copy
import numpy as np
import torch
from torch import nn
from config import SequenceConfig
from core.abstract import BaseTrajectory
from core.utils import softmax
import random
import string
from typing import Optional, List, Tuple
from collections import defaultdict

class SequenceDesign(BaseTrajectory):

    """
    Environment for the protein sequence design.

    Actions are chosen hierarchically in two levels.
    
        - Level 0: Pick a position.
        - Level 1: If position picked, pick a residue to replace. 

    Residue types are specified in the config under `residue_vocabulary`.

    We store all actions in a history, which is a list of indices indicating the action that was taken on a certain level.

    """

    def __init__(self, config: SequenceConfig, length:int, input_sequence: str = None, num_masked_sites: int = None):
        """
        Parameters:
            config [SequenceConfig]: Config
            initial_residue [int] or [List]: We always start with already one residue in the sequence to be able to diversify
                the starting point for the network.
        """

        self.config = config
        self.vocabulary_residue_idcs = list(range(0, len(self.config.residue_vocabulary))) 
        self.vocabulary_residue_names = list(self.config.residue_vocabulary.keys())
        self.residue_feasibility_mask = [not self.config.residue_vocabulary[x]["allowed"] for x in self.vocabulary_residue_names]  # if not allowed, then feasbility mask must be set to True
        self.residue_to_idx = {aa: i for i, aa in enumerate(self.config.residue_vocabulary.keys())}
        self.input_sequence = input_sequence
        self.num_masked_sites = num_masked_sites
        self.residues = self.make_residue_list_from_string(self.input_sequence)
        self.changed_positions = []

        self.edit_counts = defaultdict(int)  # counts how many times a position has been edited
        self.static_action_map = {} # make a static map of action indices

        for i, res in enumerate(self.residues):
            self.static_action_map[i] = ("existing", res)

        self.seq_string = [] # to store the actual sequence string 
        self.identifier = None

        # Current action level. Can be 0 or 1 
        self.current_action_level = 0  

        # picked_position 
        self.picked_position: Optional[int] = None # keep track of which position has been picked for lvl 0

        # The action mask indicates before each action what is feasible at the current level.
        # 0: the action is masked. 1 = action is allowed.
        self.current_action_mask: Optional[np.array] = None

        # History is a list of `actions_taken` above, indicating how you get from the initial residue to the current sequence.
        self.history: List[int] = []
        self.design_done: bool = False
        self.level_list: List[int] = []

        # Keep track of all objectives 
        self.objective: Optional[float] = None
        self.objective_dict = {
            'tape': None,
            'surrogate_mean': None, 
            'surrogate_uncertainty': None, 
            'ucb_surrogate': None, 
            'diversity': None,
            'novelty': None, 
            'alanine_scan_mean': None, 
            'ucb_alanine_scan': None,
            'alanine_scan_uncertainty': None
            }  

        # Set this to True if anything goes wrong and the sequence will always evaluate to objective -inf
        self.infeasibility_flag: bool = False

        self.update_action_mask()
        self.update_design_sequence(new_residue=self.residues)


    def update_action_mask(self):

        """
        Creates the action mask for the current action level.
        1: this action is allowed, 0: this action is not allowed.

        level 0 layout: [0 ...L = pick position]
        level 1 layout: [0...19: residues]

        """

        self.existing = self.available_existing_positions(self.static_action_map)

        if self.current_action_level == 0:
            mask = np.zeros(len(self.residues), dtype=int) # all positions 
            changed = set(self.changed_positions)
            no_select_new_position = len(changed) >= self.num_masked_sites

            if not no_select_new_position: 
                for i, (status, res) in self.static_action_map.items():
                        if (i in self.existing) and (i not in changed): #still below masking limit → allow new changes
                            mask[i] = 1
            else: 
                # once changed positions have reached num_masked_sites, only allow reeditting these positions
                for i in changed: 
                    if self.edit_counts[i] < self.config.max_limit_pos_change: #reached masking limit → restrict to already chosen position
                        mask[i] = 1

            self.current_action_mask = mask
            
        else:
            # for level 1 (no restrictions on AA selection)
            mask = np.ones(len(self.config.residue_vocabulary), dtype=int) 

            self.current_action_mask = mask

    def update_design_sequence(self, new_residue: List):
        
        """
            Update the sequence by adding new residues.

            new_residue : List[Optional[int]]
                A list of residue indices (ints) or None values.
        """
        for residue_idx in new_residue:
            if residue_idx is None:
                self.seq_string.append(None)
            else:
                residue_alphabet = self.vocabulary_residue_names[residue_idx]
                self.seq_string.append(residue_alphabet)

    def masked_log_probs_for_current_action_level(self, logits: np.ndarray) -> np.ndarray:
        
        """
        Apply current_action_mask to logits and return normalized log-probs.
        
        """
        mask = self.current_action_mask.astype(bool)
        logits = logits.copy()
        logits[~mask] = -np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.log(softmax(logits))

        return log_probs
    
    def make_residue_list_from_string(self, sequence: str) -> List:

        """
        Makes a list of residues idcs with virtual residue index for a gvien sequence string.

        Parameters:
            sequence: a string of protein with ther permissible 20 amino acids 

        Returns:
            residues: a list of residue idcs corresponding to the amino acids string 
        """
        residue_to_idx = {res: idx for res, idx in 
        zip(self.vocabulary_residue_names, self.vocabulary_residue_idcs)}
        
        residues = []

        for s in sequence:
            if s in residue_to_idx.keys():
                residues.append(residue_to_idx[s])
            else:
                raise KeyError(f"Residue '{s}' not found in residue vocabulary: {list(residue_to_idx.keys())}")
        
        return residues
    
    def take_action(self, action: int):

        """
        Takes an action on the current action level and updates everything accordingly (see inline comments).
        Note that the updates are performed in-place!

        level 0 can have "pick position":
        lvl_0 layout = [0 = pos1, 1 = pos2, 2 = pos3 .... L-1 = posL]
        level 1 is to select an amino acid for adding a residue
        lvl_1 layout: [0...19: residues]

        """
        assert not self.design_done, "Taking action on already terminated design. No no!"

        assert self.current_action_mask[action] == 1, (f"Trying to take action {action} on level {self.current_action_level}, "
                "but it is set to infeasible")
        
        if action >= len(self.current_action_mask):
            raise ValueError(f"Invalid action {action}, mask size {len(self.current_action_mask)}")

        if self.current_action_level == 0:
            if not len(self.changed_positions) >= self.num_masked_sites:
                status, res = self.static_action_map[action] 
                if res is not None and status == 'existing':
                    if action not in self.changed_positions:
                        self.changed_positions.append(action) 
                        self.picked_position = action
                        self.level_list.append(self.current_action_level)
                        next_level = 1

        elif self.current_action_level == 1:
            aa = self.vocabulary_residue_names[action]
            if self.picked_position in self.existing:
                self.seq_string[self.picked_position] = aa
                self.residues[self.picked_position] = action
                self.edit_counts[self.picked_position] += 1 
                self.level_list.append(self.current_action_level)
                next_level = 0   

        self.history.append(int(action))
        self.current_action_level = next_level
        self.update_action_mask()

    # ---- Implementation of abstract methods from `BaseTrajectory`

    @staticmethod
    def log_probability_fn(trajectories: List['SequenceDesign'], network: nn.Module, config: SequenceConfig, device: torch.device, 
                           esm3_model: None) -> List[np.array]:
        
        """
        Given a list of trajectories and a policy network,
        returns a list of numpy arrays, each having length num_actions, where each numpy array is a log-probability
        distribution over the next action level.

        Parameters:
            trajectories [List[BaseTrajectory]]
            network [torch.nn.Module]: Policy network
        Returns:
            List of numpy arrays, where i-th entry corresponds to the log-probabilities for i-th trajectory.

        """
        log_probs_to_return: List[np.array] = []
        device = torch.device("cpu") if device is None else device
        network.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=config.training_device, dtype=torch.bfloat16):
                batch_embeddings = esm3_model.get_batch_esm_embeddings(sequences=trajectories)
                batch = SequenceDesign.list_to_batch(sequences=trajectories, batch_embeddings= batch_embeddings, device=network.device)
                batch_logits_per_level = list(network(batch))
                for lvl in range(2):
                    batch_logits_per_level[lvl] = batch_logits_per_level[lvl].to(torch.float32).cpu().numpy()
                for i, seq in enumerate(trajectories):

                    # get logits for this sequence and corresponding level
                    if seq.current_action_level == 0:
                        logits = batch_logits_per_level[0][i]
                    else:
                        logits = batch_logits_per_level[1][i]
                        logits = logits[seq.picked_position]
                    log_probs_to_return.append(seq.masked_log_probs_for_current_action_level(logits))

        return log_probs_to_return
    

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        copied_sequence= copy.deepcopy(self)
        copied_sequence.take_action(action)
        is_terminable = (len(copied_sequence.changed_positions) >= copied_sequence.num_masked_sites
        and copied_sequence.current_action_level == 0)
        if is_terminable:
            # terminating design if mutational budget reached
            copied_sequence.identifier = self.generate_custom_sequence_id()
            copied_sequence.design_done = True     
        return copied_sequence, is_terminable

    def to_max_evaluation_fn(self) -> float:
        if self.objective is None:
            raise ValueError("Objective is `None`. Evaluate Sequence with `SequenceObjectiveEvaluator` first.")

        return self.objective

    def num_actions(self) -> int:
        
        """
        Returns number of current _feasible_ actions.
        """
        return int((1 - self.current_action_mask).sum())

    def generate_custom_sequence_id(self):

        """
            Generates a custom identifier for a generated finished design 

        """
        random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=6))
        return f"{random_chars}"
    
    
    @staticmethod
    def list_to_batch(sequences: List['SequenceDesign'], batch_embeddings, include_feasibility_masks: bool = False, device: torch.device = None) -> dict:
        """
        Given a list of sequence designs, prepares a batch that can be passed through the network.

        The batch is given as a dictionary with the following keys and values:
        
        * "tokens_np": direct embeddings from ESM model
        * "valid_positions": a mask tensor specifying valid positions

        if `include_feasibility_masks` is set to True, we also return
        """

        assert len(sequences) > 0, "Empty batch of sequences" 

        #selected position 
        batch_selected_position = [seq.picked_position if seq.picked_position is not None else -1 for seq in sequences]


        return_dict = dict(
            batch_selected_position = torch.tensor(batch_selected_position, device=device),
            embeds = batch_embeddings["batch_embeddings"],   # (B, L)
            valid_positions = batch_embeddings["valid_positions"]  # (B, L)
        )

        if include_feasibility_masks:
            # Build per-level feasibility masks, padded across the batch to each level's max action count.
            feasibility_mask_per_level = []

            num_actions_per_level_and_seq = [
                [len(seq.residues) for seq in sequences],  # lvl 0 
                [len(seq.vocabulary_residue_names) for seq in sequences],  # lvl 1
            ]

            for lvl, num_actions_per_seq in enumerate(num_actions_per_level_and_seq):
                max_num_actions = max(num_actions_per_seq)
                numpy_mask = torch.from_numpy(
                    np.stack([
                        np.pad(
                            ~seq.current_action_mask.astype(bool),
                            (0, max_num_actions - seq.current_action_mask.shape[0]),
                            mode='constant', constant_values=True
                        ) if seq.current_action_level == lvl 
                        else np.zeros(max_num_actions, dtype=bool)
                        for i, seq in enumerate(sequences)
                    ])).bool().to(device)
                
                feasibility_mask_per_level.append(numpy_mask)
            
            # Add to return_dict
            return_dict["feasibility_mask_level_zero"] = feasibility_mask_per_level[0] 
            return_dict["feasibility_mask_level_one"] = feasibility_mask_per_level[1]  

        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """
            Takes batch as returned from `list_to_batch` and moves it onto the given device.
        """
        return {k: v.to(device) for k, v in batch.items()}
        
    @staticmethod
    def available_existing_positions(static_action_map) -> List[int]:

        """
        Positions eligible for replacement:
        - currently assigned (not None)
        - AND were initially masked.
        """

        return [
            i for i, (status, pos) in static_action_map.items()
            if (pos is not None and status != "fixed")
        ]
            
    @staticmethod
    def design_sequences(config: SequenceConfig, seed_sequence: str) -> List['SequenceDesign']:

        """
        Returns list of designs based on the starting point and mode.
        """
        instance_list = []
        min_mutations, max_mutations = config.min_max_mutations[0], config.min_max_mutations[1]
        depths = list(range(min_mutations, (max_mutations+1))) * config.multiplier
        seed_sequence = seed_sequence
        seed_length = len(seed_sequence)
        for depth in depths:
            instance_list.append(SequenceDesign(config=config, length=seed_length, input_sequence=seed_sequence, num_masked_sites=depth))
        
        return instance_list









