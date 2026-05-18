from typing import Optional, Tuple, List, Dict

import torch
import pickle
import random
from torch.utils.data import Dataset
import copy
from config import SequenceConfig
from sequence_design import SequenceDesign
from tqdm import tqdm

def _clone_sequence(seq: SequenceDesign) -> SequenceDesign:
    seq_copy = copy.deepcopy(seq)
    return seq_copy

def precompute_flat_dataset(instances, config):

    """
    Computes flattened training datapoints from instances and saves all partial states. 
    We want to uniformly sample from partial sequences. So for each instance, check how many partial mutated sequences
    there are, and create a list of them where each entry is a tuple (int, int), where first entry is index of
    the instance, and second entry is the index in the action sequence which is the training target.

    """

    flat_sequences = [] # state BEFORE taking the target action
    flat_targets = [] # the target action (int)
    flat_levels = [] # current_action_level at that state
    tuple_to_flat : Dict[Tuple[int, int], int] = {}
    targets_to_sample : List[Tuple[int, int]] = [] # (instance_idx, target_idx)

    for i, instance in tqdm(enumerate(instances), total=len(instances), desc="Computing dataset"):
        length = len(instance["residues"]) 
        level_list = instance["level_list"]

        seed_sequence = instance["seed_sequence"]
        seq = SequenceDesign(config, length, input_sequence=seed_sequence, num_masked_sites=instance["num_masked_sites"])
        sequence_of_actions_idx = list(range(len(instance["action_seq"])))
        targets_to_sample.extend([(i, j) for j in sequence_of_actions_idx])

        for j, action in enumerate(instance["action_seq"]):
            # store state before taking this action
            flat_idx = len(flat_targets)
            tuple_to_flat[(i, j)] = flat_idx

            seq_copy = _clone_sequence(seq)
            flat_sequences.append(seq_copy)
            flat_targets.append(action)

            seq.current_action_level = level_list[j]
            flat_levels.append(seq.current_action_level)
            seq.take_action(action)

            # advance environment by one action

    return {
        "flat_sequences": flat_sequences,
        "flat_targets": flat_targets,
        "flat_levels": flat_levels,
        "tuple_to_flat": tuple_to_flat,
        "targets_to_sample": targets_to_sample,
    }



class PolicyTrainingDataset(Dataset):

    """

    Dataset for supervised training of the protein sequence design given as a list pseudo-expert sequence.
    Each sequence is given as a dictionary with the following keys and values
          "start_residue": [int] the int representing the residue from which to start
          "action_seq": List[List[int]] Actions which need to be taken on each index to create the sequence
          "smiles": [str] Corresponding sequence string
          "obj": [float] Objective function evaluation

    Each datapoint in this dataset is a partial sequence: We sample an instance, randomly choose an index up to which
    all actions will be performed. Then, ending up at action index 0, we take the next item in the action seq
    (which corresponds to a list all actions that need to be taken from index to index) as training target.

    """
    def __init__(self, config: SequenceConfig, path_to_pickle: str, batch_size: int, custom_num_batches: Optional[int],
                 no_random: bool = False, esm3_model = None, if_pretrain=None, proxy= None):
        self.config = config
        self.batch_size = batch_size
        self.custom_num_batches = custom_num_batches
        self.path_to_pickle = path_to_pickle
        self.esm3_model = esm3_model
        self.if_pretrain = if_pretrain
        self.proxy = proxy
        
        with open(path_to_pickle, "rb") as f:
            self.instances = pickle.load(f)  # list of dictionaries

        cached = precompute_flat_dataset(instances=self.instances, config=config)

        self.targets_to_sample = cached["targets_to_sample"]
        self._flat_sequences = cached["flat_sequences"]
        self._flat_targets = cached["flat_targets"]
        self._flat_levels = cached["flat_levels"]
        self._tuple_to_flat = cached["tuple_to_flat"]

        if custom_num_batches is None:
            self.length = len(self.targets_to_sample) // self.batch_size 
        else:
            self.length = custom_num_batches

        self.no_random = no_random

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        """
        :param idx: is not used, as we directly randomly sample a full batch from the datapoints here.

        Returns: Dictionary with keys:

        """
        partial_sequences: List[SequenceDesign] = []   # partial sequences which will become the batch
        instance_targets: List[List[int]] = []  # corresponding targets taken from the instances

        if self.no_random:
            batch_to_pick = self.targets_to_sample[idx * self.batch_size: (idx+1) * self.batch_size]
        else:
            batch_to_pick = random.choices(self.targets_to_sample, k=self.batch_size)  # with replacement

        # Map each (instance_idx, target_idx) to the precomputed flat index
        flat_indices = [self._tuple_to_flat[tup] for tup in batch_to_pick]

        # Gather precomputed partial sequences and their targets/levels
        partial_sequences = [self._flat_sequences[k] for k in flat_indices]
        instance_targets  = [self._flat_targets[k]   for k in flat_indices]
        levels_list       = [self._flat_levels[k]    for k in flat_indices]

        # Create the input batch from the partial sequences.
        batch_embeddings = self.esm3_model.get_batch_esm_embeddings(sequences=partial_sequences)
        batch_input = SequenceDesign.list_to_batch(sequences=partial_sequences, batch_embeddings= batch_embeddings, device=torch.device("cpu"), include_feasibility_masks=True)

        # We now create the targets. We separate it into targets for level 0 and 1.
        # We only set the target action as target for the current level the sequence is in.
        # For all other levels, we set it to -1 for a sequence. (ignore)
        
        # Vectorized level-specific targets
        targets = torch.tensor(instance_targets, dtype=torch.long)   
        levels  = torch.tensor(levels_list,      dtype=torch.long)   

        batch_targets = [
            torch.where(levels == level, 
                        targets, # # We only set the target action as target for the current level the sequence is in.
                        torch.full_like(targets, -1)) # # For all other levels, we set it to -1 for a sequence.
            for level in (0, 1)
        ]

        return dict(
            input=batch_input,
            target_zero=batch_targets[0],
            target_one=batch_targets[1]
        )
