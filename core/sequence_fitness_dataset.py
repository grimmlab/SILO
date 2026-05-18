
import copy, os, sys, ray, torch 
from model.transformer_architecture import SequenceTransformer
from sequence_design import SequenceDesign
import numpy as np
from ray.thirdparty_files import psutil
from tqdm import tqdm
import pandas as pd
from core.abstract import Config, Instance
import core.stochastic_beam_search as sbs
from typing import List, Tuple, Any, Optional
from core.incremental_sbs import IncrementalSBS
from config import SequenceConfig
from sequence_evaluator import SequenceEvaluator
from objectives.utils import oracle_fitness_scores
os.environ["RAY_DEDUP_LOGS"] = "0"


@ray.remote
class JobPool:
    def __init__(self, problem_instances: List[Instance]):
        self.jobs = [(i, instance) for i, instance in enumerate(problem_instances)]
        self.job_results = []

    def get_jobs(self, n_items: int):
        if len(self.jobs) > 0:
            items = self.jobs[:n_items]
            self.jobs = self.jobs[n_items:]
            return items
        else:
            return None

    def push_results(self, results: List[Tuple[int, Any]]):
        self.job_results.extend(results)

    def fetch_results(self):
        results = self.job_results
        self.job_results = []
        return results


class SequenceFitnessDataset:
    def __init__(self, config: SequenceConfig,
                 objective_evaluator: SequenceEvaluator, 
                 esm3_model=None, 
                 seen_protein_smiles: dict = None, 
                 proxy = None
                ):
        self.config = config
        self.self_improvement_learning = config.self_improvement_learning
        self.objective_evaluator = objective_evaluator
        self.devices_for_workers: List[str] = self.self_improvement_learning["devices_for_workers"]
        self.esm3_model = esm3_model
        self.seen_protein_smiles = seen_protein_smiles
        self.proxy = proxy


    def generate_dataset(self, network_weights: dict, best_objective: Optional[float] = None, memory_aggressive: bool = False):
        """
        Parameters:
            network_weights: [dict] Network weights to use for generating data.
            memory_aggressive: [bool] If True, IncrementalSBS is performed "memory aggressive" meaning that
                intermediate states in the search tree are not stored after transitioning from them, only their
                policies.
        """
        batch_size_gpu, batch_size_cpu = (self.self_improvement_learning["batch_size_per_worker"],
                                          self.self_improvement_learning["batch_size_per_cpu_worker"])

        # starting sequence is the one with the highest score
        starting_seqs = sorted(self.seen_protein_smiles, key=self.seen_protein_smiles.get, reverse=True)[0:1]

        problem_instances = []

        for start_seq in starting_seqs:
            problem_instance = SequenceDesign.design_sequences(config=self.config, seed_sequence=start_seq)
            for instance in problem_instance:
                problem_instances.append(instance)

        job_pool = JobPool.remote(copy.deepcopy(problem_instances))
        results = [None] * len(problem_instances)
        self.num_trajectories_to_keep =  self.config.self_improvement_learning['num_trajectories_to_keep']

        # Check if we should pin the workers to core
        cpu_cores = [None] * len(self.devices_for_workers)
        if self.self_improvement_learning["pin_workers_to_core"] and sys.platform == "linux":
            # Get available core IDs
            affinity = list(os.sched_getaffinity(0))
            cpu_cores = [affinity[i % len(cpu_cores)] for i in range(len(self.devices_for_workers))]

        # Kick off workers
        future_tasks = [async_sbs_worker.remote
                        (self.config, job_pool, network_weights, device, batch_size_gpu if device != "cpu" else batch_size_cpu, 
                                    cpu_cores[i], best_objective, memory_aggressive, esm3_model = self.esm3_model, proxy = self.proxy)
            for i, device in enumerate(self.devices_for_workers)] 

        with tqdm(total=len(problem_instances)) as progress_bar:
            while True:
                # Check if all workers are done. If so, break after this iteration
                do_break = len(ray.wait(future_tasks, num_returns=len(future_tasks), timeout=0.5)[1]) == 0
                fetched_results = ray.get(job_pool.fetch_results.remote()) 
                for (i, result) in fetched_results:
                    results[i] = result
                if len(fetched_results):
                    progress_bar.update(len(fetched_results))
                if do_break:
                    break

        ray.get(future_tasks)
        del job_pool
        del network_weights
        torch.cuda.empty_cache()

        results_as_dict = self.sequence_object_to_dict(results)
        final_candidates = oracle_fitness_scores(trajectories=results_as_dict, objective_evaluator=self.objective_evaluator, seen_protein_smiles=self.seen_protein_smiles, config=self.config)
            
        return final_candidates

    def sequence_object_to_dict(self, results):

        """
            Processes the results from wor search into a dict to save it to as a pickle. Each trajectory will be represented as a dict with the
            following keys and values
            "identifier": Unique identifier for the trajectory/sequence
            "action_seq": List[List[int]] Actions which need to be taken on each index to create the sequence
            "residues": Index-level representation of sequence string.
            "level_list": A list of level number corresponding to action sequence 
            "seq_string": [str] Corresponding sequence string as a list
            "objective": [float] Objective function evaluation 
            "smiles": String representation of the sequence
            "surrogate_objective": Surrogate model estimate (e.g., UCB score).
            "objective_dict": Dictionary containing all computed objective components (e.g., surrogate, alanine scan, etc.).
            "num_masked_sites": Number of mutation sites considered."
            "seed sequence": Original sequence before applying edits 

        """

                
        all_results = []
        for i in range(0, len(results)):
            for res in results[i]:
                all_results.append(res)

        instances_dict = dict() 

        for seq in all_results: 
            smiles = ''.join(seq.seq_string)

            instances_dict[smiles] = dict(
            identifier= seq.identifier, 
            action_seq=seq.history,
            seq_string=seq.seq_string,
            residues = seq.residues, 
            level_list = seq.level_list,
            smiles=(''.join(seq.seq_string)),
            surrogate_objective=seq.objective_dict['ucb_surrogate'],
            objective = seq.objective,
            alanine_scan=seq.objective_dict['ucb_alanine_scan'], 
            num_masked_sites = seq.num_masked_sites,
            seed_sequence = seq.input_sequence,
            objective_dict= seq.objective_dict)
        
        return instances_dict
    

@ray.remote(max_calls=1)
def async_sbs_worker(config: Config, job_pool: JobPool, network_weights: dict,
                     device: str, batch_size: int,
                     cpu_core: Optional[int] = None,
                     best_objective: Optional[float] = None,
                     memory_aggressive: bool = False, 
                     esm3_model = None, 
                     proxy = None, 
                     ):
    def child_log_probability_fn(trajectories: List[SequenceDesign]) -> [np.array]:
        return SequenceDesign.log_probability_fn(config = config, trajectories=trajectories, network=network, device=device, esm3_model=esm3_model)
    
    def do_alanine_scanning(trajectories):
        batch_alan_seqs = []

        for seq in trajectories:
            seq_list = list(seq.seq_string)
            for pos in seq.changed_positions:
                seq_list[pos] = 'A'   # substitute alanine
            batch_alan_seqs.append(''.join(seq_list))
        
        alan_scores, mean, std_deviation = proxy.calculate_alan_fitness(batch_alan_seqs, k = 1.0)

        for seq, score, m, std in zip(trajectories, alan_scores, mean, std_deviation):
            seq.objective_dict['alanine_scan_mean'] = m
            seq.objective_dict['alanine_scan_uncertainty'] = std
            seq.objective_dict['ucb_alanine_scan'] = score
        
        return alan_scores 

    def batch_leaf_evaluation_fn(trajectories: List[SequenceDesign]) -> np.array:

        """
            Make fitness predictions using trained surrogate model and get scores of alanine scanning 

        """

        objs = proxy.calculate_proxy_fitness(trajectories, k=0.1)
        alan_scores = do_alanine_scanning(trajectories)

        for seq, score, alan_score in zip(trajectories, objs, alan_scores):
            seq.objective = float(score + alan_score)

        return objs + alan_scores

    
    def child_transition_fn(trajectory_action_pairs: List[Tuple[SequenceDesign, int]]):
        return [traj.transition_fn(action) for traj, action in trajectory_action_pairs]
 
    # Pin worker to core if wanted
    if cpu_core is not None:
        os.sched_setaffinity(0, {cpu_core})
        psutil.Process().cpu_affinity([cpu_core])

    with torch.no_grad():
        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

        device = torch.device(device)
        network = SequenceTransformer(config, config.training_device)
        network.load_state_dict(network_weights)
        network.to(network.device)
        
        network.eval()
        objective_evaluator = SequenceEvaluator(config)

        while True:
            batch = ray.get(job_pool.get_jobs.remote(batch_size))

            if batch is None:
                break

            idx_list = [i for i, _ in batch]
            root_nodes = [instance for _, instance in batch]

            if config.self_improvement_learning["search_type"] == "beam_search":
                # Deterministic beam search.
                beam_leaves_batch: List[List[sbs.BeamLeaf]] = sbs.stochastic_beam_search(
                    child_log_probability_fn=child_log_probability_fn,
                    child_transition_fn=child_transition_fn,
                    root_states=root_nodes,
                    beam_width=config.self_improvement_learning["beam_width"],
                    deterministic=True
                )
            else:
                inc_sbs = IncrementalSBS(config, root_nodes, child_log_probability_fn, child_transition_fn,
                                         leaf_evaluation_fn=SequenceDesign.to_max_evaluation_fn,
                                         batch_leaf_evaluation_fn=batch_leaf_evaluation_fn,
                                         memory_aggressive=False)
                
                if config.self_improvement_learning["search_type"] == "wor":
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = inc_sbs.perform_incremental_sbs(
                        beam_width=config.self_improvement_learning["beam_width"],
                        num_rounds=config.self_improvement_learning["num_rounds"],
                        nucleus_top_p=config.self_improvement_learning["nucleus_top_p"],
                        sbs_keep_intermediate=config.self_improvement_learning["keep_intermediate_trajectories"],
                        best_objective=best_objective
                    )
                else:
                    raise ValueError(f"Unknown search_type: {config.self_improvement_learning['search_type']}. ""Expected 'wor' or 'beam_search'.") 

            results_to_push = []
            for j, result_idx in enumerate(idx_list):
                result: List[SequenceDesign] = [x.state for x in beam_leaves_batch[j]]
                # Check if they need objective evaluation (this will only be true for deterministic beam search)
                if result[0].objective is None:
                    batch_leaf_evaluation_fn(result)
                results_to_push.append((result_idx, result))
            ray.get(job_pool.push_results.remote(results_to_push)) 

            if device != "cpu":
                torch.cuda.empty_cache()

    del network
    del network_weights
    torch.cuda.empty_cache()



