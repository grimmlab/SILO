import torch, os, time, pickle, core.sequence_fitness_dataset, random
import numpy as np
from model.transformer_architecture import SequenceTransformer
from sequence_evaluator import SequenceEvaluator
from sequence_dataset import PolicyTrainingDataset
from sequence_evaluator import SequenceEvaluator 
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from config import SequenceConfig
import argparse


def save_checkpoint(checkpoint: dict, filename: str, config: SequenceConfig):
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    torch.save(checkpoint, path)


class MetricsTracker:

    """
    Tracks metrics per round, namely max, mean, and median fitnesses, novelty, and diversity
    """
    def __init__(self, config, dataset, wt_sequence=None):
        self.config = config
        self.dataset = dataset
        best_percentile=0.95
        self.best_percentile= best_percentile
        self.best_so_far = -np.inf
        self.collected_sequences = list()
        self.collected_scores = list()
        self.wt_sequence = config.tasks_configs['wt_sequences'][config.tasks_configs['task']]
        self.train_sequences = self.dataset.train if self.wt_sequence is None else self.get_best_sequences()

    def hamming_distance(self, seq1: str, seq2: str) -> int:

        """Hamming distance between two equal-length strings."""
        if len(seq1) != len(seq2):
            raise ValueError(f"Sequences must have same length: {len(seq1)} vs {len(seq2)}")
        return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
    
    def mean_novelty(self, seqs, ref_seqs):

        if not isinstance(seqs[0], str):
            seqs = [s["smiles"] for s in seqs]

        A = np.array([list(s) for s in seqs])
        B = np.array([list(s) for s in ref_seqs])

        dists = (A[:, None, :] != B[None, :, :]).sum(axis=2)

        novelties = dists.min(axis=1)

        return novelties.tolist()

    def mean_pairwise_distances(self, seqs):
        if not isinstance(seqs[0], str):
            seqs = [s["smiles"] for s in seqs]

        A = np.array([list(s) for s in seqs])

        dists = (A[:, None, :] != A[None, :, :]).sum(axis=2)

        i, j = np.triu_indices(len(seqs), 1)

        return dists[i, j].tolist()
    
    def add_records(self, trajectories):

        sorted_traj = sorted(trajectories.items(), key=lambda x: x[1][1], reverse=True)
        new_sequences = [x[1][0] for x in sorted_traj]
        new_scores = [x[1][1] for x in sorted_traj]

        self.collected_sequences += new_sequences
        self.collected_scores += new_scores

        return new_sequences, new_scores

    def get_best_sequences(self):
        
        train_sequences = self.dataset.train
        train_scores = self.dataset.train_scores
        thr = np.percentile(train_scores, self.best_percentile)
        return train_sequences[train_scores >= thr].tolist()

    def calculate_top_metrics(self, epoch, trajectories):

        new_sequences, new_scores = self.add_records(trajectories)
        indices = np.argsort(self.collected_scores)[::-1][:100]
        top_sequences = np.array(self.collected_sequences)[indices]
        top_scores = np.array(self.collected_scores)[indices]
        
        top_distances = self.mean_pairwise_distances(top_sequences)
        top_novelties = self.mean_novelty(top_sequences, self.train_sequences)
        top_novelty_wt = self.mean_novelty(top_sequences, [self.wt_sequence])

        current_best = np.max(top_scores)

        exp_results = {
            "Round": epoch + 1,
            "Diversity": np.mean(top_distances),
            "Novelty": np.mean(top_novelties),
            "Best Fitness per round": np.max(new_scores),
            "Mean Fitness": np.mean(top_scores),
            "Best Fitness so far": max(self.best_so_far, current_best),
            "novelty to wt": np.mean(top_novelty_wt), 
            "Median": np.percentile(top_scores, 50)
        }

        return exp_results

def save_pickle_file(config, sequences, path_to_pickle_file):


    """
    Takes stored pickle file after wor sampling as input. If the dataset already exists at the path where to save, we load it, merge them and take the best from the
        merged dataset.

        Then returns the following dictionary:
        - "mean_gen_obj": Mean generated obj. -> over the unmerged best sequences generated
        - "best_gen_obj": Best generated obj. -> Best obj. of the unmerged sequences generated
        - "worst_gen_obj": Worst generated obj. -> Worst obj. of the unmerged sequences generated
        - "mean_top_20_obj": Mean top 20 obj. -> over the merged best sequences
        - "mean_kept_obj": Mean of num_trajectories_to_keep sequences 
        - "top_20_sequences": A list with obj. of the top 20 obj.
    
    """

    metrics_return = dict()
    destination_path = path_to_pickle_file

    gen_seqs = sorted(sequences, key=lambda x:  x["objective_dict"]["tape"], reverse=True)
    generated_objs = np.array([x["objective_dict"]["tape"]for x in gen_seqs])
    metrics_return["mean_gen_obj"] = generated_objs.mean()
    metrics_return["best_gen_obj"] = generated_objs[0]
    metrics_return["worst_gen_obj"] = generated_objs[-1]

    merged_seqs = sequences
    temp_d = {x["smiles"]: x for x in merged_seqs}
    merged_seqs = list(temp_d.values())
    if destination_path is not None:
        if os.path.isfile(destination_path):
            with open(destination_path, "rb") as f:
                existing_seqs = pickle.load(f)  # list of dicts
            temp_d = {x["smiles"]: x for x in existing_seqs + merged_seqs}
            merged_seqs = list(temp_d.values())
            merged_seqs = sorted(merged_seqs, key=lambda x: x["objective_dict"]["tape"], reverse=True)[
                                :config.self_improvement_learning['num_trajectories_to_keep']]
        else:
            merged_seqs = sorted(merged_seqs, key=lambda x: x["objective_dict"]["tape"], reverse=True)[
                                :config.self_improvement_learning['num_trajectories_to_keep']]
        # Pickle the generated data again
        with open(destination_path, "wb") as f:
            pickle.dump(merged_seqs, f)

    all_generated_seqs = sorted(merged_seqs, key=lambda x: x["objective_dict"]["tape"], reverse=True)
    metrics_return["mean_top_20_obj"] = np.array([x["objective_dict"]["tape"] for x in all_generated_seqs[:20]]).mean()
    metrics_return["mean_kept_obj"] = np.array([x["objective_dict"]["tape"]for x in all_generated_seqs]).mean()
    metrics_return["top_20_sequences"] = [{x["identifier"]: x["objective_dict"]["tape"]for x in all_generated_seqs[:20]}]

    return metrics_return 


def train_for_one_epoch_active_cycle(epoch: int, config: SequenceConfig, network: SequenceTransformer, network_weights: dict,
                        optimizer: torch.optim.Optimizer, objective_evaluator: SequenceEvaluator, best_objective: float,
                        esm3_model=None, seen_protein_smiles=None, proxy=None, metrics_logger= None, logger=None):
    
    """
        Main training loop for updating the policy in each active learning round.

        Overview:
            Each round consists of (1) sampling candidate sequences using the current policy and (2) updating the policy using high-quality trajectories selected via
            oracle feedback.

        Full explaination:
            1. Sampling:
                - Use the current policy (via its logits) together with Stochastic Beam Search (SBS) using the 'SequenceFitnessDataset'
                to sample action trajectories.
                - Each trajectory corresponds to a sequence of edit actions that produce a mutant sequence.

            2. Scoring with trained surrogate model:
                - Evaluate sampled sequences using a trained proxy.
                - Compute:
                    (a) UCB-based surrogate fitness 
                    (b) Alanine scan score
                - Combine these scores into a single objective used for ranking candidates.

            3. Candidate selection and oracle evaluation:
                - Greedily select the top-K sequences (e.g., 128) according to the combined objective.
                - Query the oracle to obtain ground-truth fitness values for these sequences.

            4. Trajectory selection for policy training:
                - From the oracle-evaluated set, select the top `num_trajectories_to_keep`
                sequences based on true objective values.
                - These define the high-quality trajectories used as supervision.

            5. Policy update:
                - Convert selected trajectories into (state, action) training pairs using
                the `PolicyTrainingDataset`.
                - Train the policy via next-token prediction (behavior cloning) using a
                cross-entropy loss over actions.
    
    """

    sequence_fitness_dataset = core.sequence_fitness_dataset.SequenceFitnessDataset(config=config, objective_evaluator=objective_evaluator, esm3_model=esm3_model, 
                                           seen_protein_smiles=seen_protein_smiles, proxy=proxy)
    

    final_candidates = sequence_fitness_dataset.generate_dataset(network_weights, best_objective=best_objective, memory_aggressive=False)
    top_trajectories = final_candidates 
    
    # Save sequences in csv file and pickle file 
    logger.save_results_csv(config= config, trajectories = final_candidates, path_csv_file=os.path.join(config.results_path, f"{config.tasks_configs['task']}_train_results.csv"))
    metrics = save_pickle_file(config, final_candidates, os.path.join(config.results_path, f"{config.tasks_configs['task']}_oracle_train.pickle"))

    
    if len(metrics) > 0:

        print("Generated Sequences")
        print(f"Mean obj. over fresh best seqs: {metrics['mean_gen_obj']:.3f}")
        print(f"Best / worst obj. over fresh best seqs: {metrics['best_gen_obj']:.3f}, {metrics['worst_gen_obj']:.3f}")
        print(f"All time best sequence: {list(metrics['top_20_sequences'][0].values())[0]:.3f}")

        torch.cuda.empty_cache()
        time.sleep(1)
        print("---- Loading dataset")

        dataset = PolicyTrainingDataset(config, os.path.join(config.results_path, f"{config.tasks_configs['task']}_oracle_train.pickle"), batch_size=config.batch_size_training,
                                    custom_num_batches=config.num_batches_per_epoch, no_random=True, esm3_model=esm3_model, proxy=proxy)

        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, 
                                persistent_workers=False)
        
        scaler = torch.amp.GradScaler()
        criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)

        epoch_losses = []
        epoch_losses_zero = []
        epoch_losses_one = []


        # Train for n epochs 
        for epoch in range(config.num_epochs):
            network.train()

            accumulated_loss_lvl_zero = 0
            accumulated_loss_lvl_one = 0
            accumulated_total_loss = 0

            num_batches = len(dataloader)
            progress_bar = tqdm(range(num_batches))
            data_iter = iter(dataloader)

            for _ in progress_bar:
                data = next(data_iter)
                input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
                
                # targets for the logit levels
                target_zero = data["target_zero"][0].to(network.device)
                target_one = data["target_one"][0].to(network.device)

                # Optimization step
                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=config.training_device, dtype=torch.bfloat16): 
                    
                    logits_zero, logits_one = network(input_data)

                    # We mask the output according to feasibility
                    logits_zero[input_data["feasibility_mask_level_zero"]] = float("-inf")
                    
                    # loss is calculated only for the logits of the selected position
                    B = logits_one.size(0)
                    batch_idx = torch.arange(B, device=logits_one.device)
                    
                    selected_pos = input_data["batch_selected_position"].squeeze(0)
                    logits_one_for_selected_position = logits_one[batch_idx, selected_pos] 
                    logits_one_for_selected_position[input_data["feasibility_mask_level_one"]] = float("-inf")

                    loss_zero = criterion(logits_zero, target_zero)
                    loss_zero = torch.tensor(0.) if torch.isnan(loss_zero) else loss_zero
                    
                    loss_one = criterion(logits_one_for_selected_position, target_one)
                    loss_one = torch.tensor(0.) if torch.isnan(loss_one) else loss_one
                    loss = loss_zero + loss_one  

                scaler.scale(loss).backward()
                if config.optimizer["gradient_clipping"] > 0:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                accumulated_total_loss += loss.item()
                accumulated_loss_lvl_zero += loss_zero.item()
                accumulated_loss_lvl_one += loss_one.item()

            epoch_loss = accumulated_total_loss / num_batches
            epoch_loss_zero = accumulated_loss_lvl_zero / num_batches
            epoch_loss_one = accumulated_loss_lvl_one / num_batches

            epoch_losses.append(epoch_loss)
            epoch_losses_zero.append(epoch_loss_zero)
            epoch_losses_one.append(epoch_loss_one)

            print(f"Epoch {epoch+1}/{config.num_epochs} | " f"total_loss={epoch_loss:.4f}")

            del data 

        metrics["loss_level_zero"] = epoch_losses_zero[-1]
        metrics["loss_level_one"] = epoch_losses_one[-1]
        metrics["total_loss"] = epoch_losses[-1]

        del metrics["top_20_sequences"]

        return metrics, top_trajectories


def set_seed(seed=0, full_deterministic=True):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if full_deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=False)
            # Enable CuDNN deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
