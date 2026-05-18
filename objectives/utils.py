import torch, os, json
import numpy as np 
from tape import ProteinBertForValuePrediction, TAPETokenizer
import requests
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from typing import List
import numpy as np


class ESM3():
    '''
    A wrapper around the esm C protein language model. Download the model via 
    huggingface-cli download EvolutionaryScale/esmc-300m-2024-12 \
    --local-dir ./esmc-300m\
    --local-dir-use-symlinks False

    '''

    def __init__(self, config):
        self.device = torch.device(config.training_device)
        self.model = ESMC.from_pretrained("esmc_300m").to(self.device)
        self.model.eval()
        self.config = config
        self.embedding_config = LogitsConfig(sequence=True, return_embeddings=True)

    
    def extract_bulk_embeddings_per_residue(self, sequences) -> List:

        """
        Extracts embeddings from ESM3 model for residues within a sequence.

        """

        proteins = [ESMProtein(sequence=seq) for seq in sequences]
        protein_tensors = [self.model.encode(p) for p in proteins]
        outputs = []

        with torch.no_grad():
            for pt in protein_tensors:
                out = self.model.logits(pt, self.embedding_config)
                emb = out.embeddings[0, 1:-1, :]
                outputs.append(emb)   

        return outputs
    
    
    def get_batch_esm_embeddings(self, sequences): 

        seq_strings = [self.join_partial(seq.seq_string) for seq in sequences]
        max_aa_per_seq = max(len(seq.residues) for seq in sequences) # always include virtual residues for embeddings
        valid_positions = torch.zeros(len(sequences), max_aa_per_seq, dtype=torch.bool, device=sequences[0].config.training_device)
        seq_esm_embeds = torch.zeros(len(sequences), max_aa_per_seq, sequences[0].config.esm_emb_size, device=sequences[0].config.training_device) #this is also considering virtual node already 
        batch_esm_embeds = self.extract_bulk_embeddings_per_residue(seq_strings)

        for i, seq in enumerate(sequences): 
            valid_positions[i, 0:(len(seq.residues))] = True
            seq_esm_embeds[i, 0:(len(seq.residues))] = batch_esm_embeds[i]
        
        return {
            "batch_embeddings": seq_esm_embeds, 
            "valid_positions": valid_positions
            }
    
    def join_partial(self, sequence):

        """
        Joins a list of residues and replaces None with <mask> for esm. 

        """
        return "".join([aa if aa is not None else "<mask>" for aa in sequence])
    

class TAPELandscape():
    
    """
    TAPE based oracle model to simulate protein fitness landscape 
    from TAPE (https://github.com/songlab-cal/tape).

    Note that the output of this landscape is not normalized to be between 0 and 1

    """
    def __init__(self, config, oracle_path: str = './objectives', noise=0.0):
        self.config = config 
        self.device = self.config.training_device
        self.noise = noise 
        if self.config.tasks_configs['task'] == 'GFP':
            if not os.path.exists(f"{oracle_path}/oracles/fluorescence-model"):
                os.mkdir(f"{oracle_path}/oracles/fluorescence-model")
                gfp_model_path = "https://fluorescence-model.s3.amazonaws.com/fluorescence_transformer_20-05-25-03-49-06_184764/" 
                for file_name in [
                        "args.json",
                        "checkpoint.bin",
                        "config.json",
                        "pytorch_model.bin",
                    ]:
                        print("Downloading", file_name)
                        response = requests.get(gfp_model_path + file_name)
                        with open(f"{oracle_path}/oracles/fluorescence-model/{file_name}", "wb") as f:
                            f.write(response.content)
                self.model = ProteinBertForValuePrediction.from_pretrained(f"{oracle_path}/oracles/fluorescence-model/").to(self.device)
            else:
                self.model = ProteinBertForValuePrediction.from_pretrained(f"{oracle_path}/oracles/fluorescence-model/").to(self.device)
        else:
            task_dir_path = os.path.join(f'{oracle_path}/oracles/', self.config.tasks_configs['task'])
            print(task_dir_path)
            assert os.path.exists(os.path.join(task_dir_path, 'pytorch_model.bin'))
            self.model = ProteinBertForValuePrediction.from_pretrained(task_dir_path).to(self.device)

        self.tokenizer = TAPETokenizer(vocab='iupac')

    @torch.no_grad()
    def get_fitness(self, sequences):

        if isinstance(sequences, str):
            seq_string = [sequences]
            single_input = True
        else:
            if isinstance(sequences[0], dict):
                seq_string = [seq['smiles'] for seq in sequences]
            elif isinstance(sequences[0], str):
                seq_string = sequences
            else:
                seq_string = [(''.join(seq.seq_string)) for seq in sequences]
            single_input = False

        scores = []

        if self.config.noise_mode == True: 
            for i in range(0, len(seq_string), 32):
                subset = seq_string[i:i + 32]
                encoded_seqs = torch.tensor([self.tokenizer.encode(seq) for seq in subset]).to(self.device)
                raw_score = self.model(encoded_seqs)[0].detach().cpu().numpy().astype(float).reshape(-1)
                score_with_noise = raw_score + np.random.normal(scale=self.noise)
                scores.append(score_with_noise)
        else:
            for i in range(0, len(seq_string), 32):
                subset = seq_string[i:i + 32]
                encoded_seqs = torch.tensor([self.tokenizer.encode(seq) for seq in subset]).to(self.device)
                scores.append(self.model(encoded_seqs)[0].detach().cpu().numpy().astype(float).reshape(-1))

        scores = np.concatenate(scores)
        
        if single_input == True:
            return scores[0]
        else:
            for seq, score in zip(sequences, scores):
                if isinstance(sequences[0], dict):
                    seq['objective_dict']['tape'] = score
                elif isinstance(sequences[0], str):
                    return scores
                else:
                    seq.objective_dict['tape'] = score 
        return scores

class AAVLandscape:

    """
        Create AdditiveAAVPackaging landscape.

        Args:
            phenotype: One of "heart", "lung", "kidney", "liver", "blood", or "spleen".
            start: Starting index of AAV subsequence to evaluate.
            end: Ending index of AAV subsequence to evaluate.
            noise: Standard deviation of gaussian noise to add to landscape.
        Taken from FLEX github repository: https://github.com/samsinai/FLEXS/blob/master/flexs/landscapes/additive_aav_packaging.py
    
            
    """

    def __init__(
        self,
        oracle_path: str,
        phenotype: str = "liver",
        minimum_fitness_multiplier: float = 1,
        start: int = 450,
        end: int = 540,
        noise: int = 0,
    ):
        self.sequences = {}
        self.phenotype = f"log2_{phenotype}_v_wt"

        self.mfm = minimum_fitness_multiplier
        self.start = start
        self.end = end
        self.noise = noise

        with open(f'{oracle_path}/oracles/AAV2_single_subs-2.json') as f:
            self.data = {
                int(pos): val
                for pos, val in json.load(f).items()
                if self.start <= int(pos) < self.end
            }

        self.top_seq, self.max_possible = self.compute_max_possible()

    def compute_max_possible(self):
        """Compute max possible fitness of any sequence (used for normalization)."""
        best_seq = ""
        max_fitness = 0
        for pos in self.data:
            current_max = -10
            current_best = "M"
            for aa in self.data[pos]:
                current_fit = self.data[pos][aa][self.phenotype]
                if (
                    current_fit > current_max
                    and self.data[pos][aa]["log2_packaging_v_wt"] > -6
                ):
                    current_best = aa
                    current_max = current_fit

            best_seq += current_best
            max_fitness += current_max
        return best_seq, max_fitness

    def _get_raw_fitness(self, seq):
        total_fitness = 0
        for i, s in enumerate(seq):
            if s in self.data[self.start + i]:
                total_fitness += self.data[self.start + i][s][self.phenotype]

        return total_fitness + self.mfm * self.max_possible

    @torch.no_grad()
    def get_fitness(self, sequences):
        
        if isinstance(sequences, str):
            seq_string = [sequences]
            single_input = True
        else:
            if isinstance(sequences[0], dict):
                seq_string = [seq['smiles'] for seq in sequences]
            else:
                seq_string = [(''.join(seq.seq_string)) for seq in sequences]
            single_input = False

        fitnesses = []
        for seq in seq_string:
            normed_fitness = self._get_raw_fitness(seq) / (
                self.max_possible * (self.mfm + 1)
            )
            fitness_with_noise = normed_fitness + np.random.normal(scale=self.noise)
            fitnesses.append(max(0, fitness_with_noise))
        
        if single_input == True:
            return fitnesses[0]
        else:
            for seq, score in zip(sequences, fitnesses):
                if isinstance(sequences[0], dict):
                    seq['objective_dict']['tape'] = score
                else:
                    seq.objective_dict['tape'] = score 

        return np.array(fitnesses)

def oracle_fitness_scores(trajectories, objective_evaluator, 
                          seen_protein_smiles, config) -> np.array:
    local_seen = set()
    new_unique = []

    if isinstance(trajectories, dict):
        iterator = [traj for _, traj in trajectories.items()]
    elif isinstance(trajectories, list):
        iterator = [traj for traj in trajectories] 

    # Identify which trajectories are new unique SMILES
    if config.self_improvement_learning['search_type'] == 'beam_search':
        for traj in iterator:
            smiles = traj['smiles']
            new_unique.append(traj)
    else:
        for traj in iterator:
            smiles = traj['smiles']
            # duplicate within this batch
            if smiles in local_seen:
                continue
            local_seen.add(smiles)

            # already evaluated previously (global cache)
            if smiles not in seen_protein_smiles:
                # candidate for oracle evaluation
                new_unique.append(traj)

    final_trajs = best_candidates_for_oracle(new_unique, 
                                             config.self_improvement_learning["max_oracle_calls_per_round"])

    raw_objs = objective_evaluator.calculate_scores(final_trajs)
    for traj, score in zip(final_trajs, raw_objs):
        seen_protein_smiles[traj['smiles']] = float(score)  # store in cache

    final_trajs = sorted(final_trajs, key=lambda x: x['objective_dict']['tape'], reverse=True)

    return final_trajs

def best_candidates_for_oracle(sequences, oracle_budget):
    final_candidates = sorted(sequences, key=lambda x: x['objective'], reverse=True)

    return final_candidates[0:oracle_budget]


class NoisyLandscape:
    def __init__(self, ensemble_size, snr, signal_variance, config):
        noise_std = self.calculate_noise_std(snr, signal_variance)
        self.config = config
        self.oracle = [
            TAPELandscape(self.config, self.config.oracle_path, noise=noise_std) for _ in range(ensemble_size)
        ]

    def calculate_noise_std(self, snr, signal_variance):
        noise_var = signal_variance * 10 ** (-snr / 10)
        return np.sqrt(noise_var)

    def _call_models(self, sequences):
        return torch.stack([torch.Tensor(o.get_fitness(sequences)) for o in self.oracle])

    def get_fitness(self, sequences):
        outputs = self._call_models(sequences)
        return outputs.mean(dim=0)

    def get_scores(self, sequences):
        return self._call_models(sequences).mean(dim=0)

    def forward_with_uncertainty(self, sequences):
        outputs = self._call_models(sequences)
        return outputs.mean(dim=0), outputs.std(dim=0)
    
    def calculate_proxy_fitness(self, sequences, k):

        mean, std_deviation = self.forward_with_uncertainty(sequences)
        predictions = mean + (k * std_deviation)
        scores = predictions.detach().cpu().numpy()
        mean = mean.detach().cpu().numpy()
        std_deviation = std_deviation.detach().cpu().numpy()

        for seq, score, mean_score, std_score in zip(sequences, scores, mean, std_deviation):
            seq.objective_dict['surrogate_mean'] = mean_score
            seq.objective_dict['surrogate_uncertainty'] = std_score
            seq.objective_dict['ucb_surrogate'] = score

        return scores
    
    def calculate_alan_fitness(self, sequences, k):

        mean, std_deviation = self.forward_with_uncertainty(sequences)
        predictions = mean + (k * std_deviation)
        scores = predictions.detach().cpu().numpy()
        mean = mean.detach().cpu().numpy()
        std_deviation = std_deviation.detach().cpu().numpy()
        
        return scores, mean, std_deviation
