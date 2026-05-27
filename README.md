# SILO: Self-Improvement Imitation with Biologically Guided Search for Protein Design Under Oracle Budgets

This is the repository for the following paper: [Self-Improvement Imitation with Biologically Guided Search for Protein Design Under Oracle Budgets](https://arxiv.org/abs/2605.26690) (arXiv:2605.26690).

## Configuration
Configurations of our system:
- GPU: NVIDIA A40
- CUDA: 12.2
- PyTorch: 2.8.0+cu128

Note:
Running on H100 GPUs may require newer CUDA/PyTorch builds and updated NVIDIA libraries 

Run protein sequence optimization using our framework SILO.

## Installation
1. Please make sure you have miniconda installed. Create a conda environment by running the command:

```
conda create -n silo python=3.10
conda activate silo
pip install -r requirements.txt
```

2. Apart from this, please download the oracle models using bash scripts. Please ensure that these are downloaded under ./objectives. 

```
chmod +x bash/*
./bash/download_oracles.sh
```

## Run experiments 
### Protein sequence optimization 
To reproduce the main experiment on 8 diverse protein design tasks, please run the following bash script. This will run protein optimization pipeline for 10 active learning rounds with 128 candidate sequences evaluated per round from the oracle. 

```
chmod +x bash/*
./bash/run_all_landscapes.sh
```

To run SILO on a single task, please run the command with following command-line parameters available :

```
python active_loop.py --task task_name
```

| Argument             | Type  | Description                                                                 |
|----------------------|-------|-----------------------------------------------------------------------------|
| `--task`             | str   | Name of the 8 available protein fitness benchmark tasks. Defaults to AMIE. |
| `--seed`             | int   | Random seed for reproducibility. Defaults to 2.                            |
| `--device`           | str   | Device to run/train the policy (e.g., `cuda:0`). Defaults to `cuda:0`.     |
| `--results`          | str   | Path to the results directory. Defaults to `./results`.                    |
| `--low_data_setting` | bool  | Enable low-data regime for SILO. Defaults to `False`.                      |
| `--low_data_perc`    | float | Fraction of initial data used for proxy training. Defaults to `1.0`.       |
| `--noise_mode`       | bool  | Enable noisy proxy setup for SILO. Defaults to `False`.                    |
| `--noise_level`      | int   | Noise level added to proxy predictions. Defaults to `0`.                   |

### Noisy setting  
To run SILO under a noisy surrogate setting for AMIE and E4B under noise levels -5, -10, -25, please run the following command:

```
chmod +x bash/*
./bash/run_noisy_setting.sh
```

### Low data regime  
To run SILO under under low data regime setting (proxies trained only on a certain fraction of the available initial data) for UBE2I and TEM, please run the following command:

```
chmod +x bash/*
./bash/run_low_data_setting.sh
```

### Config

The following section specify the most important config parameter in `config.py`.

#### Network architecture options

- `latent_dimension`: [int] Latent dimension $`d`$. Defaults to 512.
- `num_transformer_blocks`: [int] Number of layers in the stack of transformer blocks for the architecture. Defaults to 2.
- `num_heads`: [int] Number of heads in the multihead attention. Defaults to 16.
- `dropout`: [float] Dropout for feedforward layer in a transformer block. Defaults to 0.

#### Environment options
- `residue_vocabulary`: [dict] Specifies the vocabulary that is allowed in the design process. 
- `max_limit_pos_change`: [int] The number of time the agent can change a selected position. While this can be more than 1, it makes the sampling process longer. Defaults to 1. 
- `oracle_path` = Path to store all the oracle models. Should always be under ./objectives. 
- `active_learn_cycles`: [int] Number of active learning rounds. Defaults to 10. 
- `multiplier`: [int] Number of multiple problem instances to be created for sampling. Defaults to 5. 
- `min_max_mutations`: [int, int] Minimum and maximum number of mutations to be done within a given sequence. 
- `num_dataloader_workers`: [int] Number of workers for preparing batches for supervised training. Defaults to 3.
- `CUDA_VISIBLE_DEVICES`: [str] ray.io sometimes needs help to recognize multiple GPUs. 
This variable will be set as the env var of the same name in each worker. Defaults to "0,1" (for two GPUs). Modify as needed.
- `batch_size_training`: [int] Batch size to use for the supervised training during finetuning. Defaults to 32.
- `num_batches_per_epoch`: [int] Number of batches to use for supervised training during finetuning in each epoch. Can be `None`, then one pass through the current generated dataset is done.

#### Learning algorithm for finetuning options
- `self_improvement_learning`: [dict] These are the configs for the self-improvement learning part. We discuss the most important ones here. For more details, please refer to https://github.com/grimmlab/gumbeldore
    - `num_trajectories_to_keep:` [int] Number of best designed sequences to keep, which are used for supervised training during finetuning. Default is 50.
    - `keep_intermediate_trajectories`: [bool] If this is `True`, all designed sequences encountered in the trie are considered, not only the leaves.
    - `devices_for_workers`: List[str] Number of parallel workers and on which devices their models live. Defaults to `["cuda:0"] * 1`.
    - `batch_size_per_worker`: [int] If you start from a single problem instance, keep at 1. Defaults to 3
    - `batch_size_per_cpu_worker`: [int] Same as above. This value is used for workers whose models live on the CPU. Default to 3.
    - `search_type`: [str] Select `'wor'` for using stochastic beam search or `'beam_search'` for determinstic beam search
    - `beam_width`: [int] Beam width for stochastic beam search. Defaults to 32. 
    - `num_rounds`: Union[int, Tuple[int, int]]. Only relevant for `search_type='wor'`. If it's a single integer, we sample for this many rounds exactly. If it's an (int, int)-tuple, then we sample as long as it takes to obtain a new best sequence, but for a minimum of first entry rounds and a maximum of second entry rounds. Defaults to 1. 
    - `deterministic`: [bool] Set to `True` to switch to deterministic beam seach.
    - `nucleus_top_p`: [float] Top-p sampling nucleus size. Defaults to 1.0 (no nucleus sampling)
    - `pin_workers_to_core`: [bool] Default to `False`. If `True`, workers are pinned to single CPU threads, which can help with many workers on the CPU to prevent them from jamming each other with their numpy and pytorch operations.

## Acknowledgments

This project builds from several open-source repositories:
- [Gumbeldore](https://github.com/grimmlab/gumbeldore) — for providing implementations related to incremental stochastic beam search, dataset sampling, and candidate generation.
- [ProSpero](https://github.com/szczurek-lab/ProSpero.git) — for providing benchmark datasets and oracle models used in protein fitness optimization experiments.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention.git) — for efficient attention implementations used during model training and inference.
- [Stochastic Beam Search](https://github.com/wouterkool/stochastic-beam-search/tree/stochastic-beam-search) — for reference implementations and methodology related to stochastic beam search.
