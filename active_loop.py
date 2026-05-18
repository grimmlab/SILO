import argparse, copy, os, time, ray, torch, datetime
from torch.optim.lr_scheduler import LambdaLR
from logger import Logger 
import numpy as np
from config import SequenceConfig
from objectives.utils import ESM3
from dataset import RegressionDataset
from model.transformer_architecture import SequenceTransformer, dict_to_cpu
from sequence_evaluator import SequenceEvaluator
from surrogate import ProxyModel
from utils import save_checkpoint, train_for_one_epoch_active_cycle, MetricsTracker, set_seed, str2bool


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--task",
                        type=str,
                        default = 'AAV',
                        choices=["AAV", "GFP", "TEM", "E4B", "UBE2I", "LGK", "Pab1", "AMIE"],
                        help="Specify benchmark task.")
    
    parser.add_argument("--seed",
                        type=int,
                        default=2,
                        help="Random seed")
    
    parser.add_argument("--device",
                        type=str,
                        default='cuda:0',
                        help="specify device name: either cuda:gpu_num (cuda:0) or cpu")
    
    parser.add_argument("--results",
                        type=str,
                        default="./results",
                        help="specify directory for storing results")
    
    parser.add_argument("--low_data_setting",
                        type=str2bool, default=False,
                        help="set True to run active learning to train proxy under low data mode")
    
    parser.add_argument("--low_data_perc",
                        type=float, default=1,
                        help="set to either 0.1, 0.2, 0.5 to use only 10%, 20%, or 50% of the available data for training proxy. " \
                        "Valid only when low_data_setting is True")
    
    parser.add_argument("--noise_mode",
                        type=str2bool, default=False,
                        help="set True to run the active learning under noisy proxy model")
    
    parser.add_argument("--noise_level", type=int, default=0,
                        help="set to either -5, -15, -25 to reproduce results from the paper. Valid only when noise_mode is True")
    
    args = parser.parse_args()
    return args


def main(args):

    os.environ["RAY_DEDUP_LOGS"]="0"
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]="1"


    print("------")
    if args.low_data_setting is not True:
        print(">> Protein Sequence Design using SILO")
    else:
        print(">> Protein Sequence Design using SILO under low data setting")

    config = SequenceConfig(args=args)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(","))
    ray.init(num_gpus=num_gpus, log_to_driver=False, logging_level="info") 
    print(ray.available_resources())

    config.results_path = os.path.join(f"{args.results}", f"{args.task}", f"{args.seed}")
    os.makedirs(config.results_path, exist_ok=True)

    logger = Logger(config, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)
    set_seed(config.seed)
    esm3_model = ESM3(config)
    sequence_evaluator = SequenceEvaluator(config)
    seen_protein_smiles: dict[str, float] = {} # to remove duplicates 

    # Load proxies models
    dataset = RegressionDataset(config)
    proxy = ProxyModel(config=config)
    metric_logger = MetricsTracker(config, dataset)

    # Setup the policy network for training
    network = SequenceTransformer(config, config.training_device)

    # Initalize checkpoint dict
    checkpoint = {
        "model_weights": None,
        "best_model_weights": None,
        "optimizer_state": None,
        "epochs_trained": 0,
        "validation_metric": float("-inf"),   # objective of the best sequence designed during validation.
        "best_validation_metric": float("-inf"),  # corresponding to best model weights
        "proxy_model_weights": None
    }

    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    network.eval()

    start_time = time.perf_counter()
    logger.log_metrics({"event": "training_started", "timestamp": datetime.datetime.now().isoformat()})

    print("------")
    print(f"Training surrogate")
    reference_seqs = dataset.train.tolist() + dataset.valid.tolist()
    reference_scores = dataset.train_scores.tolist() + dataset.valid_scores.tolist()
    for seq, score in zip(reference_seqs, reference_scores):
        seen_protein_smiles[seq] = score

    proxy.train(dataset=dataset, iteration= 0)
    checkpoint['proxy_model_weights'] = [model.net.state_dict() for model in proxy.proxy.models]
    save_checkpoint(checkpoint, "proxy_model.pt", config)
    print("Training finished")

    print("------")
    print("Setting up optimizer for policy.")
    optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"])

    print("Setting up LR scheduler for policy")
    _lambda = lambda epoch: config.optimizer["schedule"]["decay_factor"] ** (checkpoint["epochs_trained"] // config.optimizer["schedule"]["decay_lr_every_epochs"])
    scheduler = LambdaLR(optimizer, lr_lambda=_lambda)
    remaining_oracle_calls = config.self_improvement_learning['max_oracle_calls_per_round'] * config.active_learn_cycles
    oracle_evaluted_sequences = []

    print("------")
    print(f"Starting training for {config.active_learn_cycles} cycles.")

    for outer in range(config.active_learn_cycles):

        # -------------------
        # Training loop
        # -------------------
        
        if remaining_oracle_calls <= 0:
                break

        best_model_weights = checkpoint["best_model_weights"]  # can be None
        best_validation_metric = checkpoint["best_validation_metric"]
        
        top_candidates = []

        print("------")
        print(f"Round {outer + 1}.")
        print(f"Generating Mutant Sequences.")
        
        network_weights = copy.deepcopy(network.get_weights())
        generated_loggable_dict, top_k_trajs = train_for_one_epoch_active_cycle(epoch=outer, config=config, network=network, 
                        network_weights=network_weights, optimizer=optimizer, objective_evaluator=sequence_evaluator, 
                        best_objective=best_validation_metric, esm3_model=esm3_model, 
                        seen_protein_smiles=seen_protein_smiles, proxy=proxy, metrics_logger=metric_logger, logger=logger)
        # Save model
        checkpoint["model_weights"] = copy.deepcopy(network.get_weights())
        checkpoint["optimizer_state"] = copy.deepcopy(
            dict_to_cpu(optimizer.state_dict())
        )

        # measure by best objective found during sampling
        val_metric = generated_loggable_dict["best_gen_obj"]   
        checkpoint["validation_metric"] = val_metric
        save_checkpoint(checkpoint, "last_model.pt", config)

        if val_metric > best_validation_metric:
            print(">> Got new best model.")
            checkpoint["best_model_weights"] = copy.deepcopy(checkpoint["model_weights"])
            checkpoint["best_validation_metric"] = val_metric
            best_model_weights = checkpoint["best_model_weights"]
            best_validation_metric = val_metric
            save_checkpoint(checkpoint, "best_model.pt", config)
        
        top_candidates.extend(top_k_trajs)

        final_candidates = top_candidates
        oracle_evaluted_sequences.extend(final_candidates)
        remaining_oracle_calls = remaining_oracle_calls - len(final_candidates)

        # calculate core metrics         
        logger.text_artifact(dest_path=os.path.join(config.results_path, "train_evaluation_metrics.txt"), 
                    csv_file=os.path.join(config.results_path, f"{config.tasks_configs['task']}_train_results.csv"), 
                    metric_logger=metric_logger, epoch=outer, dataset=dataset, current_traj_epoch= oracle_evaluted_sequences)

        print("------")
        print(f"Retrain surrogate model.")

        # Append dataset and retrain the surrogate
        seqs = [traj['smiles'] for traj in final_candidates]
        scores = [traj['objective_dict']['tape'] for traj in final_candidates]
        dataset.add((seqs, scores))

        if outer + 1 < config.active_learn_cycles:
            proxy.train(dataset=dataset, iteration=outer+1)
            
        # Save proxy trained models 
        checkpoint['proxy_model_weights'] = [model.net.state_dict() for model in proxy.proxy.models]
        save_checkpoint(checkpoint, "proxy_model.pt", config)

    print("------")
    print('Training ended for policy')
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    logger.log_metrics({"event":"total_training_time_sec", "time_elapsed": elapsed})
    print("Finished. Shutting down ray.")
    ray.shutdown()

if __name__=='__main__':
    args = parse_args()
    main(args)



