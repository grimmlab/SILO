import os, inspect, json, pickle, csv
import pandas as pd 
from typing import Optional


class Logger:
    def __init__(self, config, results_path: str, log_to_file: bool):
        self.results_path = results_path
        self.log_to_file = log_to_file
        self.config = config

        self.file_log_path = os.path.join(self.results_path, "log.txt")
        if self.log_to_file:
            os.makedirs(self.results_path, exist_ok=True)

    def log_hyperparams(self, config_object):
        attributes = inspect.getmembers(config_object, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
        attribute_dict = {}

        def add_to_attribute_dict(a):
            for key, value in a:
                key = key.replace("+", "_plus")
                key = key.replace("@", "_at")
                if isinstance(value, dict):
                    add_to_attribute_dict([(f"{key}.{k}", v) for k, v in value.items()])
                else:
                    if key not in ["devices_for_eval_workers"] and len(str(value)) <= 500:
                        attribute_dict[key] = value

        add_to_attribute_dict(attributes)

        if self.log_to_file:
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps({"hyperparameters": attribute_dict}))
                f.write("\n")

    def log_metrics(self, metrics: dict, step: Optional[int] = None, step_desc: Optional[str] = "epoch"):
        if self.log_to_file:
            if step is not None:
                metrics[step_desc] = step
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

    def text_artifact(self, dest_path: str, csv_file, metric_logger, epoch= None, dataset= None, current_traj_epoch= None, 
                      if_final: bool = None):
        if if_final: 
            df = pd.read_csv(csv_file)
            traj_dict = {row['identifier']: [row['sequence'], row['raw_score']] for _, row in df.iterrows()}
        else:
            if not isinstance(current_traj_epoch[0], dict):
                traj_dict = {}
                for trajs in current_traj_epoch:
                    traj_dict[trajs.identifier] = [''.join(trajs.seq_string), trajs.objective_dict['tape']]
            else:
                traj_dict = {}
                for trajs in current_traj_epoch:
                    traj_dict[trajs['identifier']] = [trajs['smiles'], trajs['objective_dict']['tape']]

        exp_metrics = metric_logger.calculate_top_metrics(epoch=epoch, trajectories=traj_dict)
        with open(dest_path, "a") as f:
            f.write(str(exp_metrics) + "\n")

    def test_pickle_save_file(self, trajectories, destination_path):
        merged_seqs = trajectories
        if destination_path is not None:
            if os.path.isfile(destination_path):
                with open(destination_path, "rb") as f:
                    existing_seqs = pickle.load(f)  # list of dicts
                temp_d = {x["smiles"]: x for x in existing_seqs + merged_seqs}
                merged_seqs = list(temp_d.values())
                merged_seqs = sorted(merged_seqs, key=lambda x: x["objective"], reverse=True)

            # Pickle the generated data again
            with open(destination_path, "wb") as f:
                pickle.dump(merged_seqs, f)

    def save_results_csv(self, config, trajectories, path_csv_file):
        trajectories = sorted(trajectories, key=lambda x: x['objective_dict']['tape'] , reverse=True)
        file_exists = os.path.isfile(path_csv_file)
        with open(path_csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["identifier", "history", "sequence", "objective", "oracle_score"])
            for seq in trajectories:
                writer.writerow([
                    seq['identifier'],
                    seq['action_seq'],
                    (''.join(seq['seq_string'])),
                    seq['objective'], 
                    seq['objective_dict']['tape']
                ])
