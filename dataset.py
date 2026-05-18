# This code is adapted from:
# https://github.com/szczurek-lab/ProSpero.git

import numpy as np
import os
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self):
        self.rng = np.random.RandomState(142857)
    
    def validation_set(self, ratio=None):
        raise NotImplementedError()

    def add(self, batch):
        raise NotImplementedError()
    
    def top_k(self, k):
        raise NotImplementedError()
    

class RegressionDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        task = config.tasks_configs['task']

        self._load_dataset(config.proxy_config['dataset_path'], task)
        self.train_added = len(self.train)
        self.val_added = len(self.valid)
        self.added_sequences = list()
        self.added_scores = list()

    def _load_dataset(self, data_dirpath, task):
        x_path = os.path.join(task, f"{task}-x-init.npy")
        y_path = os.path.join(task, f"{task}-y-init.npy")
        x = np.load(os.path.join(data_dirpath, x_path))
        y = np.load(os.path.join(data_dirpath, y_path)).reshape(-1)

        if self.config.low_data_setting == True: 
            x, y = self._subsample_dataset(x, y, self.config.low_data_perc)

        self.train, self.valid, self.train_scores, self.valid_scores  = train_test_split(x, y, test_size=0.1, random_state=self.rng)

    def validation_set(self):
        return self.valid, self.valid_scores

    def add(self, batch):
        samples, scores = batch
        self.added_sequences += samples
        self.added_scores += scores
        train_seq, val_seq, train, val = train_test_split(np.array(samples), scores, test_size=0.1, random_state=self.rng)
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)
        self.train = np.concatenate((self.train, train_seq), axis=0)
        self.valid = np.concatenate((self.valid, val_seq), axis=0)

    def _tostr(self, seqs):
        return ["".join([str(i) for i in x]) for x in seqs]

    def _top_k(self, data, k):
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = data[1][indices]
        topk_prots = np.array(data[0])[indices]
        return self._tostr(topk_prots), topk_scores

    def top_k(self, k):
        data = (np.concatenate((self.train, self.valid), axis=0), np.concatenate((self.train_scores, self.valid_scores), axis=0))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]), axis=0)
        data = (seqs, scores)
        return self._top_k(data, k)
    
    def _subsample_dataset(self, x, y, fraction):
        n = len(x)
        k = int(fraction * n)

        indices = np.arange(n)
        rng_subsample = np.random.RandomState(self.config.seed)
        rng_subsample.shuffle(indices)

        selected = indices[:k]

        x_sub = x[selected]
        y_sub = y[selected]

        return x_sub, y_sub


