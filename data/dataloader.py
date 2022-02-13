import torch
import numpy as np

from config import *

class Dataloader:
    def __init__(self, data: list, batch_size: int, shuffle: bool = False) -> dict:
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def get_padded_sequences(self, lines: list):
        max_line_len = len(max(lines, key=len))
        
        return np.array([[BOS_IDX] + line + [EOS_IDX] + [PAD_IDX] * (max_line_len - len(line))
            for i, line in enumerate(lines)]
            )
        
    def collate_fn(self, batch: list) -> dict:
        """
        Return dict with english sentences and arabic sentences
        """
        src = []
        trg = []
        for elem in batch:
            src.append(self.data[elem][0])
            trg.append(self.data[elem][1])
        
        in_src = self.get_padded_sequences(src)  # padded inputs
        in_trg = self.get_padded_sequences(trg) 
        
        return {"src": torch.tensor(in_src, dtype=torch.long), "trg": torch.tensor(in_trg, dtype=torch.long)}

    def __iter__(self):
        num_batches =  int(np.ceil(len(self.data) / self.batch_size))
        if self.shuffle:
            perm = torch.randperm(len(self.data))
            for batch_start in range(num_batches):
              
                batch = perm[batch_start  * self.batch_size:(batch_start + 1) * self.batch_size]
                yield  self.collate_fn(batch)
        else:
            len_data = [i for i in range(len(self.data))]
            for batch_start in range(num_batches):
                batch = len_data[batch_start  * self.batch_size:(batch_start + 1) * self.batch_size]
                yield  self.collate_fn(batch)


    def __len__(self):
        """
        Return length of dataloader
        """
        return int(np.ceil(len(self.data) / self.batch_size))
        