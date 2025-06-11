import torch
import random
from torch.utils.data import Dataset, DataLoader

class CombinedDataset(Dataset):
    """
    A dataset that combines two datasets (TIMIT and EARS), selecting samples based on a probability.

    Args:
        dataset1 (Dataset): The first dataset (e.g., TIMITDataset).
        dataset2 (Dataset): The second dataset (e.g., EARS).
        switch_prob (float): Probability of picking from dataset1 (default: 0.5).
    """
    def __init__(self, dataset1, dataset2, switch_prob=0.5):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        self.switch_prob = switch_prob  # Probability of picking from dataset1

    def __len__(self):
        return max(self.len1, self.len2)  # Use the longer dataset length

    def __getitem__(self, idx):
        # Decide whether to sample from dataset1 or dataset2
        if random.random() < self.switch_prob:
            return self.dataset1[idx % self.len1]  # Sample from dataset1
        else:
            return self.dataset2[idx % self.len2]  # Sample from dataset2