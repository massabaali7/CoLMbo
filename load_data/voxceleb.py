import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import random
import os

class ZeroShotDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied to audio.
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.sample_rate = 16000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        root = "/ocean/projects/cis220031p/psamal/preprocess_TIMIT/"
        
        # Load audio file
        audio, sr = torchaudio.load(os.path.join(root, row["File_Path"]))
        
        # Apply transformation if provided
        if self.transform:
            audio = self.transform(audio)


        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            
        mean = torch.mean(audio)
        std = torch.std(audio)
        audio = (audio - mean) / (std + 1e-8)
            
        # Get total number of samples
        num_samples = audio.shape[1]
        num_samples_3s = 3 * self.sample_rate  # Samples for 3 seconds
        
        # Ensure the audio is at least 3 seconds long
        if num_samples >= num_samples_3s:
            start_sample = random.randint(0, num_samples - num_samples_3s)
            end_sample = start_sample + num_samples_3s
            audio = audio[:, start_sample:end_sample]
        else:
            # If audio is shorter than 3 seconds, pad it
            pad_size = num_samples_3s - num_samples
            audio = torch.nn.functional.pad(audio, (0, pad_size))
        
        return {
            "sid": "WBT0",
            "audio_tensor": audio,
            "answer": row["Ground_Truth"],
            "prompt": row["Prompt"],
            # "prompt": random.choice(["What is the dialect of the person?", "Based on the voice of the person, please specify the dialect of the person?", row["Prompt"]]),
            'filename': row["File_Path"],
        }
