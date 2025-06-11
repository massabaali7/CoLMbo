import os
from glob import glob
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import pickle
from copy import deepcopy
from glob import glob
import random
from sklearn.model_selection import train_test_split
import json
import os
import numpy as np 
import librosa
import torch
import soundfile as sf
import pandas as pd
import random

class EARS(Dataset):
    """
    EARS dataset for 10sec or less that 10sec segments.
    Returns:
        audio: torch.Tensor in (1,16000) or (1, <16000), audio waveform
        sid: str (p103), speaker id
        metadict: dict, metadata
        caption: str, caption
        alignment: list
    """
    def __init__(self, root, data_path, meta_path,utterance_path, prompts_path, sample_rate, train_mapper=False, split="train"):
        super().__init__()
        self.root = root

        with open(f"{data_path}", "r") as f:
            self.data = json.load(f)

        with open(f"{meta_path}", "r") as f:
            self.meta = json.load(f)
        
        with open(f"{utterance_path}", "r") as f:
            self.utterance = json.load(f)

        with open(f"{prompts_path}", "r") as f:
            self.prompts = json.load(f)

        self.new_data = []
        if train_mapper:
            for d in self.data:
                file_name = d["filename"]
                sid = file_name.split("/")[0]
                temp = random.sample(self.prompts[sid], 10)
                for qa in temp:
                    self.new_data.append({"filename": file_name, 
                                        "start": d["start"], 
                                        "end": d["end"], 
                                        "prompt": qa[0], 
                                        "answer": qa[1]})
        else:
            self.new_data = self.data
        if split == "train":
            random.shuffle(self.new_data)

        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.new_data)

    def __getitem__(self, idx):
        entry = self.new_data[idx]
        filename = entry["filename"]
        sid      = filename.split("/")[0]
        audio_path = os.path.join(self.root, filename)

        # Load audio
        audio, sample_rate = torchaudio.load(audio_path)
        start_sample, end_sample = entry["start"], entry["end"]
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(audio)

        # Compute duration in samples
        total_samples = end_sample - start_sample
        num_samples_3s = 3 * self.sample_rate  # 3 seconds worth of samples
        
        # Select a random 3s window within the available range
        if total_samples >= num_samples_3s:
            start_offset = random.randint(start_sample, end_sample - num_samples_3s)
            end_offset = start_offset + num_samples_3s
            audio = audio[:, start_offset:end_offset]
        else:
            # If less than 3s, take full segment and pad
            pad_size = num_samples_3s - total_samples
            audio = audio[:, start_sample:end_sample]
            audio = torch.nn.functional.pad(audio, (0, pad_size))

        # Normalize
        mean = torch.mean(audio)
        std = torch.std(audio)
        audio = (audio - mean) / (std + 1e-8)

        return {
            "audio_tensor": audio,
            "filename": filename,
            "sid": sid,
            "prompt": entry.get("prompt", None),
            "answer": entry.get("answer", None),
        }