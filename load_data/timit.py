import torch
from torch.utils.data import Dataset
import json
import torchaudio
import os
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import warnings
import random

class TIMITDataset(Dataset):
    """
    TIMIT dataset class that loads audio and associated metadata/transcriptions.
    
    Args:
        json_path (str): Path to the JSON file containing TIMIT data
        timit_root (str): Root directory containing TIMIT audio files
        sample_rate (int, optional): Target sample rate for audio. Defaults to 16000.
        normalize_audio (bool, optional): Whether to normalize audio. Defaults to True.
    
    Returns:
        Dict containing:
            - audio_tensor: torch.Tensor of shape (1, num_samples)
            - speaker_id: str, speaker identifier
            - metadata: dict containing speaker metadata
            - prompts: list of prompts used
            - responses: list of responses generated
            - filepath: str, path to audio file
            - phonemes: DataFrame with columns [start_sample, end_sample, phoneme]
            - words: DataFrame with columns [start_sample, end_sample, word]
            - text: str, complete transcription
    """
    def __init__(
        self,
        json_path: str,
        timit_root: str,
        sample_rate: int = 16000,
        normalize_audio: bool = True
    ):
        super().__init__()
        
        # Load the JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.timit_root = timit_root
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get sample data
        sample = self.data[idx]
        
        # Get file paths
        audio_path = os.path.join(self.timit_root, sample['audio_path'])
        
        # Load audio first
        audio, sr = torchaudio.load(audio_path)
        
        
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
        
        prompts = sample.get('prompts', [])
        answers = sample.get('responses', [])
        
        if prompts and answers and len(prompts) == len(answers):
            rand_idx = random.randint(0, len(prompts) - 1)
            prompt = prompts[rand_idx]
            answer = answers[rand_idx].replace("\n", " ").strip()  # Clean response
        else:
            prompt = None
            answer = None
        
        return {
            'audio_tensor': audio,
            'sid': sample['speaker']['id'],
            'prompt': prompt,
            'answer': answer,
            'filename': audio_path,
        }