import torch
from torch.utils.data import Dataset
import json
import torchaudio
import os
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import warnings
import random
from pathlib import Path
from collections import defaultdict




class TEARSDataset(Dataset):
    """
    TEARS dataset class that loads audio and associated metadata/responses.
    
    Args:
        json_path (str): Path to the JSON file containing TEARS data
        tears_root (str): Root directory containing TEARS audio files
        sample_rate (int, optional): Target sample rate for audio. Defaults to 16000.
        duration (float, optional): Target duration in seconds. Defaults to 3.0.
        normalize_audio (bool, optional): Whether to normalize audio. Defaults to True.
    
    Returns:
        Dict containing:
            - audio_tensor: torch.Tensor of shape (1, num_samples)
            - speaker_id: str, speaker identifier
            - metadata: dict containing speaker metadata
            - prompt: str, randomly selected prompt
            - response: str, corresponding response
            - filepath: str, path to audio file
    """
    def __init__(
        self,
        json_path: str,
        tears_root: str,
        sample_rate: int = 16000,
        duration: float = 3.0,
        normalize_audio: bool = True, 
        augment: bool = True
    ):
        super().__init__()
        
        # Load the JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.tears_root = Path(tears_root)
        self.sample_rate = sample_rate
        self.duration = duration
        self.normalize_audio = normalize_audio
        self.target_samples = int(duration * sample_rate)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)
    
    def augment_audio(self, waveform, sample_rate):
        # Randomly select augmentation methods
        augmentation_choices = ['time_stretch', 'pitch_shift', 'add_noise', 'spec_aug']
        random.shuffle(augmentation_choices)

        for aug in augmentation_choices[:random.randint(1, len(augmentation_choices))]:
            if aug == 'time_stretch':
                rate = random.uniform(0.8, 1.25)
                effect = [['speed', str(rate)], ['rate', str(16000)]]
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, 16000, effects=effect
                )

            elif aug == 'pitch_shift':
                n_steps = random.randint(-4, 4)
                effect = [['pitch', str(n)] for n in [n_steps*100 for n in [random.choice([-2, -1, 1, 2])]]]
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 16000, effect)

            elif aug == 'add_noise':
                noise = torch.randn_like(waveform) * random.uniform(0.001, 0.015)
                waveform = waveform + noise

            elif aug == 'frequency_mask':
                freq_mask = T.FrequencyMasking(freq_mask_param=random.randint(15, 30))
                waveform = freq_mask(waveform)

            elif aug == 'time_mask':
                time_mask = T.TimeMasking(time_mask_param=random.randint(20, 80))
                waveform = time_mask(waveform)

            elif aug == 'reverb':
                effect = [['reverb', '-w', str(random.randint(10, 50))]]
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 16000, effect)

            elif aug == 'pitch_shift':
                steps = random.randint(-2, 2)
                effect = [['pitch', str(steps * 100)], ['rate', '16000']]
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 16000, effect)

        return waveform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get sample data
        sample = self.data[idx]
        
        # Get file path
        audio_path = str(self.tears_root / sample['audio_path'])
        
        # Load and process audio
        try:
            audio, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            
            if self.augment:
                audio = self.augment_audio(audio, self.sample_rate)
            
            # Normalize if requested
            if self.normalize_audio:
                mean = torch.mean(audio)
                std = torch.std(audio)
                audio = (audio - mean) / (std + 1e-8)
            
            # Handle duration
            num_samples = audio.shape[1]
            
            if num_samples >= self.target_samples:
                # Randomly crop to target duration
                start_sample = random.randint(0, num_samples - self.target_samples)
                audio = audio[:, start_sample:start_sample + self.target_samples]
            else:
                # Pad if shorter than target duration
                pad_size = self.target_samples - num_samples
                audio = torch.nn.functional.pad(audio, (0, pad_size))
        
        except Exception as e:
            warnings.warn(f"Error loading audio file {audio_path}: {str(e)}")
            # Return zero tensor if audio loading fails
            audio = torch.zeros(1, self.target_samples)
        
        # Get prompt and response
        prompts = sample.get('prompts', [])
        responses = sample.get('responses', [])
        
        if prompts and responses and len(prompts) == len(responses):
            rand_idx = random.randint(0, len(prompts) - 1)
            prompt = prompts[rand_idx]
            response = responses[rand_idx].replace("\n", " ").strip()
        else:
            prompt = None
            response = None
        
        return {
            'audio_tensor': audio,
            'sid': sample['speaker']['id'],
            'metadata': sample['speaker'],
            'prompt': prompt,
            'answer': response,
            'filename': str(audio_path)
        }



    @staticmethod
    def redistribute_speakers(
        json_paths: Dict[str, str],
        split_ratios: Dict[str, float],
        seed: int = 42
    ) -> Dict[str, List[Dict]]:
        """
        Redistribute speakers across splits according to given ratios.
        
        Args:
            json_paths: Dict mapping split names to json file paths
            split_ratios: Dict mapping split names to desired ratios (should sum to 1)
            seed: Random seed for reproducibility
            
        Returns:
            Dict mapping split names to lists of samples
        """
        random.seed(seed)
        
        # Collect all samples and group by speaker
        speaker_samples = defaultdict(list)
        for split, path in json_paths.items():
            with open(path, 'r') as f:
                data = json.load(f)
                for sample in data:
                    speaker_samples[sample['speaker']['id']].append(sample)
        
        # Get list of all speakers
        all_speakers = list(speaker_samples.keys())
        random.shuffle(all_speakers)
        
        # Calculate number of speakers for each split
        total_speakers = len(all_speakers)
        split_speakers = {
            split: int(ratio * total_speakers)
            for split, ratio in split_ratios.items()
        }
        
        # Adjust for rounding errors
        remainder = total_speakers - sum(split_speakers.values())
        if remainder > 0:
            # Add remaining speakers to first split
            split_speakers[list(split_speakers.keys())[0]] += remainder
        
        # Distribute speakers to splits
        new_splits = defaultdict(list)
        current_idx = 0
        
        for split, num_speakers in split_speakers.items():
            split_speaker_ids = all_speakers[current_idx:current_idx + num_speakers]
            for speaker_id in split_speaker_ids:
                new_splits[split].extend(speaker_samples[speaker_id])
            current_idx += num_speakers
        
        return new_splits

    @staticmethod
    def save_splits(splits: Dict[str, List[Dict]], output_dir: str):
        """Save redistributed splits to JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, samples in splits.items():
            output_path = output_dir / f"tears_dataset_{split_name}_with_responses.json"
            with open(output_path, 'w') as f:
                json.dump(samples, f, indent=2)

