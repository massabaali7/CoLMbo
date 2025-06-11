import torch
import os 
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
from encoder.mha import MultiHeadAttention
from encoder.self_attn import TransformerSelfAttention
from preprocessing.ast_processor import ast
from load_data.prepare_dataloader import prepare_dataloader
from encoder.encoder import Model
from load_data.extract_fbanks import Mel_Spectrogram

from datasets import Dataset, DatasetDict, Features, ClassLabel, Value, concatenate_datasets
from torch.utils.data import Subset
from torch.utils.data import Subset
from loss.cross_entropy import cross_entropy_loss
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, SequentialLR, LambdaLR, CosineAnnealingWarmRestarts
import numpy as np 
from transformers import AutoFeatureExtractor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import math 
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import pandas as pd
import pickle 
import wandb
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pickle
# import config
import logging
import argparse
import yaml
from glob import glob
import random
from wrapper import ExpWrapper
import torchaudio

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    # load config
    config = load_config(config_path)
    config_train = config['train']
    config_sid_model = config['sid_model']
    config_data = config['data']
    config_wrapper = config['wrapper']

    train_mapper = config["train_mapper"]
    snapshot_path = config_train["snapshot_path"]
    gpu_id = config_wrapper['gpu_id']
    device = config_wrapper['device']
    # initialize mapper model
    extractor = Mel_Spectrogram()
    Exp = ExpWrapper(config_wrapper, gpu_id)
    get_text_prefix = Exp.get_text_prefix
    get_sid_prefix  = Exp.get_sid_prefix
    get_prompt_prefix = Exp.get_prompt_prefix_single
    generate_beam = Exp.generate_beam
    save_mapper = Exp.save_mapper
    load_mapper = Exp.load_mapper

    sid_model = Model(n_mels=80, embedding_dim=192, channel=1024)
    # Load the pretrained model checkpoint
    checkpoint = torch.load("./pretrained_sid/ecapa.ckpt")
    
    new_state_dict = {f"model.{k}": v for k, v in checkpoint.items()}
    
    # Assuming the checkpoint contains the state dict directly
    sid_model.load_state_dict(new_state_dict)
    
    Exp.load_sid_model(sid_model, snapshot_path, config_sid_model["sid_ck_name"]) 
    Exp.load_mapper(snapshot_path, config_wrapper["mapper_ck_name"])
        
    sid_model   = sid_model.to(gpu_id)
    sid_mapper  = Exp.sid_mapper.to(gpu_id)
    # load data
    sid_model.eval()
    Exp.gpt.eval()
    Exp.sid_mapper.eval()
    waveform_audio, sr = torchaudio.load(config_data['waveform'])
    # Use for Evaluating
    processed_waveform = extractor(waveform_audio).to(device) 
    sid_emb = sid_model(processed_waveform)
    sids_prefix = get_sid_prefix(sid_emb)
    prompt_prefix, prefix_label = get_prompt_prefix(config_data['prompt'])
    prefix_emb = torch.cat((sids_prefix, prompt_prefix), dim=1)
    generated_texts = generate_beam(sids_prefix=prefix_emb)
    print(generated_texts[0])
# load model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with specified configuration")
    parser.add_argument('--config', type=str, required=False, default="config.yaml" , help='Path to the config.yaml file')
    args = parser.parse_args()
    main(args.config)

