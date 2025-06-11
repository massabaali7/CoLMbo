import numpy as np
from transformers import AutoTokenizer
import os
import torch
from collections import OrderedDict
import librosa
from importlib_resources import files
import yaml
import argparse
import torchaudio
import torchaudio.transforms as T
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from glob import glob

from mapper import get_sid_mapper, get_text_mapper
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer


class ExpWrapper():
    def __init__(self, config_wrapper, gpu_id):
        self.tok_len          = config_wrapper['tok_len']
        self.text_prefix_length    = config_wrapper['text_prefix_length']
        self.sid_prefix_length = config_wrapper['sid_prefix_length']
        self.norm_sid_emb     = config_wrapper['norm_sid_emb']
        self.gpu_id           = gpu_id
        self.gpt = GPT2LMHeadModel.from_pretrained(config_wrapper['text_decoder'])
        self.gpt = self.gpt.to(self.gpu_id)

        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        self.sid_mapper = get_sid_mapper(config_wrapper["map_type"],None, 
                                         config_wrapper["prefix_size"], self.gpt_embedding_size, 
                                         config_wrapper["sid_prefix_length"], config_wrapper["sid_prefix_length_clip"], 
                                         config_wrapper["num_layers"])
        
        self.sid_mapper = self.sid_mapper.to(self.gpu_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config_wrapper['text_decoder'])
        self.tokenizer.add_special_tokens({'pad_token': '!'})

    def init_mapper(self):
        self.sid_mapper = DDP(self.sid_mapper, device_ids=[self.gpu_id], find_unused_parameters=True)

    def freeze_llm(self):
        for param in self.sid_mapper.parameters():
            param.requires_grad = False
        for param in self.gpt.parameters():
            param.requires_grad = False

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))
    
    def load_model(self, st, model):
        try:
            model.load_state_dict(st)
        except:
            for key in list(st.keys()):
                if "module." in key:
                    st[key.replace("module.", "")] = st.pop(key)
            model.load_state_dict(st)
        return model

    def load_model(self, st, model):
        try:
            model.load_state_dict(st)
        except:
            for key in list(st.keys()):
                if "module." in key:
                    st[key.replace("module.", "")] = st.pop(key)
            model.load_state_dict(st)
        return model

    def load_sid_model(self, sid_model, snapshot_path, sid_ck_name):
        loc = f"cuda:{self.gpu_id}"
        sid_model_path = f"{snapshot_path}/{sid_ck_name}"
        snapshot = torch.load(sid_model_path, map_location=loc)
        sid_model = self.load_model(snapshot["sid_model"], sid_model)
        best_val_loss = snapshot["val_loss"]
        epochs_run = snapshot["epochs_run"]

    def load_mapper(self, snapshot_path, mapper_ck_name):
        loc = f"cuda:{self.gpu_id}"
        mapper_path = sorted(glob(f"{snapshot_path}/mapper_*.pt"))[-1]
        mapper_path = f"{snapshot_path}/{mapper_ck_name}"
        snapshot = torch.load(mapper_path, map_location=loc)

        self.sid_mapper = self.load_model(snapshot["sid_mapper"],self.sid_mapper)
        
        self.epochs_run = snapshot["epochs_run"]
        logging.info(f"Resuming training from mapper at Epoch {self.epochs_run}")

    def preprocess_prompt(self, texts): # true false
        r"""Load list of prompts and return tokenized text"""
        tokenized_texts = []
        for ttext in texts:
            tok = self.tokenizer.encode_plus(
                        text=ttext, add_special_tokens=True,
                        max_length=10, 
                        pad_to_max_length=True, return_tensors="pt", truncation=True)
            for key in tok.keys():
                tok[key] = tok[key].reshape(-1).to(self.gpu_id)
            tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)
    
    def preprocess_prompt_single(self, texts): # true false
        r"""Load list of prompts and return tokenized text"""
        tokenized_texts = []
        tok = self.tokenizer.encode_plus(
                    text=texts, add_special_tokens=True,
                    max_length=10, 
                    pad_to_max_length=True, return_tensors="pt", truncation=True)
        for key in tok.keys():
            tok[key] = tok[key].reshape(-1).to(self.gpu_id)
        tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)


    def preprocess_text(self, texts): # true false
        r"""Load list of prompts and return tokenized text"""
        tokenized_texts = []
        for ttext in texts:
            ttext = ttext + ' <|endoftext|>'
            tok = self.tokenizer.encode_plus(
                        text=ttext, add_special_tokens=True,
                        max_length=self.tok_len, 
                        pad_to_max_length=True, return_tensors="pt", truncation=True)
            for key in tok.keys():
                tok[key] = tok[key].reshape(-1).to(self.gpu_id)
            tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)

    def _get_text_embeddings(self, preprocessed_texts):
        r"""Load preprocessed prompts and return a prompt embeddings"""
        with torch.no_grad():
            texts_embed = self.gpt.transformer.wte(preprocessed_texts['input_ids'])
        return texts_embed

    def get_sid_prefix(self, sid_embeddings):
        r"""Produces audio embedding which is fed to LM"""
        if self.norm_sid_emb:
            sid_embeddings = sid_embeddings / sid_embeddings.norm(2, -1).reshape(-1,1)

        sids_prefix = self.sid_mapper(sid_embeddings).contiguous().view(-1, self.sid_prefix_length, self.gpt_embedding_size)
        return sids_prefix
    
    def get_prompt_prefix(self, texts):
        r"""Load list of text prompts and return prompt prefix and prompt embeddings"""
        preprocessed_texts = self.preprocess_prompt(texts)
        print(preprocessed_texts)
        texts_embed = self._get_text_embeddings(preprocessed_texts)
        return texts_embed, preprocessed_texts
    def get_prompt_prefix_single(self, texts):
        r"""Load list of text prompts and return prompt prefix and prompt embeddings"""
        preprocessed_texts = self.preprocess_prompt_single(texts)
        texts_embed = self._get_text_embeddings(preprocessed_texts)
        return texts_embed, preprocessed_texts

    def get_text_prefix(self, texts):
        r"""Load list of text prompts and return prompt prefix and prompt embeddings"""
        preprocessed_texts = self.preprocess_text(texts)
        texts_embed = self._get_text_embeddings(preprocessed_texts)
        return texts_embed, preprocessed_texts
    
    def generate_beam(self, beam_size: int = 1, sids_prefix=None, entry_length=80, temperature=1., stop_token: str = ' <|endoftext|>'):
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        device = next(self.gpt.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            generated = sids_prefix # sid embedding
            for i in range(entry_length):
                outputs = self.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                
                next_token_embed = self.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        ############ Added for attn plot ###########
        # token_list = []
        # text_list = []
        # for output, length in zip(output_list, seq_lengths):
        #     for item in output[:int(length)]:
        #         token_list.append(item)
        #         text_list.append(self.tokenizer.decode(item))
        ############ Added for attn plot ###########
        output_texts = [self.tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts
        # return output_texts, token_list, text_list
    

