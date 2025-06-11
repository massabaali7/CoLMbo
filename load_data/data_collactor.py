import torch
from transformers import AutoFeatureExtractor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from preprocessing.ast_processor import ast
from util_stats.local_stats import local_extract_phn_frame_probs
from util_stats.global_stats import global_extract_phn_frame_probs
import numpy as np
import pickle
import torch.nn.functional as F

from load_data.extract_fbanks import Mel_Spectrogram

extractor = Mel_Spectrogram()

with open('new_lbl2ind.pkl', 'rb') as f:
    lbl2ind = pickle.load(f) 
with open('new_spk.pkl', 'rb') as f:
    unique_speaker_ids = pickle.load(f) 
# change the labels
number_Of_spks = len(unique_speaker_ids)


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """



    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    flag_global_local: Optional[str] = None
    dic_train_phn_frequency: Optional [dict] = None
    dic_train_frame_frequency: Optional [dict] = None
    lbl2ind: Optional [dict] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        batch={}
        batch['input_values']= [features[idx]['audio_tensor'].squeeze(0) for idx in range(len(features))]
        batch["prompt"] = [features[idx]["prompt"] for idx in range(len(features))]
        batch["answer"] = [features[idx]["answer"] for idx in range(len(features))]
        batch["filename"] = [features[idx]["filename"] for idx in range(len(features))]
        # batch["no_hot_encode"] = torch.tensor([lbl2ind[features[idx]['sid']] for idx in range(len(features))])
        batch["no_hot_encode"] = torch.tensor([0 for idx in range(len(features))])
        # if batch["no_hot_encode"].numel():
        batch["labels"]= F.one_hot(batch["no_hot_encode"], number_Of_spks) 
        batch['input_values'] = extractor(torch.stack(batch['input_values']))
        return batch
