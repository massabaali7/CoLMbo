from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from preprocessing.ast_processor import ast
from load_data.data_collactor import DataCollatorWithPadding

def prepare_dataloader(dataset: Dataset, batch_size: int, valid_train_flag: str):
    if valid_train_flag == "train":
        data_collator = DataCollatorWithPadding(padding=True)
    elif valid_train_flag == "valid":
        data_collator = DataCollatorWithPadding(padding=True)
    elif valid_train_flag == "test":
        data_collator = DataCollatorWithPadding(padding=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset), 

        collate_fn=data_collator
    )

