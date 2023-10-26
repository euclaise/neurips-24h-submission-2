from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from datasets import load_from_disk, load_dataset, concatenate_datasets
import wandb
import torch
from torch import nn
from typing import Optional, Dict, Sequence
import random

model_name = "mistralai/Mistral-7B-v0.1"

#from patch3 import replace_stablelm_attn_with_flash_attn
#replace_stablelm_attn_with_flash_attn()

from SlimTrainer import *

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)

model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

ds = load_from_disk("ds")
minipile = load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=42).select(range(1500))

prefix_map = {
    "human": "[user]\n",
    "gpt": "[assistant]\n",
    "assistant": "[assistant]\n"
}

seq_len = 512

def minipile_map_fn(row):
    toks = tokenizer.encode(row['text'] + tokenizer.eos_token, add_special_tokens=False, truncation=False)
    input_ids = toks
    labels = toks

    input_ids = input_ids[:seq_len]
    labels = labels[:seq_len]

    return {'input_ids': torch.Tensor(input_ids).int(), 'labels': torch.Tensor(labels).int()}

def ds_map_fn(row):
    input_ids = []
    labels = []

    for msg in row['conversations']:
        if msg['from'] == 'gpt_rationale':
            # Mask prefix but not value
            toks = tokenizer.encode(prefix_map['gpt'], add_special_tokens=False, padding=False)
            input_ids = input_ids + toks
            labels = labels + [-100]*len(toks)

            if msg['value'] != '':
                toks = tokenizer.encode(msg['value'], add_special_tokens=False, padding=False)
                input_ids = input_ids + toks
                labels = labels + toks
        elif msg['from'] == 'gpt_transition':
            # Mask transition
            toks = tokenizer.encode(msg['value'], add_special_tokens=False, padding=False)
            input_ids = input_ids + toks
            labels = labels + [-100]*len(toks)
        elif msg['from'] == 'gpt_target':
            # Do not mask
            toks = tokenizer.encode(msg['value'], add_special_tokens=False, padding=False)
            input_ids = input_ids + toks
            labels = labels + toks
        else:
            toks = tokenizer.encode(prefix_map[msg['from']], add_special_tokens=False, padding=False)
            input_ids = input_ids + toks
            labels = labels + [-100]*len(toks)

            toks = tokenizer.encode(msg['value'], add_special_tokens=False, padding=False)
            input_ids = input_ids + toks
            labels = labels + ([-100]*len(toks) if msg['from'] == "human" else toks)

    input_ids = input_ids[:seq_len]
    labels = labels[:seq_len]

    return {'input_ids': torch.Tensor(input_ids).int(), 'labels': torch.Tensor(labels).int()}

minipile = minipile.map(minipile_map_fn, remove_columns=minipile.column_names)
ds = ds.map(ds_map_fn, remove_columns=ds.column_names)
ds = concatenate_datasets([ds, minipile]).shuffle(seed=32)
ds = ds.filter(lambda x: (torch.Tensor(x['labels']) != -100).sum().item() > 1)

class DataCollatorForSupervisedDataset:
    tokenizer: PreTrainedTokenizer
    seq_len: int

    def __init__(self, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels
        )

dc = DataCollatorForSupervisedDataset(tokenizer=tokenizer, seq_len=seq_len)

opt = Adalite(model=model)

batch_sz=1
epochs=3

scheduler = CosineDecayWithWarmup(optimizer=opt, total_steps=(epochs*len(ds))/batch_sz, warmup_steps=epochs*(len(ds) / batch_sz)*0.2, max_lr=3e-6) # 4e-6

trainer = SlimTrainer(
    model=model,
    optim=opt,
    scheduler=scheduler,
    epochs=epochs,
    data_collator=dc,
    batch_size=batch_sz,
    wandb_entity="jpritsk",
    wandb_project="Zen",
    wandb_name="Zen 7B Human (min)",
    train_data=ds,
    neft=True,
    mixce=True,
    mixce_ratio=0.5
)
trainer.train()

model.save_pretrained("Zen_7B_human_min")
tokenizer.save_pretrained("Zen_7B_human_min")


wandb.finish()
