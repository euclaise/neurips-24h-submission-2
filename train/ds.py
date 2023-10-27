from datasets import load_dataset, Dataset, concatenate_datasets
import random
import re
import pandas

mqa = load_dataset("euclaise/mqa", split='train').shuffle(seed=42).select(range(20000))
st = load_dataset("euclaise/symtune_mini", split='train').shuffle(seed=42).select(range(250))
samsum = load_dataset("samsum", split='train').shuffle(seed=42).select(range(250))


def map_mqa(row):
    return {
        'system': "",
        'conversations': [
            {"from": "human", "value": row['msg']},
            {"from": "gpt", "value": row['resp_correct']}
        ]
    }
mqa = mqa.map(map_mqa, num_proc=8, remove_columns=mqa.column_names)

prefix_samsum = [
    "Summarize the following dialogue:\n",
    "Summarize this exchange:\n",
    "Provide a summary of the dialogue below:\n",
]
def map_samsum(row):
    prefix = random.choice(prefix_samsum)
    return {
        'system': "",
        'conversations': [
            {"from": "human", "value": prefix + row['dialogue']},
            {"from": "gpt", "value": row['summary']}
        ]
    }
samsum = samsum.map(map_samsum, num_proc=8, remove_columns=samsum.column_names)

def map_st(row):
    return {
        'system': "",
        'conversations': [
            {"from": "human", "value": row['question']},
            {"from": "gpt", "value": row['answer']}
        ]
    }
st = st.map(map_st, num_proc=8, remove_columns=st.column_names)




ds = concatenate_datasets([mqa, samsum, st]).shuffle(seed=42)


print(ds)
ds.save_to_disk("ds")
