#!/usr/bin/env python
# coding: utf-8

from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tqdm
import numpy as np

with open('final_data.csv') as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    final_data = [row for row in reader]

tags = []
for dat in final_data:
    tags.append(dat[2])
    
tags = list(set(tags))
",".join(tags)
tag2idx = {tag:idx for idx, tag in enumerate(tags)}
idx2tag = {idx:tag for idx, tag in enumerate(tags)}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

ending_names = ["is a hero", "is a villain", "is a victim"]

def preprocess_function(examples):
    first_sentences = [[context] * 3 for context in examples["sentence"]]
    question_headers = examples["aspect"]
    second_sentences = [
     [f"{header} {end}" for end in ending_names] for i, header in enumerate(question_headers)
    ]
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    print(second_sentences)
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}


from datasets import load_dataset
data = load_dataset("csv", data_files = {"train": "train_data.csv", "test": "test_data.csv"})

tokenized_data = data.map(preprocess_function,batched=True)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        labels = [tag2idx[k] for k in labels]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
     output_dir="./results",
     evaluation_strategy="epoch",
     learning_rate=5e-5,
     per_device_train_batch_size=16,
     per_device_eval_batch_size=16,
     num_train_epochs=3,
     weight_decay=0.01,
 )

trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_data["train"],
     eval_dataset=tokenized_data["test"],
     tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer)
 )

trainer.train()





