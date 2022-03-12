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
import csv
from torch import nn

with open('final_data.csv') as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    final_data = [row for row in reader]

tags = []
for dat in final_data:
    tags.append(dat[2])
tags = list(set(tags))
tag2idx = {'hero' : 0, 'villain': 1, 'victim': 2, 'other': 3}
idx2tag = {0 : 'hero', 1: 'villain', 2: 'victim', 3: 'other'}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

ending_names = ["is a hero", "is a villain", "is a victim", "is neutral"]

def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["sentence"]]
    question_headers = examples["aspect"]
    second_sentences = [
     [f"{header} {end}" for end in ending_names] for i, header in enumerate(question_headers)
    ]
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    #print(second_sentences)
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=512)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


from datasets import load_dataset
import datasets
datasets.set_caching_enabled(False)
data = load_dataset("csv", data_files = {"train": "train_data.csv", "test": "test_data.csv"})

tokenized_data = data.map(preprocess_function, batched=True)

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
            return_tensors="pt"
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
model = AutoModelForMultipleChoice.from_pretrained("vinai/bertweet-large")

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils import class_weight

class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                         classes=[idx2tag[i]for i in range(0,len(tags))],
                                                         y=list(tokenized_data["train"]["label"]))))
print('WEIGHTS: {}'.format(str(class_weights)))
weights_list = [float(class_weights[i])for i in range(0,len(tags))]


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights_list).cuda())
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }





training_args = TrainingArguments(
     output_dir="./bertweet-covid19/",
     evaluation_strategy="steps",
     eval_steps=1000,
     learning_rate=1e-5,
     per_device_train_batch_size=1,
     per_device_eval_batch_size=2,
     num_train_epochs=15,
     gradient_accumulation_steps=8,
     weight_decay=0.01,
     save_total_limit=3,
     metric_for_best_model='f1',
     logging_steps=200)



trainer = CustomTrainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_data["train"],
     eval_dataset=tokenized_data["test"],
     tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

model.eval()
train_output = trainer.predict(tokenized_data["train"])
test_output = trainer.predict(tokenized_data["test"])

train_csv = []
for i, point in enumerate(tokenized_data["train"]):
    hero, villain, victim, other = torch.softmax(torch.tensor(train_output[0][i]),dim=0).numpy()
    #print((hero,villain,victim,other))
    train_csv.append([point["sentence"], point["aspect"], point["label"], float(hero), float(villain), float(victim), float(other)])

with open('train_with_logits_bertweet_covid19.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(train_csv)

test_csv = []
for i, point in enumerate(tokenized_data["test"]):
    hero, villain, victim, other = torch.softmax(torch.tensor(test_output[0][i]),dim=0).numpy()
    #print((hero,villain,victim, other))
    test_csv.append([point["sentence"], point["aspect"], point["label"], float(hero), float(villain), float(victim), float(other)])

with open('test_with_logits_bertweet_covid19.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(test_csv)


