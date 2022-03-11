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

tag2idx = {'hero' : 0, 'villain': 1, 'victim': 2, 'other': 3}
idx2tag = {0 : 'hero', 1: 'villain', 2: 'victim', 3: 'other'}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")

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
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


from datasets import load_dataset
data = load_dataset("csv", data_files = {"test": "final_testdata.csv"})
tokenized_data = data.map(preprocess_function, batched=True)

from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
model = AutoModelForMultipleChoice.from_pretrained("./")

model.eval()
test_output = trainer.predict(tokenized_data["test"])

test_csv = []
for i, point in enumerate(tokenized_data["test"]):
    hero, villain, victim, other = torch.softmax(torch.tensor(train_output[0][i]),dim=0).numpy()
    #print((hero,villain,victim,other))
    train_csv.append([point["sentence"], point["aspect"], point["image"], float(hero), float(villain), float(victim), float(other)])

with open('testfinal_with_logits.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(train_csv)

