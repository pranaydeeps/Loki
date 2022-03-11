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
import os

tag2idx = {'hero' : 0, 'villain': 1, 'victim': 2, 'other': 3}
idx2tag = {0 : 'hero', 1: 'villain', 2: 'victim', 3: 'other'}

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("./checkpoint-5000")

# ending_names = ["is a hero", "is a villain", "is a victim", "is neutral"]

# def preprocess_function(examples):
#     first_sentences = [[context] * 4 for context in examples["sentence"]]
#     question_headers = examples["aspect"]
#     second_sentences = [
#      [f"{header} {end}" for end in ending_names] for i, header in enumerate(question_headers)
#     ]
#     first_sentences = sum(first_sentences, [])
#     second_sentences = sum(second_sentences, [])
#     #print(second_sentences)
#     tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
#     return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


# from datasets import load_dataset
# data = load_dataset("csv", data_files = {"test": "final_testdata.csv"})
# tokenized_data = data.map(preprocess_function, batched=True)

# from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
# model = AutoModelForMultipleChoice.from_pretrained("./checkpoint-5000")


# from dataclasses import dataclass
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
# from typing import Optional, Union
# import torch


# @dataclass
# class DataCollatorForMultipleChoice:
#     """
#     Data collator that will dynamically pad the inputs for multiple choice received.
#     """

#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None

#     def __call__(self, features):
#         batch_size = len(features)
#         num_choices = len(features[0]["input_ids"])
#         flattened_features = [
#             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
#         ]
#         flattened_features = sum(flattened_features, [])

#         batch = self.tokenizer.pad(
#             flattened_features,
#             padding=True,
#             max_length=128,
#             return_tensors="pt"
#         )

#         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
#         return batch



# training_args = TrainingArguments(
#      output_dir=".",
#      evaluation_strategy="epoch",
#      learning_rate=5e-5,
#      per_device_train_batch_size=2,
#      per_device_eval_batch_size=2,
#      num_train_epochs=5,
#      gradient_accumulation_steps=4,
#      weight_decay=0.01,
#      save_strategy='no')


# trainer = Trainer(
#      args=training_args,
#      model=model,
#      train_dataset=tokenized_data["test"],
#      eval_dataset=tokenized_data["test"],
#      tokenizer=tokenizer,
#      data_collator=DataCollatorForMultipleChoice(tokenizer)
# )

# model.eval()
# test_output = trainer.predict(tokenized_data["test"])

# test_csv = []
# for i, point in enumerate(tokenized_data["test"]):
#     hero, villain, victim, other = torch.softmax(torch.tensor(test_output[0][i]),dim=0).numpy()
#     #print((hero,villain,victim,other))
#     test_csv.append([point["sentence"], point["aspect"], point["image"], float(hero), float(villain), float(victim), float(other)])

# with open('testfinal_with_logits.csv', 'w') as f:
#     write = csv.writer(f)
#     write.writerows(test_csv)

# pranayfileloc = "testfinal_with_logits.csv"
# pranaydf = pd.read_csv(pranayfileloc)
# pranaydf = pranaydf.drop_duplicates(subset=["aspect","sentence"])
# print(pranaydf)

# targetloc = os.path.join("TweetSAPreds", "targetSentResults_siebert_num.csv")
# targetdf = pd.read_csv(targetloc)
# targetdf = targetdf[["entity","dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert',]]
# print(targetdf, targetdf.columns)

# combinedf = pd.merge(left=pranaydf, right=targetdf, how="left", left_on="aspect", right_on="entity")
# combinedf = combinedf.drop_duplicates(subset=["aspect","sentence"])
# combinedf =combinedf[["image", "sentence","hero", "villain", "victim", "other", "aspect","dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert',]]

# combinedf["dom_tweet_sent_siebert"] = combinedf["dom_tweet_sent_siebert"].fillna("Neutral")
# combinedf["dom_tweet_sent_siebert"] = combinedf["dom_tweet_sent_siebert"].replace({"Negative": 0, "Neutral": 1, "Positive": 2})
# combinedf = combinedf.fillna(0.33)

# # combinedf["label"] = combinedf["label"].replace({"villain": 0, "hero" : 1, "victim": 2, "other":3})

# print(combinedf, combinedf.columns)

# outpath = os.path.join("ClassifierFiles", "ClassDF_TestFinal.csv")
# combinedf.to_csv(outpath, index=False)


from unicodedata import category
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import classification_report
# from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

test_features = os.path.join("ClassifierFiles", "ClassDF_TestFinal.csv")
test_df = pd.read_csv(test_features)
test_features = test_df[[ "hero", "villain", "victim", "other", "dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert']]
test_features["dom_tweet_sent_siebert"] = test_features["dom_tweet_sent_siebert"].astype("category")
    
loaded_model = pickle.load(open('models/modelSVM.pkl', 'rb'))
print("Model Loaded")
print("Done")

results = loaded_model.predict(X=test_features)
results = results.tolist()


# print(classification_report(y_true=true_labels, y_pred= results))
test_df["pred"] = results
test_df["pred"] = test_df["pred"].replace({0: "villain", 1 : "hero", 2 : "victim", 3 : "other"})
test_df.to_csv(os.path.join("ClassifierFiles", "Predictions_TestFinal.csv"), index=False)