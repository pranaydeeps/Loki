import os
from tabnanny import check
import pandas as pd

trainpath = os.path.join("..", "annotations", "train.jsonl")
trainfile = pd.read_json(trainpath, lines=True)
print(trainfile)
print(trainfile.columns)
entcols = ["hero", "villain", "victim", "other"]

entities = []
for col in entcols:
    for row in trainfile[col]:
        if row != []:
            entities += row
entities = list(set(entities))
entitiesfile = "entities.txt"

with open(entitiesfile, "w") as writer:
    for entity in entities:
        writer.write(f"{entity}\n")

entscol = []
for i, row in trainfile.iterrows():
    ents = []
    for col in entcols:
        ents += row[col]
    entscol.append(ents)
trainfile["allentities"] =entscol
print(trainfile[[ "hero", "villain", "victim", "other", "allentities"]])


testoutputpath = "textAllEntities.csv"
trainfile.to_csv(testoutputpath, index=False)
print("Complete")
