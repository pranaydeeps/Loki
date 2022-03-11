import os
from tabnanny import check
import pandas as pd

trainpath = os.path.join("..", "annotations", "unseen_test.jsonl")
trainfile = pd.read_json(trainpath, lines=True)
print(trainfile)
print(trainfile.columns)
entcols = ["OCR", "entity_list","image"]

entities = []
for col in entcols:
    for row in trainfile[col]:
        if row != []:
            entities += row
entities = list(set(entities))
entitiesfile = "entities.txt"

print(trainfile[[ "OCR", "entity_list","image"]])


testoutputpath = "testFinal.csv"
trainfile.to_csv(testoutputpath, index=False)
print("Complete")
