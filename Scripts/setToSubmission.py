from re import A
import pandas as pd
import numpy as np
import os

def main():
    path = os.path.join("unseen_test.jsonl")
    df = pd.read_json(path, lines=True)

    predpath = os.path.join("ClassifierFiles", "Predictions_TestFinal_ensembleV4.csv")
    preddf = pd.read_csv(predpath)
    # .replace({"villain": 0, "hero" : 1, "victim": 2, "other":3})

    #df = df.iloc[:5]
    print(df)
    print(preddf, preddf.columns)

    herospertweet = []
    villainpertweet =[]
    victimspertweet = []
    otherspertweet = []
    for i, row in df.iterrows():
        entsdf = preddf[preddf["image"] == row["image"]]
        herolist = []
        villainlist= []
        victimlist = []
        otherlist = []
        for z, entity in entsdf.iterrows():
            if entity["pred"] == "villain" :
                villainlist.append(entity["aspect"])
            elif entity["pred"] == "hero":
                herolist.append(entity["aspect"])
            elif entity["pred"] == "victim":
                victimlist.append(entity["aspect"])
            else:
                otherlist.append(entity["aspect"])
        if herolist == []:
            herospertweet.append(np.empty(0).tolist())
        else:
            herospertweet.append(list(set(herolist)))
        if villainlist == []:
            villainpertweet.append(np.empty(0).tolist())
        else:
            villainpertweet.append(list(set(villainlist)))
        if victimlist == []:
            victimspertweet.append(np.empty(0).tolist())
        else:
            victimspertweet.append(list(set(victimlist)))
        if otherlist == []:
            otherspertweet.append(np.empty(0).tolist())
        else:
            otherspertweet.append(list(set(otherlist)))
    df["hero"] = herospertweet
    df["villain"] = villainpertweet
    df["other"] = otherspertweet
    df["victim"] = victimspertweet

    #df[["hero" ,"villain", "victim", "other"]] = df[["hero" ,"villain", "victim", "other"]].fillna(np.array([]))
    
    print(df, df.columns)
    df =df[['image', 'hero', 'villain', 'other', 'victim']]
    df.to_json("testoutput.jsonl",orient="records",lines=True)
    df.to_csv("testoutput.csv", index=False)
        
    
if __name__ == "__main__":
    main()
    print("Complete")
