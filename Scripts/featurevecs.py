from matplotlib.pyplot import axis
import pandas as pd
import os
import numpy as np
import random

def main():
    pranayfileloc = os.path.join("ClassifierFiles", "train_with_logits.csv")
    pranaydf = pd.read_csv(pranayfileloc)
    pranaydf = pranaydf.drop_duplicates(subset=["aspect","sentence"])
    print(pranaydf)

    targetloc = os.path.join("TweetSAPreds", "targetSentResults_siebert_num.csv")
    targetdf = pd.read_csv(targetloc)
    targetdf = targetdf[["entity","dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert',]]
    print(targetdf, targetdf.columns)

    combinedf = pd.merge(left=pranaydf, right=targetdf, how="left", left_on="aspect", right_on="entity")
    combinedf = combinedf.drop_duplicates(subset=["aspect","sentence"])
    combinedf =combinedf[["sentence","hero", "villain", "victim", "label", "aspect","dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert',]]
    
    combinedf["dom_tweet_sent_siebert"] = combinedf["dom_tweet_sent_siebert"].fillna("Neutral")
    combinedf["dom_tweet_sent_siebert"] = combinedf["dom_tweet_sent_siebert"].replace({"Negative": 0, "Neutral": 1, "Positive": 2})
    combinedf = combinedf.fillna(0.33)
    
    combinedf["label"] = combinedf["label"].replace({"villain": 0, "hero" : 1, "victim": 2, "other":3})


    """
    herosim = []
    victimsim = []
    villainsim = []
    for i in range(len(combinedf)):
        herosim.append(random.uniform(0,1))
        villainsim.append(random.uniform(0,1))
        victimsim.append(random.uniform(0,1))

    combinedf["herosim"] = herosim
    combinedf["villainsim"] = villainsim
    combinedf["victimsim"] = victimsim"""
    print(combinedf, combinedf.columns)

    outpath = os.path.join("ClassifierFiles", "ClassDF_Train.csv")
    combinedf.to_csv(outpath, index=False)




if __name__ == "__main__":
    main()
    print("Complete")
