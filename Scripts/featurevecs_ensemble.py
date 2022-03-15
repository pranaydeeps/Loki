from matplotlib.pyplot import axis
import pandas as pd
import os
import numpy as np
import random

def main():
    split = 'test'
    pranayfileloc = "{}ensemble_with_logits.csv".format(split)
    pranaydf = pd.read_csv(pranayfileloc)
    pranaydf = pranaydf.drop_duplicates(subset=["aspect","sentence"])
    print(pranaydf)

    pranayfileloc2 = "{}ensemble_with_logits2.csv".format(split)
    pranaydf2 = pd.read_csv(pranayfileloc2)
    pranaydf2 = pranaydf2.drop_duplicates(subset=["aspect","sentence"])
    print(pranaydf2)

    pranayfileloc3 = "{}ensemble_with_logits3.csv".format(split)
    pranaydf3 = pd.read_csv(pranayfileloc3)
    pranaydf3 = pranaydf3.drop_duplicates(subset=["aspect","sentence"])
    print(pranaydf3)


    targetloc = os.path.join("TweetSAPreds", "targetSentResults_finalversion.csv")
    targetdf = pd.read_csv(targetloc)
    targetdf = targetdf[["search_query","dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert',]]
    print(targetdf, targetdf.columns)

    combinedf = pd.merge(left=pranaydf, right=targetdf, how="left", left_on="aspect", right_on="search_query")
    combinedf = combinedf.drop_duplicates(subset=["aspect","sentence"])
    combinedf =combinedf[["sentence","hero", "villain", "victim", "other", "label", "aspect","dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert',]]
    
    combinedf["hero2"] = pranaydf2["hero"]
    combinedf["villain2"] = pranaydf2["villain"]
    combinedf["victim2"] = pranaydf2["victim"]
    combinedf["other2"] = pranaydf2["other"]

    combinedf["hero3"] = pranaydf3["hero"]
    combinedf["villain3"] = pranaydf3["villain"]
    combinedf["victim3"] = pranaydf3["victim"]
    combinedf["other3"] = pranaydf3["other"]
    
    combinedf["dom_tweet_sent_siebert"] = combinedf["dom_tweet_sent_siebert"].fillna("Neutral")
    combinedf["dom_tweet_sent_siebert"] = combinedf["dom_tweet_sent_siebert"].replace({"Negative": 0, "Neutral": 1, "Positive": 2})
    combinedf = combinedf.fillna(0.33)
    
    combinedf["label"] = combinedf["label"].replace({"villain": 0, "hero" : 1, "victim": 2, "other":3})

    combinedf["other"].loc[combinedf['other'] >=0.9] = 0.0
    combinedf["other2"].loc[combinedf['other2'] >=0.9] = 0.0
    combinedf["other3"].loc[combinedf['other3'] >=0.9] = 0.0



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

    outpath = os.path.join("ClassifierFiles", "ClassDF_Test.csv")
    combinedf.to_csv(outpath, index=False)




if __name__ == "__main__":
    main()
    print("Complete")
