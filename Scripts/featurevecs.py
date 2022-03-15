from matplotlib.pyplot import axis
import pandas as pd
import os
import numpy as np
import random
from gensim.models.wrappers import FastText
import csv

def main():
    pranayfileloc = "logit_backups_bertweet/test_with_logits_bertweet.csv"
    fastText_model = FastText.load_fasttext_format('../result_tokenized/backgroundcorpus')
    pranaydf = pd.read_csv(pranayfileloc)
    pranaydf = pranaydf.drop_duplicates(subset=["aspect","sentence"])
    print(pranaydf)

    targetloc = os.path.join("TweetSAPreds", "targetSentResults_finalversion.csv")
    targetdf = pd.read_csv(targetloc)
    targetdf = targetdf[["search_query","dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert',]]
    print(targetdf, targetdf.columns)

    combinedf = pd.merge(left=pranaydf, right=targetdf, how="left", left_on="aspect", right_on="search_query")
    combinedf = combinedf.drop_duplicates(subset=["aspect","sentence"])
    combinedf =combinedf[["sentence","hero", "villain", "victim", "other", "label", "aspect","dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert',]]
    
    combinedf["dom_tweet_sent_siebert"] = combinedf["dom_tweet_sent_siebert"].fillna("Neutral")
    combinedf["dom_tweet_sent_siebert"] = combinedf["dom_tweet_sent_siebert"].replace({"Negative": 0, "Neutral": 1, "Positive": 2})
    combinedf = combinedf.fillna(0.33)
    fastTextvecs = []
    replaced = 0
    for word in combinedf["aspect"]:
        try:
            fastTextvecs.append(fastText_model.wv[word])
        except:
            fastTextvecs.append(fastText_model.wv['<UNK>'])
            replaced+=1
    print("USING UNK TOKENS FOR {} ASPECTS".format(replaced))
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

    outpath = os.path.join("ClassifierFiles", "ClassDF_Test.csv")
    combinedf.to_csv(outpath, index=False)
    with open('ClassDF_Test_FT.csv', 'w') as f:
    
        # using csv.writer method from CSV package
        write = csv.writer(f)  
        write.writerows(fastTextvecs)



if __name__ == "__main__":
    main()
    print("Complete")
