
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
from xgboost import XGBClassifier
from tpot import TPOTClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive




def main():
    test_features = os.path.join("ClassifierFiles", "ClassDF_Test.csv")
    test_df = pd.read_csv(test_features)
    test_features = test_df[["hero", "villain", "victim", "other", "hero2", "villain2", "victim2", "other2", "hero3", "villain3", "victim3", "other3", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert']]
    #test_features["dom_tweet_sent_siebert"] = test_features["dom_tweet_sent_siebert"].astype("int")
    true_labels = test_df["label"]

    featurepath = os.path.join("ClassifierFiles", "ClassDF_Train.csv")
    featuresdf  = pd.read_csv(featurepath)
    features = featuresdf[["hero", "villain", "victim", "other", "hero2", "villain2", "victim2", "other2", "hero3", "villain3", "victim3", "other3",  'Negative_siebert', 'Neutral_siebert', 'Positive_siebert']] 
    #features["dom_tweet_sent_siebert"] = features["dom_tweet_sent_siebert"].astype("int")
    labels = featuresdf["label"]

    # tpot = TPOTClassifier(generations=5, population_size=50, verbosity=3, random_state=42, scoring='f1_macro')
    tpot = make_pipeline(
    SelectFwe(score_func=f_classif, alpha=0.014),
    StackingEstimator(estimator=BernoulliNB(alpha=100.0, fit_prior=False)),
    StackingEstimator(estimator=GaussianNB()),
    LogisticRegression(C=5.0, dual=False, penalty="l2")
    )
# Fix random state for all the steps in exported pipeline
    set_param_recursive(tpot.steps, 'random_state', 42)
    tpot.fit(features, labels)
    # print(tpot.score(test_features, true_labels))
    # tpot.export('tpot_pipeline.py')
        
    #print(model.best_params_)
    path = os.path.join("models","modelTPOT.pkl")
    with open(path, 'wb') as file:
        pickle.dump(tpot, file)
    print("Model saved")
    print("Done")
    
    results = tpot.predict(X=test_features)
    results = results.tolist()


    print(classification_report(y_true=true_labels, y_pred= results))

if __name__ == "__main__":
    main()
    print("Complete")
