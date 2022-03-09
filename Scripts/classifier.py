
from unicodedata import category
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import classification_report
# from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
def main():
    test_features = os.path.join("ClassifierFiles", "ClassDF_Test.csv")
    test_df = pd.read_csv(test_features)
    test_features = test_df[["hero", "villain", "victim", "dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert']]
    test_features["dom_tweet_sent_siebert"] = test_features["dom_tweet_sent_siebert"].astype("category")
    true_labels = test_df["label"]

    featurepath = os.path.join("ClassifierFiles", "ClassDF_Train.csv")
    featuresdf  = pd.read_csv(featurepath)
    features = featuresdf[["hero", "villain", "victim", "dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert']] 
    features["dom_tweet_sent_siebert"] = features["dom_tweet_sent_siebert"].astype("category")
    labels = featuresdf["label"]

    models = ["RF", "SVM", "LinSVC"]

    for modelname in models:
        print("Model: {}".format(modelname))
        if modelname == "RF":
            model = RandomForestClassifier(random_state=0)
        elif modelname == "SVM":
            model = SVC(decision_function_shape='ovo')
        elif modelname == "LinSVC":
            model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
        model.fit(X=features, y=labels)

        path = os.path.join("models","model{}.pkl".format(modelname))
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        print("Model saved")
        print("Done")
        
        results = model.predict(X=test_features)
        results = results.tolist()


        print(classification_report(y_true=true_labels, y_pred= results))
        test_df["{}_pred".format(modelname)] = results
    test_df.to_csv(os.path.join("ClassifierFiles", "Predictions_Test.csv"), index=False)


if __name__ == "__main__":
    main()
    print("Complete")
