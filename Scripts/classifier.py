
from unicodedata import category
import pandas as pd
import os
import pickle
import numpy as np
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
from sklearn.ensemble import ExtraTreesClassifier
from ast import literal_eval
from tpot import TPOTClassifier

def main():
    test_features = os.path.join("ClassifierFiles", "ClassDF_Test.csv")
    test_df = pd.read_csv(test_features)
    test_features = test_df[["hero", "villain", "victim", "other", "dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert',]]
    # test_features["dom_tweet_sent_siebert"] = test_features["dom_tweet_sent_siebert"].astype("category")
    test_ft = np.genfromtxt("ClassDF_Test_FT.csv",delimiter=',')
    test_npy = test_features.to_numpy()
    print(test_ft)
    print(test_npy.shape)
    print(test_ft.shape)
    final_test_features = np.concatenate((test_npy,test_ft),axis=1)
    print("FEATURES DIM:")
    print(final_test_features.shape)
    
    true_labels = test_df["label"]

    featurepath = os.path.join("ClassifierFiles", "ClassDF_Train.csv")
    featuresdf  = pd.read_csv(featurepath)
    features = featuresdf[["hero", "villain", "victim", "other", "dom_tweet_sent_siebert", 'Negative_siebert', 'Neutral_siebert', 'Positive_siebert']] 
    train_ft = np.genfromtxt("ClassDF_Train_FT.csv",delimiter=',')
    train_npy = features.to_numpy()
    final_train_features = np.concatenate((train_npy,train_ft),axis=1)
    print("FEATURES DIM:")
    print(final_train_features.shape)
    # features["dom_tweet_sent_siebert"] = features["dom_tweet_sent_siebert"].astype("category")
    labels = featuresdf["label"]

    models = ["TPOT"]

    for modelname in models:
        if modelname == "TPOT":
            # model = TPOTClassifier(generations=5, population_size=50, verbosity=3, random_state=42, scoring='f1_macro')
            model = ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.45, min_samples_leaf=16, min_samples_split=10, n_estimators=100)

        print("Model: {}".format(modelname))
        if modelname == "RF":
            param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [4,5,6,7],
            'criterion' :['gini', 'entropy']}
            model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=-1, scoring='f1_macro', verbose=3, refit=True)

        elif modelname == "SVM":
            param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear','rbf']}
            model = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit = True, verbose = 3, scoring='f1_macro', n_jobs=-1)
        
        
        model.fit(final_train_features,labels)
        # model.export('tpot_pipeline.py')
        # print(model.score(final_test_features, true_labels))
        
        # print(model.best_params_)
        path = os.path.join("models","model{}.pkl".format(modelname))
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        print("Model saved")
        print("Done")
        
        results = model.predict(X=final_test_features)
        results = results.tolist()


        print(classification_report(y_true=true_labels, y_pred= results))
        test_df["{}_pred".format(modelname)] = results
    test_df.to_csv(os.path.join("ClassifierFiles", "Predictions_Test.csv"), index=False)


if __name__ == "__main__":
    main()
    print("Complete")
