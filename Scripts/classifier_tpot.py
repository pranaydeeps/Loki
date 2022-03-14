
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

    models = ["XGB"]

    for modelname in models:
        print("Model: {}".format(modelname))
        if modelname == "XGB":
            param_grid = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
            }
            xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='multi:softmax', use_label_encoder=False, verbosity=1)
            model = GridSearchCV(xgb, param_grid=param_grid, n_jobs=-1, scoring='f1_macro', verbose=3, refit=True)


        elif modelname == "RF":
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
        
        
        model.fit(X=features, y=labels)
        #print(model.best_params_)
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
