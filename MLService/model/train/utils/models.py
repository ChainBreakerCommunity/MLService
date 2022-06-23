import json
from importlib_metadata import distribution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import utils.settings as st
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from sklearn.svm import LinearSVC
from typing import List

def prepare_randomforest_classifier():

    # Initialize RandomForestClassifier
    clf = RandomForestClassifier(
        criterion = st.CRITERION,
        random_state = st.RANDOM_STATE, 
        max_features = st.MAX_FEATURES
    )

    # Read Train/Test
    train = pd.read_csv(st.TRAIN_DATASET_PATH)
    test = pd.read_csv(st.TEST_DATASET_PATH)

    # Train
    y_train = train.pop(st.PRED_VARIABLE)
    X_train = train[st.FINAL_DATA]

    # Test
    y_test = test.pop(st.PRED_VARIABLE)
    X_test = test[st.FINAL_DATA]
    return clf, X_train, y_train, X_test, y_test

def feature_selection(
        train_dataset_path: str, 
        metric_name: str, 
        vars_filtered_path: str
    ):
    """
    Feature selection using Support Vector Machine with Linear Kernel
    K(x, y) = <x, y>_{R^d}
    with Lasso Regression (L1) for feature selection.
    """
    # Load dataset.
    data = pd.read_csv(train_dataset_path)
    data.drop("Unnamed: 0", axis = 1, inplace = True)
    X_train = data.drop(metric_name, axis = 1)
    Y_train = data[metric_name]

    # Eliminate 
    corr = data.corr()
    cor_target = abs(corr[metric_name])
    relevant_features = cor_target[cor_target > 0.2]
    names = [index for index, value in relevant_features.iteritems()]
    names.remove(metric_name)
    names = set(names)

    # Select L1 regulated features from LinearSVC output 
    selection = SelectFromModel(LinearSVC(C=1, penalty='l1', dual=False))
    selection.fit(X_train, Y_train)

    # Get best features.
    feature_names = data.drop(metric_name, axis = 1).columns[(selection.get_support())]
    feature_names = list(feature_names)

    # Intersection.
    #feature_names = list(feature_names.intersection(names))
    print("Selected variables:\n", feature_names)

    # Save final features.
    with open(vars_filtered_path, "w") as f:
        json.dump(
            {
                "FINAL_DATA": feature_names
            }, 
            f, indent=4
        )
    return feature_names

def train_randomforest(clf, X_train, y_train, cv = 10):
    # Initialize Random Search with CV
    rsearch = RandomizedSearchCV(
        estimator = clf,
        param_distributions = st.DISTRIBUTIONS,
        random_state = st.RANDOM_STATE,
        scoring = 'precision',
        cv = cv,
        verbose = 2,
        n_iter = 20
    )

    # Train model
    print('Training...\n')
    models = rsearch.fit(X_train, y_train)
    model = models.best_estimator_
    # Export model
    print('Exporting model...')
    joblib.dump(model, st.MODEL_PATH)
    return model

def predict_and_calculate_metrics(model, X_test, Y_test):
    '''Get model evaluation metrics on the test set.'''
    
    # Predict scores.
    y_pred_logits = model.predict_proba(X_test)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(Y_test, y_pred_logits)

    # Get metrics.
    f1 = (2 * (precisions * recalls)) / (precisions + recalls)
    threshold = float(thresholds[np.argmax(f1)])
    precision = precisions[np.argmax(f1)]
    recall = recalls[np.argmax(f1)]

    # Get labels according to best threshold.
    y_predict_r = (y_pred_logits >= threshold).astype(int)

    # Calculate evaluation metrics for assesing performance of the model.
    roc = roc_auc_score(Y_test, y_pred_logits)
    acc = accuracy_score(Y_test, y_predict_r)
    f1_metric = f1_score(Y_test, y_predict_r)

    metrics_dict = {
        f"optimal_threshold": round(threshold, 3),
        f"roc_auc_score": round(roc, 3), 
        f"accuracy_score": round(acc, 3),
        f"precision_score": round(precision, 3), 
        f"recall_score": round(recall, 3), 
        f"f1": round(f1_metric, 3)
    }

    # Save metrics
    with open(st.METRICS_PATH + "metrics.json", 'w') as fd:
        json.dump(
            metrics_dict,
            fd, indent=4
        )
    return y_pred_logits, y_predict_r, metrics_dict

def get_feature_importance(model, X_test):
    # Plot feature importance
    plt.figure(figsize=(15, 12))
    feat_importances = pd.Series(model.feature_importances_, index= X_test.columns)
    feat_importances.sort_values(ascending=False).plot(kind='barh', color = "red")
    plt.title("Permutation Feature Importance")
    plt.savefig(st.IMAGES_PATH + 'permutation_feature_importance.png')
    plt.close()

def save_predictions(X_train, y_train,
                     train_y_pred_scores, train_y_pred_labels,
                     X_test, y_test,
                     test_y_pred_scores, test_y_pred_labels):
    # Train dataset
    X_train['REAL_LABEL'] = y_train
    X_train['PREDICTED_SUSPICIOUS_SCORE'] = train_y_pred_scores
    X_train['PREDICTED_SUSPICIOUS_LABEL'] = train_y_pred_labels

    # Test dataset
    X_test['REAL_LABEL'] = y_test
    X_test['PREDICTED_SUSPICIOUS_SCORE'] = test_y_pred_scores
    X_test['PREDICTED_SUSPICIOUS_LABEL'] = test_y_pred_labels

    # Save datasets
    X_train.to_csv(st.TRAIN_SCORE_PATH, index=False)
    X_test.to_csv(st.TEST_SCORE_PATH, index=False)

if __name__ == "__main__":
    
    #feature_names = feature_selection(st.TRAIN_DATASET_PATH, st.PRED_VARIABLE, st.VARS_FILTERED_PATH)
    #print(feature_names)
    clf, X_train, y_train, X_test, y_test = prepare_randomforest_classifier() 
    #print(X_train)
    #model = train_randomforest(clf, X_train, y_train, cv = 5)
    #predict_and_calculate_metrics(model, X_test, y_test)

    import joblib
    model = joblib.load(st.MODEL_PATH)
    get_feature_importance(model, X_test)
    #train_randomforest(clf, X_train, y_train)

    save_predictions()