import numpy as np
import pandas as pd
import utils.settings as st
import utils.features as ft
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from collections import Counter

def compute_features(raw_path: str, raw_processed_path: str):
    """
    Input: raw_path of dataset.
    Output: None
    """
    # Read dataset
    data = pd.read_csv(raw_path, sep=";")
    data.drop("Unnamed: 0", inplace = True, axis = 1)

    # Get features based on text.
    data["first_person_plural"] = np.vectorize(ft.get_first_person_plural)(data["text"])
    data["third_person"] = np.vectorize(ft.get_third_person)(data["text"])
    data["service_is_restricted"] = np.vectorize(ft.service_is_restricted)(data["text"])
    data["ht_find_keywords"] = np.vectorize(ft.ht_find_keywords)(data["text"])
    data["service_place"] = np.vectorize(ft.service_place)(data["text"])
    data.columns = data.columns.str.upper()

    # Save new dataset.
    data.to_csv(raw_processed_path, sep = ";")

    # Return dataset.
    return data

def oversampling(X, Y):
    # Oversampling using SMOTE.
    sm = SMOTE(random_state = st.RANDOM_STATE)
    X_res, Y_res = sm.fit_resample(X, Y)
    return X_res, Y_res, 

def split_dataset(
        raw_processed_path: str, 
        test_size: float, 
        metric_name: str,
        train_dataset_path: str, 
        test_dataset_path: str
    ):

    # Read dataset
    data = pd.read_csv(raw_processed_path, sep=";")

    # Get variables.
    X = data[st.BOOLEAN_FEATURES]
    Y = data[st.PRED_VARIABLE]

    # Split dataset.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size = test_size, 
        stratify =Y, 
        random_state = st.RANDOM_STATE
    )

    # Oversampling train dataset using SMOTE.
    X_train, Y_train = oversampling(X_train, Y_train)

    # Save datasets.
    X_train[metric_name] = Y_train
    X_test[metric_name] = Y_test
    X_train.to_csv(train_dataset_path)
    X_test.to_csv(test_dataset_path)

    return None