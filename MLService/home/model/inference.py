import pandas as pd
import home.model.settings as st
import home.model.features as ft
import joblib
from home.model.exceptions import ModelParamException

def get_feature_vector(feature_dict):
    # Validate input
    input_vars = set(feature_dict.keys())
    input_vars.remove("csrfmiddlewaretoken")

    if input_vars != st.MODEL_INPUT:
        # Check if there is missing param
        missing = set(st.MODEL_INPUT) - set(input_vars)
        if len(missing) > 0:
            raise ModelParamException(f'Missing params: {list(missing)}')

    text = str(feature_dict["text"][0])
    data = [ft.get_third_person(text),
            ft.get_first_person_plural(text),
            ft.ht_find_keywords(text),
            ft.service_is_restricted(text),
            ft.service_place(text)]

    # Get feature vector.
    X = pd.DataFrame(columns = st.FINAL_DATA)
    X.loc[0] = data
    return X

def get_model():
    model = joblib.load(st.MODEL_PATH)
    return model

def calculate_score(X, model):
    proba = model.predict_proba(X)[0][1]
    proba = round(float(proba), 6)
    return proba

def get_model_response(json_data):
    X = get_feature_vector(json_data)
    model = get_model()
    probability = calculate_score(X, model)
    if probability >= st.THRESHOLD:
        suspicious = 1
        label = 'SUSPICIOUS'
    else:
        suspicious = 0
        label = 'NOT SUSPICIOUS'
    return {
        'status': 'ok',
        'score': probability,
        'label': label,
        'suspicious': suspicious,
        'version': 'v1.0.0',
    }