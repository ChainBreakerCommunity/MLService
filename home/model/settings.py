import joblib
import json

APP = 'ht-ai-detection'
import os
print(os.getcwd())

t_file = open('./home/model/data/metrics.json')
threshold = json.load(t_file)["optimal_threshold"]
t_file.close()

# Variables
with open('./home/model/data/vars_raw.json', 'r') as file:
    vars_input = json.load(file)
with open('./home/model/data/vars_filtered.json', 'r') as file:
    vars_filtered_input = json.load(file)
with open("./home/model/data/metrics.json", "r") as file:
    metrics = json.load(file)

MODEL_INPUT = vars_input.get("MODEL_INPUT")
FINAL_DATA = vars_filtered_input.get("FINAL_DATA")
BOOLEAN_FEATURES = vars_input.get("BOOLEAN_FEATURES")
MODEL_PATH = "./home/model/data/model.joblib.dat"
THRESHOLD = metrics["optimal_threshold"]