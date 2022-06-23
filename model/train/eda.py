import pandas as pd
import sweetviz as sv
import utils.settings as st

def generate_report():
    df = pd.read_csv(st.TRAIN_DATASET_PATH)
    df.drop("Unnamed: 0", axis = 1, inplace = True)
    my_report = sv.analyze(source=df,
                           target_feat= st.PRED_VARIABLE,
                           pairwise_analysis="on")
    my_report.show_html(filepath="eda.html")

if __name__ == "__main__":
    generate_report()