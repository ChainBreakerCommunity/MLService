import utils.preprocess as pp
import utils.settings as st

def main():
    #pp.compute_features(
    #    raw_path = st.DATASET_PATH,
    #    raw_processed_path = st.DATASET_PROCESSED_PATH
    #)
    pp.split_dataset(
        raw_processed_path = st.DATASET_PROCESSED_PATH, 
        test_size = 0.2, 
        metric_name = st.PRED_VARIABLE,
        train_dataset_path = st.TRAIN_DATASET_PATH, 
        test_dataset_path = st.TEST_DATASET_PATH
    )
    
if __name__ == "__main__":
    main()