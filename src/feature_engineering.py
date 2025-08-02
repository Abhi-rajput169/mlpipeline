import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import yaml

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_path=os.path.join(log_dir,'feature_engineering_logs')
file_handler=logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str):
    """load the params file rom the the path"""
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug('Parameters retrived from %s',params_path)
        return params
    except FileNotFoundError:
        logger.error('file not found:%s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error:%s',e)
        raise
    except Exception as e:
        logger.error('unexpected error:%s',e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    """ load the data from csv file"""
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug("data loaded and nan filled from %s',file_pat")
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv fle:%s',e)
        raise
    except Exception as e:
        logger.error('unexpexted errorwhiele loading:%s',e)
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TF-IDF vectorization to training and testing data.

    Parameters:
        train_data (pd.DataFrame): Training dataset containing 'text' and 'target' columns.
        test_data (pd.DataFrame): Testing dataset containing 'text' and 'target' columns.
        max_features (int): Maximum number of features for TF-IDF.

    Returns:
        tuple: (train_df, test_df) with TF-IDF features and 'label' column.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        x_train = train_data['text'].values
        y_train = train_data['target'].values
        x_test = test_data['text'].values
        y_test = test_data['target'].values

        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug('TF-IDF applied and data transformed')
        return train_df, test_df

    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the dataframe to a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise
def main():
    try:
        params=load_params(params_path='params.yaml')
        max_features=params['feature_engineering']['max_features']

        train_data = load_data(r'data\processed\training_process_csv')
        test_data = load_data(r'data\processed\testing_process_csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "new_data", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "new_data", "test_tfidf.csv"))

    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()