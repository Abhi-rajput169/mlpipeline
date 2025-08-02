import os
import pandas as pd
import pickle
import json
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import logging


log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_path=os.path.join(log_dir,'model_evaluation_logs')
file_handler=logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path:str):
    """we will load the model from a file"""
    try:
        with open(file_path,'rb') as file:
            model=pickle.load(file)
        logger.debug('model loaded from %s',file_path)
        return model
    except FileNotFoundError :
        logger.error('file not found:%s',file_path)
        raise
    except Exception as e:
        logger.error('unexpexted error came:%s',e)
        raise
def load_data(file_path:str)->pd.DataFrame:
    """ load the data from th csv file"""
    try:
        df=pd.read_csv(file_path)
        logger.debug('data loaded from the %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file :%s",e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the data :%s',e)
        raise

def evaluate_model(rf,X_test:np.ndarray,y_test:np.ndarray)-> dict:
    """evaluate the model and return the evaluation metrics."""
    try:
        y_pred=rf.predict(X_test)
        y_pred_proba=rf.predict_proba(X_test)[:,1]

        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_proba)

        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug('model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('error during model evaluation:%s',e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        clf = load_model(r'models\model.pkl')
        test_data = load_data(r'data\new_data\test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)

        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()