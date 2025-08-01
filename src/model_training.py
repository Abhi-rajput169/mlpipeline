import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_path=os.path.join(log_dir,'model_training_logs')
file_handler=logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)-> pd.DataFrame:
    """ load data from a csv file
    :param file_path:path to the csv file
    :return: loaded dataframe
    """
    try:
        df=pd.read_csv(file_path)
        logger.debug('data loaded from %s with shape %s',file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv file:%s',e)
        raise
    except FileNotFoundError as e:
        logger.error('file not found:%s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occurred while loading the data:%s',e)
        raise

def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    """
    train the random forest model"""
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("the number of samples in X_train and y_train must be the same.")
        
        logger.debug('initializing randomforest model with parameters:%s',params)
        rf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug('model training started with %d samples',X_train.shape[0])
        rf.fit(X_train,y_train)
        logger.debug('model training complted')
        return rf
    except ValueError as e:
        logger.error('valuerror during model training:%s',e) 
        raise
    except Exception as e:
        logger.error('error during model training:%s,e')
        raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the model using pickle
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)

    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise

    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Model hyperparameters
        params = {'n_estimators': 25, 'random_state': 2}

        # Load training data
        train_data = load_data(r'data\new_data\train_tfidf.csv')

        # Extract features and labels
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train model
        clf = train_model(X_train, y_train, params)

        # Save the model
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

        

