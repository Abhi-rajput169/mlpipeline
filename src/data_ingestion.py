import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

#ensure the logs directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#logging configuration
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handeler=logging.StreamHandler()
console_handeler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handeler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handeler)
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

def load_data(data_url:str) -> pd.DataFrame:
    """load data from csv file"""
    try:
        df=pd.read_csv(data_url)
        logger.debug('Data loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv file:%s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the data : %s',e)
        raise

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """preprocess the data"""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug('data preprocess completed')
        return df
    except KeyError as e:
        logger.error('missing coloumn in the data frame:%s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while preprocessing the data : %s',e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str) -> None:
    """save the train and test data"""
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug('train  and test data save succesfuly to %s',raw_data_path)
    except Exception as e:
        logger.error('unexpected error occured while preprocessing the data : %s',e)
        raise

def main():
    try:
        params=load_params(params_path='params.yaml')
        test_pize=params['data_ingestion']['test_size']
        data_path='https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv'
        df=load_data(data_url=data_path)
        finl_df=preprocess_data(df)
        train_data,test_data=train_test_split(finl_df,test_size=test_pize,random_state=2)
        save_data(train_data=train_data,test_data=test_data,data_path='./data')
    except Exception as e:
        logger.error('failed to complete the data ingestion process:%s',e)
        print(f"error:{e}")

if __name__=='__main__':
    main()
