import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re

#ensure the log directotyt exist
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#setting up the logger
logger=logging.getLogger('data_processing')
logger.setLevel('DEBUG')

#setting up the handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_path=os.path.join(log_dir,'data_preprocesing.log')
file_handler=logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

#formatter
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """ transform the text by convertig the lowwercase,removing the stopwords and stemimg"""

    #remove punctuations and remove the nukbers extra
    text=re.sub(r'[^A-Za-z\s]','',text).lower()

    #WORD TOKENIZE
    text=nltk.word_tokenize(text)

    #set the stopwords
    stop_words=set(stopwords.words('english'))
    text=[word for word in text if word not in stop_words]

    # setting up the stemmer
    stemmer=PorterStemmer()
    text=[stemmer.stem(word) for word in text]

    #join the tokens back
    return " ".join(text)

def preprocess_df(df,text_column='text',target_column='target'):
    """
    preprocess the df by encoding the taregt column and removing the duplicates from the df"""
    try:
        logger.debug("df preprocess stated")
        #encode the target column
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug("label encoding done")

        #remove the duplicates
        df=df.drop_duplicates()
        logger.debug('duplicates remove')

        #apply the tranform_text to text
        df.loc[:, text_column]=df[text_column].apply(transform_text)
        logger.debug('text coloumn transformed')
        return df
    except KeyError as e:
        logger.error('column not found:%s',e)
        raise
    except Exception as e:
        logger.error('error during text normalization:%s',e)
        raise

def main(text_column='text',target_column='target'):
    """main function to load data and preorces the data and save it"""
    try:
        train_data=pd.read_csv(r'data\raw\train.csv')
        test_data=pd.read_csv(r'data\raw\test.csv')
        logger.debug('data loaded properly')

        #transform the data
        train_process_data=preprocess_df(train_data,text_column=text_column,target_column=target_column)
        test_process_data=preprocess_df(test_data,text_column=text_column,target_column=target_column)
    
        data_path=os.path.join("./data",'processed')
        os.makedirs(data_path,exist_ok=True)

        train_process_data.to_csv(os.path.join(data_path,'training_process_csv'),index=False)
        test_process_data.to_csv(os.path.join(data_path,'testing_process_csv'),index=False)

        logger.debug('proccessed data saved to %s',data_path)

    except FileNotFoundError as e:
        logger.error('file not found:%s',e)
    except pd.errors.EmptyDataError as e:
        logger.error('no data:%s',e)
    except Exception as e:
        logger.error('failed to complete the data transformtaion process:%s',e)
        print(f"error:{e}")

if __name__=='__main__':
    main()


