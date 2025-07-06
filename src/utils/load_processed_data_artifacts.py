import joblib
import json
from src.logger import logging
from src.exception import MyException
import sys

def load_data_and_label():
    try:
        logging.info("Loading processed data artifacts..........")
        # Load model inputs
        train_data = joblib.load('./data/processed/train_data.pkl')
        train_labels = joblib.load('data/processed/train_label.pkl')
        validation_data = joblib.load('data/processed/validation_data.pkl')
        validation_labels = joblib.load('data/processed/validation_label.pkl')
        label_encoder = joblib.load('data/processed/label_encoder.pkl')
        # Load metadata
        with open('data/processed/classes.json', 'r') as f:
            class_info = json.load(f)
            classes = class_info['classes']

        logging.info(f"Loaded processed data artifacts for {len(classes)} classes: {classes}")
        return train_data,train_labels,validation_data,validation_labels,label_encoder
    except Exception as e:
        raise MyException(e,sys)
    

if __name__=='__main__':
    load_data_and_label()