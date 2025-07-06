import torch
from src.logger import logging
from src.exception import MyException
import sys
import os

def saving_model_with_state_and_logs(model,optimizer,results,file="model.pt"):
    try:
        os.makedirs('checkpoints',exist_ok=True)
        path=os.path.join('checkpoints',file)
        contents={
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'history':results,
        }
        torch.save(contents,path)
    except Exception as e:
        logging.error("There was an unexpect error during saving the model artifacts")
        raise MyException(e,sys)
    else:
        logging.info(f"Succesfully saved the model artifact at {path}")