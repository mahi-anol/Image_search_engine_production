import torch.nn as nn
import logging 
import os
from src.logger import logging


class transfer_learning_model(nn.Module):
    def __init__(self,b_model,output_class): 
        super().__init__()
        try:
            logging.info("Creating model")
            self.base_model=b_model
            self.base_model.classifier=nn.Sequential(
                nn.Dropout(p=0.2,inplace=False),
                nn.Linear(in_features=1280,out_features=1000,bias=True),
                nn.ReLU(),
                nn.Linear(in_features=1000,out_features=512,bias=True),
                nn.ReLU(),
                nn.Linear(in_features=512,out_features=1024,bias=True),
                nn.ReLU(),
                nn.Linear(in_features=1024,out_features=512,bias=True),
                nn.ReLU(),
                nn.Linear(in_features=512,out_features=256,bias=True),
                # nn.ReLU(),
                # nn.Linear(in_features=256,out_features=output_class)
            )
            self.label=nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=256,out_features=output_class)
            )
        except Exception as e:
            logging.error("During creation of the model, there was a unexpected error: %s",e)
            raise
        else:
            logging.info("Successfully created the model")

    def forward(self,input):
        feature_extractor=self.base_model(input)
        label=self.label(feature_extractor)
        return feature_extractor,label

def return_model(b_model,no_of_class):
    f_model=transfer_learning_model(b_model,no_of_class)
    return f_model