import os
from torch.utils.data import Dataset,DataLoader
from typing import Tuple
from PIL import Image
import torch
import numpy as np
from src.logger import logging
import random
from src.utils import load_processed_data_artifacts


class create_dataset(Dataset):
    def __init__(self,data,label,transform=None):
        logging.info('Starting creation of Dataset Class')
        self.data_paths=data
        self.transform=transform
        self.label=label
        logging.info('Successfully created Dataset Class')

    def load_img(self,image)->Image.Image:
        try:
            img=Image.open(image).convert('RGB')
            img=img.resize((224,224),resample=Image.LANCZOS) ### resizing to match mobilenet input dimention.
        except Exception as e:
            logging.error("During loading the image using pil there was some error: %s",e)
            raise
        else:
            return img

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index)->Tuple[torch.tensor,int]:
        data_path=self.data_paths[index]
        label=self.label[index]
        image=self.load_img(data_path)
        # image=torch.asarray(image)
        if self.transform:
            transformed_image=self.transform(image)
            return transformed_image,label
        else:
            return torch.asarray(np.array(image)),label
        

def get_data_loaders(train_data,test_data,num_worker,batch_size,seed):
    try:
        logging.info("Creating dataloader class")
        def seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        train_data_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,pin_memory=True ,num_workers=num_worker,worker_init_fn=seed_worker,generator=g)
        test_data_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=num_worker,worker_init_fn=seed_worker,generator=g)
    except Exception as e:
        logging.error("Unexpected error during creation of dataloader class: %s",e)
        raise
    else:
        logging.info("Succesfully created the data loader class")    
        return train_data_loader,test_data_loader
    

if __name__=="__main__":
    train_data,train_label,validation_data,validation_label,label_encoder=load_processed_data_artifacts.load_data_and_label()
    train_dataset_class=create_dataset(train_data,train_label)
    validation_dataset_class=create_dataset(validation_data,validation_label)
    # print(set(data.label))
    ### Testing get_data_loader
    Train_loader,Test_loader=get_data_loaders(train_dataset_class,validation_dataset_class,0,32,42)

    