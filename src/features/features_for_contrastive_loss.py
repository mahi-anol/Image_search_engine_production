import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch
import numpy as np
from collections import defaultdict
from src.logger import logging
import random
from src.utils import load_processed_data_artifacts


class create_dataset(Dataset):
    def __init__(self,data,labels,transform=None):
        logging.info('Starting creation of Dataset Class')
        self.data_paths=data
        self.transform=transform
        self.labels=labels
        self.label_specific_indices=defaultdict(list) ### contains {'label(x)':[idxes]}

        for idx,label in enumerate(self.labels):
            self.label_specific_indices[label].append(idx) ### Adding label-wise indices
        logging.info('Successfully Created Dataset Class')

    def load_img(self,image):
        try:
            img=Image.open(image).convert('RGB')
            img=img.resize((224,224),resample=Image.LANCZOS) ### resizing to match mobilenet input dimention.
        except Exception as e:
            logging.error("During loading the image using pil there was some error: %s",e)
            raise
        else:
            return img

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        ### first sample
        first_item=self.data_paths[index] 
        first_item_label=self.labels[index] 

        ### second sample
        pos_or_neg=torch.randint(0,2,(1,))### choose a sample type randomly. pos=1,neg=0
        
        if pos_or_neg==1:
            ### a random index which has the same label as first item.
            temp_idx=torch.randint(0,len(self.label_specific_indices[self.labels[index]]),(1,))
            while temp_idx==index:
                temp_idx=torch.randint(0,len(self.label_specific_indices[self.labels[index]]),(1,)) ###contains a index for a specific label which is not __getitem__() index.

            desired_idx=self.label_specific_indices[self.labels[index]][temp_idx] ### a index which have the same label as __getitems()__ index
            second_item=self.data_paths[desired_idx]
            second_item_label=self.labels[desired_idx]
        else:
            ### a random index which has different index from the first item.
            available_labels=list(set(self.labels)-{self.labels[index]})### Available labels
            choosen_label_idx=torch.randint(0,len(available_labels),(1,))
            choosen_label=available_labels[choosen_label_idx]
            temp_idx=torch.randint(0,len(self.label_specific_indices[choosen_label]),(1,))
            desired_idx=self.label_specific_indices[choosen_label][temp_idx]
            second_item=self.data_paths[desired_idx]
            second_item_label=self.labels[desired_idx]
            

        ###loading and transforming sample

        image1=self.load_img(first_item)
        image2=self.load_img(second_item)

        if self.transform:
            transformed_image1=self.transform(image1)
            transformed_image2=self.transform(image2)

            ##return img1,label1,img2,label2,sample_type=pos or neg.....
            return transformed_image1,first_item_label,transformed_image2,second_item_label,pos_or_neg
        else:
            ##return img1,label1,img2,label2,sample_type=pos or neg.....
            return torch.asarray(np.array(image1)),first_item_label,torch.asarray(np.array(image2)),second_item_label,pos_or_neg
        

def get_data_loaders(train_data,test_data,num_worker=0,batch_size=32,seed=42):
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
        logging.info("Succesfully created the data loader class for train and test dataset class")    
        return train_data_loader,test_data_loader




## Testing contrastive loss dataset....

if __name__=="__main__":
    train_data,train_label,validation_data,validation_label,label_encoder=load_processed_data_artifacts.load_data_and_label()
    train_dataset_class=create_dataset(train_data,train_label)
    validation_dataset_class=create_dataset(validation_data,validation_label)
    # print(set(data.label))
    ### Testing get_data_loader
    Train_loader,Test_loader=get_data_loaders(train_dataset_class,validation_dataset_class,0,32,42)

    

