import os
import shutil
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.exception import MyException
from src.utils.get_parameters import load_params
import sys
# Original dataset folder (with subfolders for each class)
source_dir = './data/raw'
# New destination base directory
dest_dir = './data/interim'
train_dir = os.path.join(dest_dir, 'train')
test_dir = os.path.join(dest_dir, 'test')

# Create destination directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


def data_split(test_size,random_state):
    logging.info('starting dataset train/test split')
    try:
        # Iterate through each class folder
        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)

            if not os.path.isdir(class_path):
                continue  # Skip files

            images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]
            
            # Split images
            train_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)

            # Destination subdirectories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Move training images
            for img in train_images:
                shutil.copy2(os.path.join(class_path, img), os.path.join(train_class_dir, img))

            # Move testing images
            for img in test_images:
                shutil.copy2(os.path.join(class_path, img), os.path.join(test_class_dir, img))
    except Exception as e:
        raise MyException(e,sys)

    logging.info("Dataset split and copied successfully!")



def main():
    params=load_params('./params.yaml')
    test_size=params['data_spliting']['test_size']
    random_state=params['data_spliting']['random_state']
    data_split(test_size,random_state)


if __name__=='__main__':
    main()