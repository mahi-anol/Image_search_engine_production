import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from src.logger import logging
from src.utils.get_parameters import load_params
import json
import joblib
import shutil

### CUSTOM EXCEPTION HANDLING CLASS
class DataLoadingError(Exception): ### Raised when no data is found in the directory than can be loaded.
    pass
class UnsupportedTypeError(Exception):
    pass

def is_supported_types(data_paths_with_label,support_types):
    unique_extentions=set()
    for label,data_path in data_paths_with_label:
        if not data_path.lower().endswith(support_types):
            logging.error(f'The following file is of unsupported type : {data_path}')
            return False
        else:
            _,ext=os.path.splitext(data_path)
            unique_extentions.add(ext)
    logging.info(f"Found the following data types in the dataset: {unique_extentions}")
    return True

def save_pkl_artifacts(**kwargs):

    logging.info("saving data and label encoder artifacts on ./data/processed....")

    os.makedirs('./data/processed',exist_ok=True)

    for key, value in kwargs.items():
        # Save paths
        joblib.dump(value, f'./data/processed/{key}.pkl')

    label_encoder=kwargs.get('label_encoder')
    # Optional: metadata
    with open("data/processed/classes.json", "w") as f:
        json.dump({"classes": list(label_encoder.classes_)}, f)

    logging.info("sucessfully saved data and label encoder artifacts on ./data/processed")



### getting the data.
def generate_processed_data(train_dataset_path,test_ratio,rand_state=42)->tuple:
    try: 
        logging.info("starting train/validation data separation and label encoding on the train set")
        data_paths_with_label=[] #store dataset and label
        supported_extentions=('.jpg','.png','.jpeg')
        for dir_path,dir_name,file_names in os.walk(train_dataset_path):
            if len(file_names)>0: ### CONSIDERING ONLY THOOSE FOLDERS AS CLASS WHICH HAS MORE THAN 1 FILE.
                for file_name in file_names:
                    if file_name.lower().endswith(supported_extentions):
                        data_paths_with_label.append((os.path.basename(dir_path),os.path.join(dir_path,file_name))) ### (label,data)
        if len(data_paths_with_label)<0:
            raise DataLoadingError('NO SUCH DATA FOUND THAT COULD BE LOADED')
        
        if not is_supported_types(data_paths_with_label,supported_extentions):
            raise UnsupportedTypeError("There are some unsupported file format present in the dataset. ")
        
    except DataLoadingError as e:
        logging.error('%s',e)
        raise
    except UnsupportedTypeError as e:
        logging.error('%s',e)
        raise
    except Exception as e:
        logging.error('There was some Unexpected ERROR')
        raise
    else:
        label,data=zip(*data_paths_with_label)
        le=LabelEncoder()
        encoded_label=le.fit_transform(label)

        ### Here the test the will be considired as validation data as a test data is already separated before....
        ### so here the test data is actualy referning to validation data.....
        #### actual test data can be found on ./data/processed/test folder....

        ### also the train data and test data is actaully refering to train data and test data paths.....Not the actually image data.....

        train_data,test_data,train_label,test_label=train_test_split(data,encoded_label,test_size=test_ratio,random_state=rand_state)

        ### save these artifacts in a pkl
        save_pkl_artifacts(train_data=train_data,validation_data=test_data,train_label=train_label,validation_label=test_label,label_encoder=le)
        logging.info("Successfully executed train and validation data separation with label encoding on the train set stored in ./data/processed")



        logging.info("copying test set as it is from ./data/interim to ./data/processed.")
        source_folder = "./data/interim/test"
        destination_folder = "./data/processed/test"

        # Remove destination folder if it exists
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
        # Copy folder
        shutil.copytree(source_folder, destination_folder)
        logging.info("successfully copied test set as it is from ./data/interim to ./data/processed.")
        logging.info("data process is completed and artifacts are stored on ./data/processed")
        

        

# logger.info(f"There are total {e.classes_} classes")
# logger.info(f"{}")
### Driver Code
if __name__=="__main__":
    params=load_params('./params.yaml')['data_spliting']
    generate_processed_data(r'./data/interim/train',params['test_size'],params['random_state'])
    # print(len(e.classes_))