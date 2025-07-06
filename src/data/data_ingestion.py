import os
from src.connections.mongodb_connection import MongoDBClient
from src.constants import DATABASE_NAME,COLLECTION_NAME
from src.exception import MyException
from src.logger import logging
import sys
import gridfs



class image_search_engine_data:

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)
        
    def get_raw_data_from_source(self,collection_name,database_name=None):
        try:
            # Access specified collection from the default db initialized from constructor or specified database.....
            if database_name is None:
                db=self.mongo_client.database
            else:
                db = self.mongo_client.client[database_name]

            fs = gridfs.GridFS(db, collection=collection_name)

            # Set your desired output directory here
            output_root = r"./data/raw/"

            # Make sure the output directory exists
            os.makedirs(output_root, exist_ok=True)

            logging.info("Ingesting raw data from source...")

            # Restore all files from GridFS
            for file in fs.find():
                # Get filename and subfolder from metadata
                filename = file.filename
                subfolder = file.subfolder
                # print(subfolder)

                # Construct full output path
                output_dir = os.path.join(output_root, subfolder)
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(output_dir, filename)

                # Write file to disk
                with open(output_path, "wb") as out_file:
                    out_file.write(file.read())

                logging.info(f"Restored: {output_path}")

        except Exception as e:
            raise MyException(e, sys)
        


if __name__=="__main__":
    data_ingestion=image_search_engine_data().get_raw_data_from_source(COLLECTION_NAME,DATABASE_NAME)