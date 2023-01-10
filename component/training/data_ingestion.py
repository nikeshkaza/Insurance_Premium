import os
import re
import sys
import time
import uuid
from collections import namedtuple
from typing import List

import json
import pandas as pd
from pandas import DataFrame
import requests
#aa

from config.pipeline.training import InsuranceConfig
from constant.training_pipeline_config.data_ingestion import DATABASE_NAME, COLLECTION_NAME
from component.configuration.mongo_db_connection import MongoDBClient
from pyspark.sql import SparkSession

from entity.artifact_entity import DataIngestionArtifact
from entity.config_entity import DataIngestionConfig
from exception import InsuranceException
from logger import logger
from insurance_data import InsuranceData
from datetime import datetime

DownloadUrl = namedtuple("DownloadUrl", ["url", "file_path" ])


class DataIngestion:
    # Used to download data in chunks.
    def __init__(self, data_ingestion_config: DataIngestionConfig ):
        """
        data_ingestion_config: Data Ingestion config
        n_retry: Number of retry filed should be tried to download in case of failure encountered
        n_month_interval: n month data will be downloded
        """
        try:
            logger.info(f"{'>>' * 20}Starting data ingestion.{'<<' * 20}")
            self.data_ingestion_config = data_ingestion_config
            self.failed_download_urls: List[DownloadUrl] = []
            

        except Exception as e:
            raise InsuranceException(e, sys)


    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export mongo db collection record as data frame XX-into feature-XX
        """
        try:
            logger.info("Exporting data from mongodb to feature store")
            insurance_data = InsuranceData()
            dataframe = insurance_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME)
            data_ingestion_dir = self.data_ingestion_config.data_ingestion_dir 
            file_name = self.data_ingestion_config.file_name     
            download_files_dir=self.data_ingestion_config.download_dir      

            #creating folder
            dir_path = os.path.dirname(data_ingestion_dir)
            download_path=os.path.join(dir_path,download_files_dir)
            os.makedirs(download_path,exist_ok=True)
            
            file_path = os.path.join(dir_path,download_files_dir,file_name)
            dataframe.to_csv(file_path,index=False,header=True)
            return dataframe
        except  Exception as e:
            raise  InsuranceException(e,sys)

    

    def convert_files_to_parquet(self, ) -> str:
        """
        downloaded files will be converted and merged into single parquet file
        json_data_dir: downloaded json file directory
        data_dir: converted and combined file will be generated in data_dir
        output_file_name: output file name 
        =======================================================================================
        returns output_file_path
        """
        try:
            spark_session=SparkSession.builder.appName('Insurace_Premium').getOrCreate()
            csv_data_dir = self.data_ingestion_config.download_dir
            data_dir = self.data_ingestion_config.feature_store_dir
            output_file_name = self.data_ingestion_config.file_name
            os.makedirs(data_dir, exist_ok=True)
            file_path = os.path.join(data_dir, f"{output_file_name}")
            logger.info(f"Parquet file will be created at: {file_path}")
            if not os.path.exists(csv_data_dir):
                return file_path
            for file_name in os.listdir(csv_data_dir):
                json_file_path = os.path.join(csv_data_dir, file_name)
                logger.debug(f"Converting {json_file_path} into parquet format at {file_path}")
                df = spark_session.read.csv(csv_data_dir, header = True)
                if df.count() > 0:
                    df.repartition(3).write.mode('overwrite').parquet(file_path)

            return file_path
        except Exception as e:
            raise InsuranceException(e, sys)

    

    


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info(f"Started downloading json file")
            

            self.export_data_into_feature_store()
            
            if os.path.exists(self.data_ingestion_config.download_dir):
                logger.info(f"Converting and combining downloaded json into parquet file")
                self.convert_files_to_parquet()
                

            feature_store_file_path = os.path.join(self.data_ingestion_config.feature_store_dir,
                                                   self.data_ingestion_config.file_name)
            artifact = DataIngestionArtifact(
                feature_store_file_path=feature_store_file_path,
                download_dir=self.data_ingestion_config.download_dir

            )

            logger.info(f"Data ingestion artifact: {artifact}")
            return artifact
        except Exception as e:
            raise InsuranceException(e, sys)


def main():
    try:
        config = InsuranceConfig()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise InsuranceException(e, sys)


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        logger.exception(e)
