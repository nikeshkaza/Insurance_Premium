from time import strftime
from entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig
from constant.training_pipeline_config import *
from constant.training_pipeline_config.data_ingestion import *
from constant.training_pipeline_config.data_validation import *
from constant.training_pipeline_config.data_transformation import *
from logger import logger
from exception import InsuranceException
import sys
from datetime import datetime
from constant import TIMESTAMP

class InsuranceConfig:
    def __init__(self, pipeline_name=PIPELINE_NAME, timestamp=TIMESTAMP):
        """
        Organization: iNeuron Intelligence Private Limited

        """
        self.timestamp=timestamp
        self.pipeline_name = pipeline_name
        self.pipeline_config = self.get_pipeline_config()

    def get_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            artifact_dir = PIPELINE_ARTIFACT_DIR
            pipeline_config = TrainingPipelineConfig(pipeline_name=self.pipeline_name,
                                                     artifact_dir=artifact_dir)

            logger.info(f"Pipeline configuration: {pipeline_config}")

            return pipeline_config
        except Exception as e:
            raise InsuranceException(e, sys)
    def get_data_ingestion_config(self ) -> DataIngestionConfig:

        """
        from date can not be less than min start date

        if to_date is not provided automatically current date will become to date

        """


        """
        master directory for data ingestion
        we will store metadata information and ingested file to avoid redundant download
        """
        data_ingestion_master_dir = os.path.join(self.pipeline_config.artifact_dir,
                                                 DATA_INGESTION_DIR)

        # time based directory for each run
        data_ingestion_dir = os.path.join(data_ingestion_master_dir,
                                          self.timestamp)

        

        data_ingestion_config = DataIngestionConfig(
            data_ingestion_dir=data_ingestion_dir,
            download_dir=os.path.join(data_ingestion_dir, DATA_INGESTION_DOWNLOADED_DATA_DIR),
            file_name=DATA_INGESTION_FILE_NAME,
            feature_store_dir=os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR),
            failed_dir=os.path.join(data_ingestion_dir, DATA_INGESTION_FAILED_DIR),
            datasource_url=DATA_INGESTION_DATA_SOURCE_URL

        )
        logger.info(f"Data ingestion config: {data_ingestion_config}")
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """

        """
        try:
            data_validation_dir = os.path.join(self.pipeline_config.artifact_dir,
                                               DATA_VALIDATION_DIR, self.timestamp)

            accepted_data_dir = os.path.join(data_validation_dir, DATA_VALIDATION_ACCEPTED_DATA_DIR)
            rejected_data_dir = os.path.join(data_validation_dir, DATA_VALIDATION_REJECTED_DATA_DIR)

            data_preprocessing_config = DataValidationConfig(
                accepted_data_dir=accepted_data_dir,
                rejected_data_dir=rejected_data_dir,
                file_name=DATA_VALIDATION_FILE_NAME
            )

            logger.info(f"Data preprocessing config: {data_preprocessing_config}")

            return data_preprocessing_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            data_transformation_dir = os.path.join(self.pipeline_config.artifact_dir,
                                                   DATA_TRANSFORMATION_DIR, self.timestamp)

            transformed_train_data_dir = os.path.join(
                data_transformation_dir, DATA_TRANSFORMATION_TRAIN_DIR
            )
            transformed_test_data_dir = os.path.join(
                data_transformation_dir, DATA_TRANSFORMATION_TEST_DIR
            )

            export_pipeline_dir = os.path.join(
                data_transformation_dir, DATA_TRANSFORMATION_PIPELINE_DIR
            )
            data_transformation_config = DataTransformationConfig(
                export_pipeline_dir=export_pipeline_dir,
                transformed_test_dir=transformed_test_data_dir,
                transformed_train_dir=transformed_train_data_dir,
                file_name=DATA_TRANSFORMATION_FILE_NAME,
                test_size=DATA_TRANSFORMATION_TEST_SIZE,
            )

            logger.info(f"Data transformation config: {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise InsuranceConfig(e, sys)