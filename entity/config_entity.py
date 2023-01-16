from collections import namedtuple

TrainingPipelineConfig = namedtuple("PipelineConfig", ["pipeline_name", "artifact_dir"])
DataIngestionConfig = namedtuple("DataIngestionConfig", [
                                                         "data_ingestion_dir",
                                                         "download_dir",
                                                         "file_name",
                                                         "feature_store_dir",
                                                         "failed_dir",
                                                         "datasource_url"
                                                         ])

DataValidationConfig = namedtuple('DataValidationConfig', ["accepted_data_dir", "rejected_data_dir", "file_name"])


DataTransformationConfig = namedtuple('DataTransformationConfig', ['file_name', 'export_pipeline_dir',
                                                                   'transformed_train_dir', "transformed_test_dir",

                                                                   "test_size"
                                                                   ])