from statistics import mode
from exception import InsuranceException
import sys
from logger import logger,LOG_DIR
from entity.config_entity import ModelPusherConfig
from entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from pyspark.ml.pipeline import PipelineModel
from entity.estimator import S3InsuranceEstimator
from constant.training_pipeline_config import PIPELINE_ARTIFACT_DIR
from constant.s3bucket import *
from cloud_storage.s3_syncer import S3Sync
from constant import TIMESTAMP
import os,shutil


class ModelPusher:

    def __init__(self, model_trainer_artifact: ModelTrainerArtifact, model_pusher_config: ModelPusherConfig):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        self.s3_sync=S3Sync()

    def push_model(self) -> str:
        try:
            model_registry = S3InsuranceEstimator(bucket_name=self.model_pusher_config.bucket_name,s3_key=self.model_pusher_config.model_dir)
            model_file_path = self.model_trainer_artifact.model_trainer_ref_artifact.trained_model_file_path
            model_registry.save(model_dir=os.path.dirname(model_file_path),
                                key=self.model_pusher_config.model_dir
                                )
            # model = PipelineModel.load(self.model_trainer_artifact.model_trainer_ref_artifact.trained_model_file_path)
            # pushed_dir = self.model_pusher_config.model_dir
            # model.save(pushed_dir)
            return model_registry.get_latest_model_path()
        except Exception as e:
            raise InsuranceException(e, sys)

    def model_artifact_pusher(self):
        try:
            aws_buket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{TIMESTAMP}"
            aws_bucket_log_url = f"s3://{TRAINING_LOG_NAME}/logs/{TIMESTAMP}"
            log_path=LOG_DIR
            self.s3_sync.sync_folder_to_s3(folder = PIPELINE_ARTIFACT_DIR,aws_buket_url=aws_buket_url)
            self.s3_sync.sync_folder_to_s3(folder=log_path,aws_buket_url=aws_bucket_log_url)
            if os.path.exists(PIPELINE_ARTIFACT_DIR):
                shutil.rmtree(PIPELINE_ARTIFACT_DIR)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
        except Exception as e:
            raise InsuranceException(e,sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            pushed_dir = self.push_model()
            self.model_artifact_pusher()
            model_pusher_artifact = ModelPusherArtifact(model_pushed_dir=pushed_dir)
            logger.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise InsuranceException(e, sys)
