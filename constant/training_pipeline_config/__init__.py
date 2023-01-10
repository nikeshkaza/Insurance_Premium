import os

PIPELINE_NAME = "insurance-premium"
PIPELINE_ARTIFACT_DIR = os.path.join(os.getcwd(), "insurance_artifact")

from constant.training_pipeline_config.data_ingestion import *