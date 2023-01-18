import shutil

import yaml

from typing import List
from pyspark.sql import DataFrame
from exception import InsuranceException
from logger import logger
import os, sys
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,RegressionEvaluator





def get_score(dataframe: DataFrame, metric_name, label_col, prediction_col) -> float:
    try:
        evaluator = RegressionEvaluator(
            labelCol=label_col, predictionCol=prediction_col,
            metricName=metric_name)
        score = evaluator.evaluate(dataframe)
        print(f"{metric_name} score: {score}")
        logger.info(f"{metric_name} score: {score}")
        return score
    except Exception as e:
        raise InsuranceException(e, sys)