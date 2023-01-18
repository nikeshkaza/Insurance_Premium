import os
import sys
from collections import namedtuple
from typing import List, Dict

from pyspark.sql import DataFrame,SQLContext
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler, VectorAssembler, OneHotEncoder, StringIndexer, Imputer
from pyspark.sql.types import TimestampType, StringType, FloatType, StructType, StructField, IntegerType
from pyspark.ml.pipeline import Pipeline

from config.spark_manager import spark_session
from entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from entity.config_entity import DataTransformationConfig
from entity.schema import InsuranceDataSchema
from exception import InsuranceException
from logger import logger

from pyspark.sql.functions import lit,array

"""
Start spark session
"""
#spark_session=SparkSession.builder.appName('Insurace_Premium').getOrCreate()


class DataTransformation():

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig,
                 schema=InsuranceDataSchema()
                 ):
        try:
            super().__init__()
            self.data_val_artifact = data_validation_artifact
            self.data_tf_config = data_transformation_config
            self.schema = schema
        except Exception as e:
            raise InsuranceException(e, sys)

    def read_data(self) -> DataFrame:
        try:
            file_path = self.data_val_artifact.accepted_file_path
            dataframe: DataFrame = spark_session.read.parquet(file_path)
            dataframe.printSchema()
            dataframe=dataframe.withColumn("age",dataframe["age"].cast(IntegerType()))
            dataframe=dataframe.withColumn("bmi",dataframe["bmi"].cast(FloatType()))    
            dataframe=dataframe.withColumn("children",dataframe["children"].cast(IntegerType()))  
            dataframe=dataframe.withColumn("expenses",dataframe["expenses"].cast(FloatType()))            
            dataframe.printSchema()
            dataframe.show()
            return dataframe
        except Exception as e:
            raise InsuranceException(e, sys)

    def read_data_sql(self) -> DataFrame:

        try:
           file_path = self.data_val_artifact.accepted_file_path 
           sqlContext= SQLContext(spark_session)
           df = sqlContext.read.format("parquet").load(file_path,schema=self.schema.dataframe_schema)
           #df=sqlContext.read.parquet(file_path)
           df.printSchema()
           df.show()
        except Exception as e:
            raise InsuranceException(e, sys)


    def get_data_transformation_pipeline(self) ->Pipeline:

        stages=[]

        for catcols,string_indexer_col in zip(self.schema.categorical_features,self.schema.string_index_output):
            string_indexer= StringIndexer(inputCol=catcols,outputCol=string_indexer_col)
            stages.append(string_indexer)

        one_hot_encoder=OneHotEncoder(inputCols=self.schema.string_index_output ,
                                        outputCols=self.schema.tf_one_hot_encoding_features)

        stages.append(one_hot_encoder)

        vector_assembler= VectorAssembler(inputCols=self.schema.vector_assembler_input,
                                            outputCol=self.schema.vector_assembler_output)

        stages.append(vector_assembler)

        pipeline=Pipeline(stages=stages)
        logger.info(f"Data transformation pipeline: [{pipeline}]")
        print(pipeline.stages)

        return pipeline



    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info(f">>>>>>>>>>>Started data transformation <<<<<<<<<<<<<<<")
            dataframe: DataFrame = self.read_data()
            # dataframe = self.get_balanced_shuffled_dataframe(dataframe=dataframe)
            logger.info(f"Number of row: [{dataframe.count()}] and column: [{len(dataframe.columns)}]")

            test_size = self.data_tf_config.test_size
            logger.info(f"Splitting dataset into train and test set using ration: {1 - test_size}:{test_size}")
            train_dataframe, test_dataframe = dataframe.randomSplit([1 - test_size, test_size])
            logger.info(f"Train dataset has number of row: [{train_dataframe.count()}] and"
                        f" column: [{len(train_dataframe.columns)}]")

            logger.info(f"Train dataset has number of row: [{train_dataframe.count()}] and"
                        f" column: [{len(train_dataframe.columns)}]")

            pipeline = self.get_data_transformation_pipeline()


            transformed_pipeline = pipeline.fit(train_dataframe)


            # selecting required columns
            

            transformed_trained_dataframe = transformed_pipeline.transform(train_dataframe)
            transformed_trained_dataframe = transformed_trained_dataframe.select(self.schema.final_required_columns)
            transformed_trained_dataframe.show()

            transformed_test_dataframe = transformed_pipeline.transform(test_dataframe)
            transformed_test_dataframe = transformed_test_dataframe.select(self.schema.final_required_columns)
            transformed_test_dataframe.show()

            export_pipeline_file_path = self.data_tf_config.export_pipeline_dir

            # creating required directory
            os.makedirs(export_pipeline_file_path, exist_ok=True)
            os.makedirs(self.data_tf_config.transformed_test_dir, exist_ok=True)
            os.makedirs(self.data_tf_config.transformed_train_dir, exist_ok=True)
            transformed_train_data_file_path = os.path.join(self.data_tf_config.transformed_train_dir,
                                                            self.data_tf_config.file_name
                                                            )
            transformed_test_data_file_path = os.path.join(self.data_tf_config.transformed_test_dir,
                                                           self.data_tf_config.file_name
                                                           )

            logger.info(f"Saving transformation pipeline at: [{export_pipeline_file_path}]")
            transformed_pipeline.write().overwrite().save(export_pipeline_file_path)
            logger.info(f"Saving transformed train data at: [{transformed_train_data_file_path}]")
            print(transformed_trained_dataframe.count(), len(transformed_trained_dataframe.columns))
            transformed_trained_dataframe.write.parquet(transformed_train_data_file_path)

            logger.info(f"Saving transformed test data at: [{transformed_test_data_file_path}]")
            print(transformed_test_dataframe.count(), len(transformed_trained_dataframe.columns))
            transformed_test_dataframe.write.parquet(transformed_test_data_file_path)

            data_tf_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_data_file_path,
                transformed_test_file_path=transformed_test_data_file_path,
                exported_pipeline_file_path=export_pipeline_file_path,

            )

            logger.info(f"Data transformation artifact: [{data_tf_artifact}]")
            return data_tf_artifact
        except Exception as e:
            raise InsuranceException(e, sys)