from typing import List
from pyspark.sql.types import TimestampType, StringType, FloatType, StructType, StructField
from exception import InsuranceException
import os, sys

from pyspark.sql import DataFrame
from typing import Dict


class InsuranceDataSchema:

    def __init__(self):
        self.col_age: str = 'age'
        self.col_sex: str = 'sex'
        self.col_bmi = 'bmi'
        self.col_children: str = 'children'
        self.col_smoker: str = 'smoker'
        self.col_region: str = 'region'
        self.col_expenses: str = 'expenses'

    @property
    def dataframe_schema(self) -> StructType:
        try:
            schema = StructType([
                StructField(self.col_age, StringType()),
                StructField(self.col_sex, StringType()),
                StructField(self.col_bmi, StringType()),
                StructField(self.col_children, StringType()),
                StructField(self.col_smoker, TimestampType()),
                StructField(self.col_region, TimestampType()),
                StructField(self.col_expenses, StringType()),

            ])
            return schema

        except Exception as e:
            raise InsuranceException(e, sys) from e

    @property
    def target_column(self) -> str:
        return self.col_expenses

    @property
    def categorical_features(self) -> List[str]:
        features = [
            self.col_sex,
            self.col_smoker,
            self.col_region,
        ]
        return features

   

    @property
    def numerical_features(self) -> List[str]:
        features = [
            self.col_age,
            self.col_bmi,
            self.col_children,
        ]
        return features

    @property
    def one_hot_encoding_features(self) -> List[str]:
        features = [
            self.col_sex,
            self.col_smoker,
            self.col_region,
            
        ]
        return features

    @property
    def input_features(self) -> List[str]:
        in_features = self.categorical_features + self.numerical_features
        return in_features

    @property
    def required_columns(self) -> List[str]:
        features = [self.target_column] + self.categorical_features + self.numerical_features
        return features

    @property
    def string_index_output(self) -> List[str]:

        return [f"si_{col}" for col in self.categorical_features]

    @property
    def tf_one_hot_encoding_features(self) -> List[str]:
        return [f"tf_{col}" for col in self.one_hot_encoding_features]

    

    @property
    def vector_assembler_output(self) -> str:
        return "va_input_features"

    

    @property
    def scaled_vector_input_features(self) -> str:
        return "scaled_input_features"

    @property
    def target_indexed_label(self) -> str:
        return f"indexed_{self.target_column}"

    @property
    def prediction_column_name(self) -> str:
        return "prediction"

    @property
    def prediction_label_column_name(self) -> str:
        return f"{self.prediction_column_name}_{self.target_column}"
