from typing import List
from pyspark.sql.types import TimestampType, StringType, FloatType, StructType, StructField
from exception import InsuranceException
import os, sys

from pyspark.sql import DataFrame
from typing import Dict


class FinanceDataSchema:

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
            raise FinanceException(e, sys) from e

    @property
    def target_column(self) -> str:
        return self.col_consumer_disputed

    @property
    def one_hot_encoding_features(self) -> List[str]:
        features = [
            self.col_company_response,
            self.col_consumer_consent_provided,
            self.col_submitted_via,
        ]
        return features

    @property
    def im_one_hot_encoding_features(self) -> List[str]:
        return [f"im_{col}" for col in self.one_hot_encoding_features]

    @property
    def string_indexer_one_hot_features(self) -> List[str]:
        return [f"si_{col}" for col in self.one_hot_encoding_features]

    @property
    def tf_one_hot_encoding_features(self) -> List[str]:
        return [f"tf_{col}" for col in self.one_hot_encoding_features]

    @property
    def tfidf_features(self) -> List[str]:
        features = [
            self.col_issue
        ]
        return features

    @property
    def derived_input_features(self) -> List[str]:
        features = [
            self.col_date_sent_to_company,
            self.col_date_received
        ]
        return features

    @property
    def derived_output_features(self) -> List[str]:
        return [self.col_diff_in_days]

    @property
    def numerical_columns(self) -> List[str]:
        return self.derived_output_features

    @property
    def im_numerical_columns(self) -> List[str]:
        return [f"im_{col}" for col in self.numerical_columns]

    @property
    def tfidf_feature(self) -> List[str]:
        return [self.col_issue]

    @property
    def tf_tfidf_features(self) -> List[str]:
        return [f"tf_{col}" for col in self.tfidf_feature]

    @property
    def input_features(self) -> List[str]:
        in_features = self.tf_one_hot_encoding_features + self.im_numerical_columns + self.tf_tfidf_features
        return in_features

    @property
    def required_columns(self) -> List[str]:
        features = [self.target_column] + self.one_hot_encoding_features + self.tfidf_features + \
                   [self.col_date_sent_to_company, self.col_date_received]
        return features

    @property
    def required_prediction_columns(self) -> List[str]:
        features =  self.one_hot_encoding_features + self.tfidf_features + \
                   [self.col_date_sent_to_company, self.col_date_received]
        return features



    @property
    def unwanted_columns(self) -> List[str]:
        features = [
            self.col_complaint_id,
            self.col_sub_product, self.col_complaint_what_happened]

        return features

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
