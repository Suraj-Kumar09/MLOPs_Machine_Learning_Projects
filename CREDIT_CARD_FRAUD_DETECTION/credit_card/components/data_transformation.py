import sys
import os
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

# PROJECT NAMESPACE CONSTANTS & ARTIFACTS
from credit_card.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from credit_card.entity.config_entity import DataTransformationConfig
from credit_card.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from credit_card.exception import CreditCardException
from credit_card.logger import logging
from credit_card.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from credit_card.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        :param data_validation_artifact: Output reference of data validation artifact stage
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            
            # YAML schema configuration load kiya memory me
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditCardException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Helper method to read static csv files securely into dataframes."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditCardException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            logging.info("Initializing StandardScaler, OneHotEncoder, OrdinalEncoder standard modules.")
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown='ignore')
            ordinal_encoder = OrdinalEncoder()

            # 🟢 SAFE FIX: .get() method lagaya taaki yaml me key na hone par empty list mile aur KeyError na aaye
            oh_columns = self._schema_config.get('oh_columns', [])
            or_columns = self._schema_config.get('or_columns', [])
            transform_columns = self._schema_config.get('transform_columns', [])
            num_features = self._schema_config.get('num_features', [])

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            # Dynamic list banayege taaki khali variables pipeline crash na karein
            transformers_list = []

            # Sirf unhi steps ko add karenge jinki keys data configuration me maujud hain
            if len(oh_columns) > 0:
                transformers_list.append(("OneHotEncoder", oh_transformer, oh_columns))
            
            if len(or_columns) > 0:
                transformers_list.append(("Ordinal_Encoder", ordinal_encoder, or_columns))
            
            if len(transform_columns) > 0:
                transformers_list.append(("Transformer", transform_pipe, transform_columns))
            
            if len(num_features) > 0:
                transformers_list.append(("StandardScaler", numeric_transformer, num_features))

            # Final preprocessor blueprint compilation
            preprocessor = ColumnTransformer(transformers_list)

            logging.info("Created preprocessor object from ColumnTransformer safely without anomalies.")
            return preprocessor

        except Exception as e:
            raise CreditCardException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        """
        try:
            # 🟢 CHECK: Pichle step ka validation status agar True hai tabhi aage badhega
            if self.data_validation_artifact.validation_status:
                logging.info("Starting production scale data transformation layer execution.")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the core preprocessor structural object")

                # Local files se data load operations matrix memory space me kiya
                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # Training split mapping variables separation
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                drop_cols = self._schema_config.get('drop_columns', [])
                if len(drop_cols) > 0:
                    logging.info(f"Dropping explicit columns target configurations lists: {drop_cols}")
                    input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)
                
                # Target column mapping check (Integers standard validation bypass tracking)
                target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())

                # Testing split mapping variables separation
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                if len(drop_cols) > 0:
                    input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)

                target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())

                logging.info("Applying preprocessor transformation scaling layers matrices on datasets split.")
                # Fit transform and transform matrices compilation
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Applying SMOTEENN oversampling strategy engine on highly imbalanced targets densities.")
                smt = SMOTEENN(sampling_strategy="minority")
                
                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )

                logging.info("Concatenating input arrays and resampled target columns vectors arrays together.")
                # Final numpy compression metrics building matrices rows
                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

                # Objects serialization and permanent saving loops
                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor pkl and transformed compressed matrices files successfully.")

                return DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise CreditCardException(e, sys) from e