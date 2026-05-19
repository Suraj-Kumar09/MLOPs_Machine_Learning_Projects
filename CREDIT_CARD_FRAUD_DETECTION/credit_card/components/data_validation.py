import json
import sys
import os
import pandas as pd
from pandas import DataFrame

# 🟢 LATEST FRAMEWORK: Naye Evidently (v0.4+) ke Reports aur Presets modules ko import kiya
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 🛑 CONFIGURATION IMPORT: Root namespace aur custom exception mapping targets
from credit_card.exception import CreditCardException
from credit_card.logger import logging  
from credit_card.utils.main_utils import read_yaml_file, write_yaml_file
from credit_card.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from credit_card.entity.config_entity import DataValidationConfig


class DataValidation:
    """
    Class Name  :   DataValidation
    Description :   Yeh class credit card dataset ki quality, column structures aur 
                    statistical data drift ko up-to-date standard par validate karti hai.
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            
            # Schema yaml file ko load kiya memory me dynamic checking ke liye
            self._schema_config = read_yaml_file(file_path=self.data_validation_config.schema_file_path)
        except Exception as e:
            raise CreditCardException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Description : Check karta hai ki input dataframe me column count exact schema ke barabar hai ya nahi.
        """
        try:
            # Columns length validation rule
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column count matching: [{status}]")
            return status
        except Exception as e:
            raise CreditCardException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Description : Dataset me numerical aur categorical features ki absolute existence verify karta hai.
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            
            # Step A: Numerical features verify loop
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            # Step B: Categorical labels verify loop
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            # Dono missing lists khali hone par hi validation True hogi
            return len(missing_categorical_columns) == 0 and len(missing_numerical_columns) == 0
        except Exception as e:
            raise CreditCardException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """Description : Helper method to extract CSV sheets securely."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditCardException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Description : Up-to-date Evidently Report engine se training aur testing data ke beech statistical drift dhoondhta hai.
        """
        try:
            # 🟢 LATEST STRUCTURE: Evidently ka modern Report framework bina kisi dependency conflict ke
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(reference_data=reference_df, current_data=current_df)

            # Execution string ko loadable JSON parameters me serialize kiya
            report = data_drift_report.json()
            json_report = json.loads(report)

            # Local artifact storage disk path initialize aur folder management automations
            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)

            # 🟢 LATEST STRUCTURE: Naye JSON dictionary schemas se metrics nikalne ka tarika
            n_features = json_report["metrics"][0]["result"]["number_of_columns"]
            n_drifted_features = json_report["metrics"][0]["result"]["number_of_drifted_columns"]

            logging.info(f"{n_drifted_features}/{n_features} features drifted statistically.")
            
            # Dataset drift status (True ya False) trace engine target area
            drift_status = json_report["metrics"][0]["result"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise CreditCardException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Description : Orchestrates step-by-step rules checks on columns length, existence, and data distribution loops.
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation process stack execution seamlessly.")
            
            # Data Ingestion folder snapshots se split files memory stack read operations
            train_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            # Suraksha Kachav 1: Validate Total Columns Length (31 check)
            if not self.validate_number_of_columns(dataframe=train_df):
                validation_error_msg += "Columns count mismatch in training sheet. "
            if not self.validate_number_of_columns(dataframe=test_df):
                validation_error_msg += "Columns count mismatch in testing sheet. "

            # Suraksha Kachav 2: Validate Schema Column Names Existence
            if not self.is_column_exist(df=train_df):
                validation_error_msg += "Schema features missing in training data split. "
            if not self.is_column_exist(df=test_df):
                validation_error_msg += "Schema features missing in testing data split. "

            validation_status = len(validation_error_msg) == 0

            # Suraksha Kachav 3: Mathematical Pattern Analytics using Modern Evidently Engine
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info("Statistical Data Drift detected between train and test distributions.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Schema structural checks failed: {validation_error_msg}")

            # Packing complete summary metrics to send inside Artifacts entities block
            return DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
        except Exception as e:
            raise CreditCardException(e, sys) from e