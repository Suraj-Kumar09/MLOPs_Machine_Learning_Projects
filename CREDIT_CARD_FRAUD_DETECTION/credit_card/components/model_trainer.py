import os
import sys
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf import ModelFactory

from credit_card.exception import CreditCardException
from credit_card.logger import logging
from credit_card.utils.main_utils import load_numpy_array_data, load_object, save_object
from credit_card.entity.config_entity import ModelTrainerConfig
from credit_card.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from credit_card.entity.estimator import CreditCardModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        try:
            logging.info("Loading neuro_mf ModelFactory using parameters configurations path matrix.")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            logging.info("Evaluating optimal model performance vectors metrics structures.")
            y_pred = model_obj.predict(x_test)
            
            f1 = f1_score(y_test, y_pred)  
            precision = precision_score(y_test, y_pred)  
            recall = recall_score(y_test, y_pred)
            
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            return best_model_detail, metric_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Triggered initiate_model_trainer sequence processing framework loops.")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("Validation Failure: Extracted optimal score lacks standard base limit accuracy.")
                raise Exception("No best model found with score more than base score")

            # Packing final assembly object state models modules pipelines
            credit_card_model = CreditCardModel(preprocessing_object=preprocessing_obj, trained_model_object=best_model_detail.best_model)
            
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, credit_card_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Successfully generated model trainer evaluation artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CreditCardException(e, sys) from e