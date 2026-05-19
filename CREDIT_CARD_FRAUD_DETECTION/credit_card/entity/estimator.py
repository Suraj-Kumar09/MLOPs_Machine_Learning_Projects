import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from credit_card.exception import CreditCardException
from credit_card.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.Normal: int = 0
        self.Fraud: int = 1
        
    def _asdict(self):
        return self.__dict__
        
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

class CreditCardModel:      
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocessor
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        logging.info("Entered predict method of CreditCardModel class execution loops.")
        try:
            logging.info("Transforming raw data using preprocessor registry state.")
            transformed_feature = self.preprocessing_object.transform(dataframe)
            
            logging.info("Running parallel predictions matrices arrays.")
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise CreditCardException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"