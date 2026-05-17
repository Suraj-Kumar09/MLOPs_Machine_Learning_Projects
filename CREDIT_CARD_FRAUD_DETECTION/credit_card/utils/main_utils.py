import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame

from credit_card.exception import CreditCardException
from credit_card.logger import logging


#  yaml file read krne ke liye (jo config/model.yaml aur config/schema.yaml ) file ko read krne ke liye

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CreditCardException(e, sys) from e
    

#  yaml file write krne ke liye (jo config/model.yaml aur config/schema.yaml ) file ko read krne ke liye


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CreditCardException(e, sys) from e
    


#  Binary object ya .pkl object ko load krne ke liye (jo model.pkl file ko load krne ke liye use hoga)

def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise CreditCardException(e, sys) from e
    

#  Binary object ya .pkl object ko save krne ke liye (jo model.pkl file ko save krne ke liye use hoga) ya numpy array ko save krne ke liy

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CreditCardException(e, sys) from e
    


#  Numpy array ko load krne ke liye (jo model.pkl file ko load krne ke liye use hoga) ya numpy array ko load krne ke liy

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CreditCardException(e, sys) from e



# Binary object ya .pkl object ko save krne ke liye (jo model.pkl file ko save krne ke liye use hoga) ya numpy array ko save krne ke liy

def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise CreditCardException(e, sys) from e


#  pandas DataFrame se columns drop krne ke liye (jo data transformation ke time use hoga)

def drop_columns(df: DataFrame, cols: list)-> DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise CreditCardException(e, sys) from e