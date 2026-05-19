import os
from datetime import date, datetime


# MONGODB CONNECTION CONSTANTS
DATABASE_NAME = "creditcardproject"   # DB_NAME ko DATABASE_NAME kar diya taaki connection file crash na ho
COLLECTION_NAME = "credit_data" 
MONGODB_URL_KEY = "MONGODB_URL"       # Yeh exact wahi naam hai jo aapne terminal me export kiya hai


# PIPELINE & ARTIFACT CONSTANTS
PIPELINE_NAME: str = "creditcard"   
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
FILE_NAME: str = "credit_data.csv"
MODEL_FILE_NAME = "model.pkl"



# 4. 🎯 DATA VALIDATION CENTRAL CONSTANTS (Naye project me badlega)
TARGET_COLUMN = "Class"  # 🛑 KYA BADLEGA: Naye dataset ki dependent/target column ka naam (Jaise: "Class")
CURRENT_YEAR = date.today().year  # 🟢 SAME RAHEGA: Dynamic tarike se current year nikalega feature engineering ke liye
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"  # 🟢 SAME RAHEGA: Data transform karne wale pickle object ka naam
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")  # 🟢 SAME RAHEGA: Config folder ke andar schema rule file ka path


# ESKE BAD ENTITY ME JAO
# entity/config_entity.py 

"""
Data ingestion related constants start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "credit_data"  
DATA_INGESTION_DIR_NAME: str = "data_ingestion"  
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"  
DATA_INGESTION_INGESTED_DIR: str = "ingested"  
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2



"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"  # 🟢 SAME RAHEGA: Artifact folder ke andar data_validation folder banega
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"  # 🟢 SAME RAHEGA: Data Validation ke andar drift report ka sub-folder banega
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"  # 🟢 SAME RAHEGA: Evidently AI se bnegi jo report uska final naam



# Artifact folder ke andar data_transformation folder create hoga jisme transformation folder create hoga jisme tranformed aur transformed_object file bange 
# Ab entity/config_entity jana hai aur code likha hai uske bad entity/artifact_entity file me jana hai aur code likhna hai  folder me jana hai aur 
"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"





# ==============================================================================
# MODEL TRAINER PARAMETERS CONSTANTS
# ==============================================================================
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
