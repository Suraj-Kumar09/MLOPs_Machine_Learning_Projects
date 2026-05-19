import os  # 🟢 SAME RAHEGA: Artifacts folders aur report file ke absolute paths jodhne ke liye use hota hai
from dataclasses import dataclass  # 🟢 SAME RAHEGA: Bina boilerplate code ke clean configuration classes banane ke liye use hota hai
from datetime import datetime  # 🟢 SAME RAHEGA: Har execution run ko ek unique folder dene ke liye use hota hai

# 🛑 KYA BADLEGA: Naye project (Credit Card) ke constants file se sab kuch import karenge
from credit_card.constants import * # 🟢 SAME RAHEGA: Unique timestamp string banayega taaki har baar run karne par purana data mix na ho
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    """
    Description : Base configuration class jo pure pipeline ke liye main artifact folder set karti hai.
    """
    # 🟢 SAME RAHEGA: Pipeline ka naam constants se uthayega (Jaise: "creditcard")
    pipeline_name: str = PIPELINE_NAME
    # 🟢 SAME RAHEGA: Rasta banayega (artifact/05_19_2026_09_08_43)
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


# 🟢 SAME RAHEGA: Iska instance niche data validation config me use hoga base directory path nikalne ke liye
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    """
    Description : Data Ingestion ke saare paths jo humne pehle step me use kiye the.
    """
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


# ==============================================================================
# 🎯 NEW DATA VALIDATION CONFIGURATION ENTITY
# ==============================================================================

@dataclass
class DataValidationConfig:
    """
    Description : Yeh dataclass specifically data validation component ke liye saare output paths set karti hai.
    """
    # 🟢 SAME RAHEGA: Main artifact folder ke andar 'data_validation' ka sub-folder path generate karega
    # Output path: artifact/timestamp/data_validation
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    
    # 🟢 SAME RAHEGA: Data validation folder ke andar drift_report/report.yaml ka absolute path set karega
    # Jahan Evidently AI apni research ka safe output file dump karegi
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
    
    # 🟢 BUG FIX: Jo schema yaml file humne pichle step me bnai, uska dynamic global path configuration me bind kiya
    # Iska use component file direct schema validation ke liye karegi
    schema_file_path: str = SCHEMA_FILE_PATH




# Path hai jo data transformation ke bad hoga 
# Ab entity/artifact_entity me jao aur code add kro
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCSSING_OBJECT_FILE_NAME)
    


# ==============================================================================
# MODEL TRAINER CONFIG ENTITY
# ==============================================================================
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH