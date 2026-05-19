from dataclasses import dataclass  # 🟢 SAME RAHEGA: Static data objects ko fast mapping entity dene ke liye use hota hai

@dataclass
class DataIngestionArtifact:
    """
    Description : Data Ingestion component ka final output, jo train aur test 
                  CSV files ke absolute target paths ko store karta hai.
    """
    # 🟢 SAME RAHEGA: Ingested folder ke andar bne hue 'train.csv' ka absolute local rasta
    trained_file_path: str 
    # 🟢 SAME RAHEGA: Ingested folder ke andar bne hue 'test.csv' ka absolute local rasta
    test_file_path: str 


# ==============================================================================
# 🎯 NEW DATA VALIDATION ARTIFACT ENTITY
# ==============================================================================

@dataclass
class DataValidationArtifact:
    """
    Description : Data Validation component ka final output container.
                  Yeh pipeline ko batata hai ki validation success hui ya nahi 
                  aur report file kahan save hui hai.
    """
    # 🟢 SAME RAHEGA: True/False state hold karega. Agar data validation rules pass hue 
    # toh True, agar data columns mismatch ya missing hue toh False.
    validation_status: bool
    
    # 🟢 SAME RAHEGA: Pipeline console me accurate message capture karne ke liye 
    # (Jaise: "Drift detected" ya "Drift not detected" ya "Columns missing")
    message: str
    
    # 🟢 SAME RAHEGA: Evidently AI engine se jo output 'report.yaml' file disk par generate 
    # hui hai, uske absolute target raste (path) ko lock karke state me capture karega.
    drift_report_file_path: str




# Ye file path return karega like niche diy gye files name ka path return karega
#  Ab component/data_transformation file me jao aur code add karo

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str



# ==============================================================================
# EVALUATION METRICS & MODEL ARTIFACT ENTITIES
# ==============================================================================
@dataclass
class ClassificationMetricArtifact:
    f1_score: float           
    precision_score: float    
    recall_score: float       

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str 
    metric_artifact: ClassificationMetricArtifact