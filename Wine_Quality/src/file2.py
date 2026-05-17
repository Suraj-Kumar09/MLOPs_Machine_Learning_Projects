import dagshub
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# 1. DagsHub Initialization (Yeh authentication handle karega)
dagshub.init(repo_owner='Suraj-Kumar09', repo_name='MLOPs_Machine_Learning_Projects', mlflow=True)

# 2. Tracking URI set karein
mlflow.set_tracking_uri("https://dagshub.com/Suraj-Kumar09/MLOPs_Machine_Learning_Projects.mlflow")

# Data Loading
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.1, random_state=42)

# Experiment Name
mlflow.set_experiment('DagsHub-Wine-Quality')

# Unique Run Name with Timestamp
run_name = f"DagsHub_Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

with mlflow.start_run(run_name=run_name):
    # Model Training
    max_depth = 15
    n_estimators = 15
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    # Accuracy Calculation
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Logging to DagsHub
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_metric('accuracy', acc)
    
    # Model Logging (Optional but recommended)
    mlflow.sklearn.log_model(rf, "random_forest_model")
    
    print(f"Run Successful! Accuracy: {acc}")
    print("Ab DagsHub par 'Experiments' tab check karein.")