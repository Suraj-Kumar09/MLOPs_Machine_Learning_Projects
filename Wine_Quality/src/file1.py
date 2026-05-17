import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 1. Tracking URI set karein (Server se sync karne ke liye)
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Dataset load karein
wine = load_wine()
X = wine.data
y = wine.target

# 3. Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# 4. Hyperparameters define karein
max_depth = 10
n_estimators = 5

# 5. Experiment name set karein
mlflow.set_experiment('Suraj-Exp1')

# 6. Date aur Time ke saath Unique Run Name banayein
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"RF_Run_{current_time}"

# 7. MLflow Run start karein
with mlflow.start_run(run_name=run_name):
    
    # Model train karein
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions aur Accuracy
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Metrics aur Params log karein
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # 8. Confusion Matrix banayein
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=wine.target_names, 
                yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix\n(Accuracy: {accuracy:.2f})')

    # Plot save karein
    plot_path = "Confusion_Matrix.png"
    plt.savefig(plot_path)
    plt.close() # Memory free karne ke liye plot close karein

    # 9. Artifacts log karein (Image aur Code file)
    mlflow.log_artifact(plot_path)
    mlflow.log_artifact(__file__)

    # 10. Tags set karein
    mlflow.set_tags({
        "Author": "Suraj",
        "Project": "Wine Quality Prediction",
        "Model_Type": "Random Forest"
    })

    # 11. Model ko log karein (Future use ke liye)
    mlflow.sklearn.log_model(rf, 'random_forest_model')

    print(f"Successfully tracked run: {run_name}")
    print(f"Accuracy: {accuracy}")