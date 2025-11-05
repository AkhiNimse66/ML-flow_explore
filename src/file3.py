
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

#  Load
iris = datasets.load_iris()
X = iris.data + np.random.normal(0, 0.2, iris.data.shape)  # add small noise
y = iris.target

# 50% split makes accuracy more varied
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

#  models to compare 
models = {
    "LogisticRegression": LogisticRegression(max_iter=300),
    "RandomForest": RandomForestClassifier(n_estimators=120, random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
}

#  MLflow experiment 
mlflow.set_experiment("Iris_Model_Comparison_Enhanced")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        #  Compute metrics 
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")

        # Log metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        #  Confusion Matrix Plot 
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{model_name} Confusion Matrix")

        # Save plot as temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            mlflow.log_artifact(tmpfile.name, artifact_path="confusion_matrix")

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name}: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")

print("\n Models logged with richer metrics and plots.")
