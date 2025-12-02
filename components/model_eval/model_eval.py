import argparse
import numpy as np
import torch
import json
import mlflow
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import sys
import joblib

def main(args):
    """
    Evaluate trained Iris classifier model
    """
    print("Starting model evaluation...")
    
    # Start MLflow run
    mlflow.start_run()

    # Get model architecture
    spec_archi = importlib.util.spec_from_file_location("iris_architecture", args.archi)
    archi_module = importlib.util.module_from_spec(spec_archi)
    sys.modules["iris_architecture"] = archi_module
    spec_archi.loader.exec_module(archi_module)
    IrisArchitecture = archi_module.IrisArchitecture
    
    # Load test data (NumPy arrays)
    print(f"Loading test data from: {args.xy_test}")
    X_test = np.load(Path(args.xy_test) / "X.npy")
    y_test = np.load(Path(args.xy_test) / "y.npy")
    
    print(f"Test set: {len(y_test)} samples, {X_test.shape[1]} features")

    # Load scaler
    scaler_file = Path(args.scaler) / "scaler.pkl"
    scaler = joblib.load(scaler_file)
    print(f"Scaler loaded from: {scaler_file}")
    
    # Convert to PyTorch tensors
    X_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_scaled)

    # Load trained model
    print(f"Loading model from: {args.model}")
    model_path = Path(args.model) / "model.pth"
    model = IrisArchitecture()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    print("Model loaded successfully")
    print(f"Model architecture:\n{model}")
    
    # Make predictions
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix
    metrics_path = Path(args.metrics)
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    cm_path = metrics_path / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Log confusion matrix to MLflow
    mlflow.log_artifact(str(cm_path))
    
    # Save metrics as JSON output
    metrics_dict = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }
    
    metrics_file = metrics_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Metrics saved to: {metrics_file}")
    
    mlflow.end_run()
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--scaler", type=str, help="Path to the scaler")
    parser.add_argument("--archi", type=str, help="Path to model architecture file")
    parser.add_argument("--xy_test", type=str, help="Path to test data")
    parser.add_argument("--metrics", type=str, help="Output metrics path")
    
    args = parser.parse_args()
    main(args)