import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt
import importlib.util
import sys
import joblib
from sklearn.preprocessing import StandardScaler

def main(args):
    """
    Train Iris classifier with PyTorch
    """
    print("Starting model training...")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed_train)
    
    # Start MLflow run
    mlflow.start_run()

    # Get model architecture
    spec_archi = importlib.util.spec_from_file_location("iris_architecture", args.archi)
    archi_module = importlib.util.module_from_spec(spec_archi)
    sys.modules["iris_architecture"] = archi_module
    spec_archi.loader.exec_module(archi_module)
    IrisArchitecture = archi_module.IrisArchitecture

    # Log hyperparameters
    mlflow.log_param("learning_rate", args.lr)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("seed_train", args.seed_train)
    mlflow.log_param("architecture", args.archi)
    
    # Load training data (NumPy arrays)
    print(f"Loading training data from: {args.xy_train}")
    X_train = np.load(Path(args.xy_train) / "X.npy")
    y_train = np.load(Path(args.xy_train) / "y.npy")

    print(f"Training set: {len(y_train)} samples, {X_train.shape[1]} features")
    print(f"Target distribution:\n{np.bincount(y_train)}")
    
    # Fit scaler and transform features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = IrisArchitecture()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize tracking lists
    loss_history = []
    lr_history = []
    step_history = []
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs with batch_size={args.batch_size}...")
    model.train()
    
    global_step = 0  # Global batch counter
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store batch-level metrics for plotting
            loss_history.append(loss.item())
            lr_history.append(args.lr)
            step_history.append(global_step)
            
            global_step += 1
            
            # Track epoch metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        
        # Log epoch metrics to MLflow
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_accuracy", accuracy, step=epoch)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("\nTraining completed!")
    
    # Save model first (need model_path)
    model_path = Path(args.model)
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "model.pth"
    
    torch.save(model.state_dict(), model_file)
    print(f"Model (complete object) saved to: {model_file}")

    # Save scaler
    scaler_path = Path(args.scaler)
    scaler_path.mkdir(parents=True, exist_ok=True)
    
    scaler_file = scaler_path / "scaler.pkl"
    joblib.dump(scaler, scaler_file)
    print(f"\nScaler saved to: {scaler_file}")

    # Print scaler parameters
    print(f"\nScaler parameters:")
    for mean, std in zip(scaler.mean_, scaler.scale_):
        print(f"mean={mean:.4f}, std={std:.4f}")
    
    # Create training plot (batch-level granularity)
    _, ax1 = plt.subplots(figsize=(6, 3))
    ax1.set_xlabel('Training Step (Batch)', fontsize=12)
    ax1.set_ylabel('Loss', color='blue', fontsize=10)
    ax1.plot(step_history, loss_history, color='blue', linewidth=1, label='Loss', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='red', fontsize=10)
    ax2.plot(step_history, lr_history, color='red', linewidth=2, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plot_path = model_path / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(plot_path))
    
    print(f"Training curves saved to: {plot_path}")
    
    mlflow.end_run()
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xy_train", type=str, help="Path to training data")
    parser.add_argument("--archi", type=str, help="Path to model architecture file")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--seed_train", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, help="Output model path")
    parser.add_argument("--scaler", type=str, help="Output scaler path")
    
    args = parser.parse_args()
    main(args)