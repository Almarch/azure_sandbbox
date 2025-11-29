import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def main(args):
    """
    Split prepared Iris data into train and test sets
    """
    print(f"Loading prepared data from: {args.xy}")
    
    # Load prepared data (NumPy arrays)
    X = np.load(Path(args.xy) / "X.npy")
    y = np.load(Path(args.xy) / "y.npy")
    
    print(f"Data loaded: {len(y)} samples, {X.shape[1]} features")
    
    # Split into train and test
    train_size = args.train_percent / 100.0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=args.seed_split,
        stratify=y  # Stratify to maintain class balance
    )
    
    print(f"\nSplit with seed={args.seed_split}")
    print(f"Train set: {len(y_train)} samples ({args.train_percent}%)")
    print(f"Test set: {len(y_test)} samples ({100 - args.train_percent}%)")
    print(f"Train target distribution: {np.bincount(y_train)}")
    print(f"Test target distribution: {np.bincount(y_test)}")
    
    # Save train set
    train_path = Path(args.xy_train)
    train_path.mkdir(parents=True, exist_ok=True)
    
    np.save(train_path / "X.npy", X_train)
    np.save(train_path / "y.npy", y_train)
    print(f"\nTrain data saved to: {train_path}")
    
    # Save test set
    test_path = Path(args.xy_test)
    test_path.mkdir(parents=True, exist_ok=True)
    
    np.save(test_path / "X.npy", X_test)
    np.save(test_path / "y.npy", y_test)
    print(f"Test data saved to: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xy", type=str, help="Input prepared data path")
    parser.add_argument("--train_percent", type=float, default=70, help="Train set percentage")
    parser.add_argument("--seed_split", type=int, default=42, help="Random seed for split")
    parser.add_argument("--xy_train", type=str, help="Output train set path")
    parser.add_argument("--xy_test", type=str, help="Output test set path")
    
    args = parser.parse_args()
    main(args)