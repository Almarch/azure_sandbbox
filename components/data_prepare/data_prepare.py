import argparse
import pandas as pd
import numpy as np
import mltable
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def main(args):
    """
    Prepare Iris data: load MLTable, convert to X, y format, normalize, and save as NumPy arrays
    """
    print("Loading data from MLTable...")
    
    # Load MLTable
    tbl = mltable.load(args.data)
    df = tbl.to_pandas_dataframe()
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract features (X) and target (y)
    feature_columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    X = df[feature_columns].values
    
    # Encode species to numeric
    species_mapping = {
        "setosa": 0,
        "versicolor": 1,
        "virginica": 2
    }
    y = df["Species"].map(species_mapping).values
    
    # Create reverse mapping for inference
    id_to_species = {v: k for k, v in species_mapping.items()}
    
    print(f"\nOriginal feature statistics:")
    print(pd.DataFrame(X, columns=feature_columns).describe())
    
    # Fit scaler and transform features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nScaled feature statistics:")
    print(pd.DataFrame(X_scaled, columns=feature_columns).describe())
    
    print(f"\nPrepared data: {X_scaled.shape[0]} samples, {len(feature_columns)} features")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Save prepared data as NumPy arrays
    xy_path = Path(args.xy)
    xy_path.mkdir(parents=True, exist_ok=True)
    
    X_file = xy_path / "X.npy"
    y_file = xy_path / "y.npy"
    
    np.save(X_file, X_scaled)
    np.save(y_file, y)
    
    print(f"X saved to: {X_file}")
    print(f"y saved to: {y_file}")
    
    # Save scaler
    scaler_path = Path(args.scaler)
    scaler_path.mkdir(parents=True, exist_ok=True)
    
    scaler_file = scaler_path / "scaler.pkl"
    joblib.dump(scaler, scaler_file)
    print(f"\nScaler saved to: {scaler_file}")
    
    # Print scaler parameters
    print(f"\nScaler parameters:")
    for feature, mean, std in zip(feature_columns, scaler.mean_, scaler.scale_):
        print(f"  {feature}: mean={mean:.4f}, std={std:.4f}")
    
    # Save label mapping
    mapping_path = Path(args.mapping)
    mapping_path.mkdir(parents=True, exist_ok=True)
    
    mapping_file = mapping_path / "label_mapping.json"
    mapping_data = {
        "species_to_id": species_mapping,
        "id_to_species": id_to_species,
        "features": feature_columns
    }
    
    with open(mapping_file, "w") as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"Label mapping saved to: {mapping_file}")
    print(f"\nMapping: {species_mapping}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Input MLTable path")
    parser.add_argument("--xy", type=str, help="Output prepared data path")
    parser.add_argument("--scaler", type=str, help="Output scaler path")
    parser.add_argument("--mapping", type=str, help="Output label mapping path")
    
    args = parser.parse_args()
    main(args)