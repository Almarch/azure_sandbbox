import argparse
import mlflow
from pathlib import Path
import importlib.util
import sys

def main(args):
    """
    Save the model and all its dependencies as artefacts
    """
    print("Saving the model...")
    
    # Start MLflow run
    mlflow.start_run()

    # Get model architecture
    spec_archi = importlib.util.spec_from_file_location("iris_architecture", args.archi)
    archi_module = importlib.util.module_from_spec(spec_archi)
    sys.modules["iris_architecture"] = archi_module
    spec_archi.loader.exec_module(archi_module)

    # Get inference class
    spec_infer = importlib.util.spec_from_file_location("iris_inference", args.infer)
    infer_module = importlib.util.module_from_spec(spec_infer)
    sys.modules["iris_inference"] = infer_module
    spec_infer.loader.exec_module(infer_module)
    IrisInference = infer_module.IrisInference

    artifacts_dict = {
        "model": str(Path(args.model) / "model.pth"),
        "scaler": str(Path(args.scaler) / "scaler.pkl"),
        "mapping": str(Path(args.mapping) / "mapping.json"),
        "archi": args.archi,
    }

    mlflow.pyfunc.log_model(
        python_model=IrisInference(),
        artifact_path="model",
        registered_model_name="iris-model-ready",
        artifacts=artifacts_dict,
        code_paths=[
            str(args.infer),
            str(args.archi)
        ],
    )

    mlflow.end_run()
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archi", type=str, help="Path to model architecture file")
    parser.add_argument("--model", type=str, help="Output model path")
    parser.add_argument("--scaler", type=str, help="Output scaler path")
    parser.add_argument("--mapping", type=str, default=None, help="Input mapping path")
    parser.add_argument("--infer", type=str, default=None, help="Infer class for MLFlow model artifact")
    
    args = parser.parse_args()
    main(args)