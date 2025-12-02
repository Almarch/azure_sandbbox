from mlflow.pyfunc import PythonModel
import torch
import joblib
import json
import importlib.util
import sys

class IrisInference(PythonModel):
    
    def load_context(self, context):
        """
        This method is called when the model is loaded. 
        It loads the PyTorch model architecture, the trained weights, the scaler, 
        and the label mapping from the MLflow artifacts.
        """

        # 1. Load the PyTorch Model Architecture
        spec_arch = importlib.util.spec_from_file_location("iris_arch", context.artifacts["archi"])
        arch_module = importlib.util.module_from_spec(spec_arch)
        sys.modules["iris_arch"] = arch_module
        spec_arch.loader.exec_module(arch_module)
        IrisArchitecture = arch_module.IrisArchitecture

        # 2. Instantiate the architecture and load the trained weights
        self.model = IrisArchitecture() 
        self.model.load_state_dict(torch.load(context.artifacts["model"], weights_only=True))
        self.model.eval()

        # 3. Load preprocessing and postprocessing artifacts
        self.scaler = joblib.load(context.artifacts["scaler"])
        mapping = json.load(open(context.artifacts["mapping"], 'r'))

        self.features = mapping["features"]
        self.id_to_labels = {int(v): k for k, v in mapping["labels"].items()}

    def predict(self, context, data):
        """
        This method is called during inference.
        It applies scaling before inference and maps the output IDs to string labels.
        
        Args:
            data (pd.DataFrame): Input data from the request.
        """
        
        # 1. Preprocessing: Scaling (Scaler)
        # Ensure input data only contains the features used for training
        X = data[self.features].values
        X_scaled = self.scaler.transform(X)
        
        # 2. Conversion to a PyTorch Tensor
        input_tensor = torch.from_numpy(X_scaled).float()
        
        # 3. Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            # Get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
        
        # 4. Post-processing: Mapping the predicted label (ID) to the name (string)
        predictions = [self.id_to_labels[p.item()] for p in predicted]
        
        return predictions