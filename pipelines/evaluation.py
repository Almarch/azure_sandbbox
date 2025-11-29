from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

data_split = ml_client.components.get(name="data_split", label="latest")
model_train = ml_client.components.get(name="model_train", label="latest")
model_eval = ml_client.components.get(name="model_eval", label="latest")

@pipeline()
def iris_evaluation_pipeline(
    xy: Input,
    compute: str,
    train_percent: float = 70,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 16,
    seed_split: int = 42,
    seed_train: int = 42,
):
    """Evaluate an iris clasifier model"""

    split = data_split(
        xy = xy,
        train_percent = train_percent,
        seed_split = seed_split,
    )
    split.compute = compute

    model = model_train(
        xy_train = split.outputs.xy_train,
        lr = lr,
        epochs = epochs,
        seed_train = seed_train,
        batch_size = batch_size,
        log_model = False,
    )
    model.compute = compute

    metrics = model_eval(
        model = model.outputs.model,
        xy_test = split.outputs.xy_test
    )
    metrics.compute = compute

    return {
        "metrics": metrics.outputs.metrics,
    }


