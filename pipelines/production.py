from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

data_prepare = ml_client.components.get(name="data_prepare", label="latest")
model_train = ml_client.components.get(name="model_train", label="latest")
model_save = ml_client.components.get(name="model_save", label="latest")
iris_evaluation_pipeline = ml_client.components.get(name="iris_evaluation_pipeline", label="latest")

@pipeline()
def iris_production_pipeline(
    data: Input,
    archi: Input,
    infer: Input,
    compute: str,
    train_percent: float = 70,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 16,
    seed_split: int = 42,
    seed_train: int = 42,
):
    """Full production pipeline: validate recipe then train on 100%"""

    xy = data_prepare(
        data = data
    )
    xy.compute = compute

    metrics = iris_evaluation_pipeline(
        xy = xy.outputs.xy,
        archi = archi,
        compute = compute,
        train_percent = train_percent,
        lr = lr,
        epochs = epochs,
        seed_split = seed_split,
        seed_train = seed_train,
        batch_size = batch_size,
    )

    model = model_train(
        xy_train = xy.outputs.xy,
        archi = archi,
        lr = lr,
        epochs = epochs,
        seed_train = seed_train,
        batch_size = batch_size,
    )
    model.compute = compute

    save = model_save(
        model = model.outputs.model,
        scaler = model.outputs.scaler,
        archi = archi,
        infer = infer,
        mapping = xy.outputs.mapping,
    )
    save.compute = compute

    return {
        "metrics": metrics.outputs.metrics,
    }

