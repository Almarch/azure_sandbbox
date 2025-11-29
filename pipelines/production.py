from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from pipelines.evaluation import iris_evaluation_pipeline

components_dir = "./components/"
data_prepare = load_component(source=f"{components_dir}/data_prepare/data_prepare.yaml")
model_train = load_component(source=f"{components_dir}/model_train/model_train.yaml")

@pipeline()
def iris_production_pipeline(
    data: Input,
    compute: str,
    train_percent: float = 70,
    lr: float = 1e-3,
    epochs: int = 10,
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
        lr = lr,
        epochs = epochs,
        seed_train = seed_train,
        batch_size = batch_size,
        log_model = True,
    )
    model.compute = compute

    return {
        "model": model.outputs.model,
        "metrics": metrics.outputs.metrics,
        "scaler": xy.outputs.scaler,
        "mapping": xy.outputs.mapping,
    }

