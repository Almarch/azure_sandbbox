from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

components_dir = "./components/"
data_split = load_component(source=f"{components_dir}/data_split/data_split.yaml")
model_train = load_component(source=f"{components_dir}/model_train/model_train.yaml")
model_eval = load_component(source=f"{components_dir}/model_eval/model_eval.yaml")

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


