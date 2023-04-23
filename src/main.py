import pandas as pd
from pathlib import Path
import warnings
import shutil
from argparse import Namespace
import json
import mlflow
from numpyencoder import NumpyEncoder
import optuna
from optuna.integration.mlflow import MLflowCallback
import tempfile
import joblib
import matplotlib.pyplot as plt

from config import config
from src import utils, data, train, predict


warnings.filterwarnings("ignore")


def elt_data():
    """Extract, load and transform our data assets."""
    # Extract + Load
    tweets = pd.read_csv(config.TWEETS_PATH)

    # Save raw data
    tweets.to_csv(Path(config.DATA_DIR, "tweets.csv"), index=False)
    shutil.copyfile(config.TAGS_PATH, Path(config.DATA_DIR, "tags.txt"))

    # Transform
    df = tweets
    dict_tags = data.get_idx_tag(Path(config.DATA_DIR, "tags.txt"))
    df.label = data.idx_to_tag(df.label, dict_tags)

    df = df[df.label.notnull()]  # drop rows w/ no tag
    df.to_csv(Path(config.DATA_DIR, "labeled_tweets.csv"), index=False)

    print("ELT ✅")


def train_model(args_fp, experiment_name, run_name):
    """Train a model given arguments."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_tweets.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    # Set experiment
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)

        # Log key metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(artifacts["performance"], Path(dp, "performance.json"))
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            plt.imsave(Path(dp, "confusion.jpg"), artifacts["confusion"])
            mlflow.log_artifacts(dp)

    # Save results
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.RESULT_DIR, "performance.json"))
    plt.imsave(Path(config.RESULT_DIR, "confusion.jpg"), artifacts["confusion"])

    print(json.dumps(performance, indent=2))
    print("Training ✅")


def optimize(args_fp, study_name, num_trials):
    """Optimize hyperparameters."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_tweets.csv"))

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name, direction="maximize", pruner=pruner
    )
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1"
    )
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    args = {**args.__dict__, **study.best_trial.params}
    utils.save_dict(d=args, filepath=args_fp, cls=NumpyEncoder)

    print("Optimize ✅")

def predict_tag(text, run_id=None):
    """Predict tag for text."""
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction
 
def load_artifacts(run_id):
    """Load artifacts for a given run_id."""
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(file_path=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance
    }
