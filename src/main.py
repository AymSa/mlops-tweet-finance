import json
import shutil
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from src import data, predict, train, utils

warnings.filterwarnings("ignore")


def elt_data():
    """
    Extract, load and transform our data assets.
    """
    # Save raw data locally
    shutil.copyfile(config.TAGS_PATH, Path(config.DATA_DIR, "tags.json"))
    shutil.copyfile(config.TWEETS_PATH, Path(config.DATA_DIR, "tweets.csv"))

    # Extract + Load
    tweets = pd.read_csv(config.TWEETS_PATH)

    # Transform
    dict_tags = data.get_idx_tag(Path(config.DATA_DIR, "tags.json"))
    tweets.label = data.idx_to_tag(tweets.label, dict_tags)

    tweets = tweets[tweets.label.notnull()]  # drop rows w/ no tag
    tweets.to_csv(Path(config.DATA_DIR, "labeled_tweets.csv"), index=False)

    config.logger.info("ELT ✅")


def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "logreg_sgd",
    test_run: bool = False,
) -> None:
    """
    Train a model given arguments.

    Args :
        args_fp (str) : location of args config file.
        experiment_name (str): name of experiment.
        run_name (str): name of specific run in experiment.
        test_run (bool, optional): If True, artifacts will not be saved. Defaults to False.
    """
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
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.RESULT_DIR, "performance.json"))
        plt.imsave(Path(config.RESULT_DIR, "confusion.jpg"), artifacts["confusion"])

    config.logger.info(json.dumps(performance, indent=2))
    config.logger.info("Training ✅")


def optimize(
    args_fp: str = "config/args.json",
    study_name: str = "optimization",
    num_trials: int = 20,
) -> None:
    """
    Optimize hyperparameters.

    Args :
        args_fp (str) : location of args config file.
        study_name (str): name of optimization study.
        num_trials (int): number of trials to run in study.

    Returns :
        None
    """
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_tweets.csv"))

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
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

    config.logger.info(json.dumps(args, indent=2))
    config.logger.info("Optimize ✅")


def predict_tag(text: str, run_id: str = None) -> List[Dict]:
    """
    Predict tag for text.

    Args :
        text (str): Text to classify
        run_id (str, optional) : run id to load artifacts for prediction. Defaults to None.

    Returns :
        List[Dict]: Text with associated predicted label
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)

    config.logger.info(json.dumps(prediction, indent=2))
    config.logger.info("Predict ✅")

    return prediction


def load_artifacts(run_id: str) -> Dict:
    """
    Load artifacts for a given run_id.

    Args:
        run_id : run id to load artifacts for prediction.

    Returns:
        Dict: artifacts from the specified run.
    """
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(file_path=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    config.logger.info("Load artificats ✅")

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }
