from pathlib import Path

import mlflow
import pytest

from config import config
from src import main


def delete_experiment(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)


@pytest.fixture(scope="function")
def args_fp():
    return Path(config.BASE_DIR, "tests", "src", "test_args.json")


@pytest.fixture(scope="function")
def run_id():
    return open(Path(config.CONFIG_DIR, "run_id.txt")).read()


def test_load_artifacts(run_id):
    artifacts = main.load_artifacts(run_id=run_id)
    assert len(artifacts)


def test_elt_data():  # Only check if code for ELT is OK ! Not for Data Testing
    main.elt_data()


@pytest.mark.training
def test_train_model(args_fp):  # Only check if code for training is OK ! Not for Model Testing
    experiment_name = "test_experiment"
    run_name = "test_run"

    main.train_model(
        args_fp=args_fp, experiment_name=experiment_name, run_name=run_name, test_run=True
    )

    delete_experiment(experiment_name=experiment_name)


@pytest.mark.training
def test_optimize(args_fp):
    study_name = "test_study"
    num_trials = 1

    main.optimize(args_fp=args_fp, study_name=study_name, num_trials=num_trials)


def test_load_artifacts(run_id):
    artifacts = main.load_artifacts(run_id=run_id)
    assert len(artifacts)


def test_predict_tag(run_id):
    text = "Elon Musk is selling dragon flame throwers as a new product market"
    main.predict_tag(text=text, run_id=run_id)
