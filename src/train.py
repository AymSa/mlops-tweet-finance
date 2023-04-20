from imblearn.over_sampling import RandomOverSampler
import json
import numpy as np
import pandas as pd
import mlflow
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from src import data, utils, predict, evaluate


def objective(args, trial):
    """Objective function for optimization trials."""
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

    # Train & evaluate
    artifacts = train(args=args, trial=trial)

    # Set additional attributes
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))
    trial.set_user_attr("precision", performance["precision"])
    trial.set_user_attr("recall", performance["recall"])
    trial.set_user_attr("f1", performance["f1"])

    return performance["f1"]


def train(args, df, trial=None):
    # Setup
    utils.set_seeds()
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.subset]  # None = all samples
    df = data.preprocess(df, lower=args.lower, stem=args.stem, min_freq=args.min_freq)
    label_encoder = data.LabelEncoder().fit(df.label)
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(
        X=df.text.to_numpy(), y=label_encoder.encode(df.label)
    )
    test_df = pd.DataFrame({"text": X_test, "label": label_encoder.decode(y_test)})

    # Tf-idf
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer, ngram_range=(2, args.ngram_max_range)
    )  # char n-grams
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # Oversample
    oversample = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # Model
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        max_iter=1,
        learning_rate="constant",
        eta0=args.learning_rate,
        power_t=args.power_t,
        warm_start=True,
    )  # max_iter=1 is 1 epoch.

    # Training
    for epoch in range(args.num_epochs):
        model.fit(X_over, y_over)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))
        if not epoch % 10:
            print(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}"
            )
        # Log
        if not trial:
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )

        # Pruning
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Threshold
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    args.threshold = np.quantile(
        [y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25
    )  # Q1

    # Evaluate
    other_index = label_encoder.tag_to_idx["other"]
    y_prob = model.predict_proba(X_test)
    y_pred = predict.custom_predict(
        y_prob=y_prob, threshold=args.threshold, index=other_index
    )

    performance = evaluate.get_metrics(
        y_true=y_test, y_pred=y_pred, classes=label_encoder.tags, df=test_df
    )

    decoded_pred = label_encoder.decode(y_pred)
    decoded_label = label_encoder.decode(model.classes_)
    confusion = evaluate.get_confusion(
        y_true=test_df.label, y_pred=decoded_pred, classes=decoded_label
    )

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
        "confusion": confusion,
    }
