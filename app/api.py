from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Request

from app.schemas import PredictPayload
from config import config
from config.config import logger
from src import main, predict

app = FastAPI(title="FinBirdAI", description="Classify financial tweets.", version="0.1")


def construct_response(f):
    """
    Custom decorator for constructing our API response.


    Args:
        f (function): function to decorate

    Returns:
        function: decorator
    """

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        if "data" in results.keys():
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """
    Health check.

    Args:
        request (Request): request from the user.

    Returns:
        Dict: Health check response.
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.on_event("startup")
def load_artifacts():
    """Load artificats for inference."""
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    logger.debug(f"run id : {run_id}")
    logger.info("Ready for inference ✅")


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """
    Get the performance metrics.

    Args:
        request (Request): request from the user.
        filter (str, optional): filter to specify the type of metric. Defaults to None (all metrics returned).

    Returns:
        Dict: response with specified metrics.
    """
    performance = artifacts["performance"]
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@app.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """
    Get all arguments used for the run.

    Args:
        request (Request): request from the user.

    Returns:
        Dict: response with all arguments used
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": vars(artifacts["args"]),
        },
    }
    return response


@app.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> Dict:
    """
    Get the specified argument used for the run.

    Args:
        request (Request): request from the user.

    Returns:
        Dict: response with the specified argument.
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: vars(artifacts["args"]).get(arg, ""),
        },
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictPayload) -> Dict:
    """
    Predict tags for a list of texts.

    Args:
        request (Request): request from the user
        payload (PredictPayload): payload of texts to classify.

    Returns:
        Dict: response with predicted texts labels
    """
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response
