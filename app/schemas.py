from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class Text(BaseModel):
    text: str = Query(
        None, min_length=1
    )  # string that defaults to None and must have a minimum length of 1 character.


class PredictPayload(BaseModel):
    texts: List[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    {"text": "Real Estate market is on the decline."},
                    {"text": "Breaking news : Elon Musk is solding Twitter for 1$ !!!"},
                ]
            }
        }
