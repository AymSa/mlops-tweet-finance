# MLOps - Tweet Classification for financial application

The goal of this project is to implement MLOps practices to develop, deploy and maintain production machine learning applications for our use case applied to finance.

## Context

Text classification and sentiment analysis are natural language processing subjects that can be used to aid in the analysis and prediction of the financial market. These methods allow for the processing of large amounts of textual data from various sources such as news, financial reports, social media, etc.

By automatically classifying this textual data into different categories, such as industry sectors or types of companies, it becomes possible to identify market trends and investment opportunities. Similarly, sentiment analysis can evaluate the opinions of investors and analysts towards a company or sector, which can have an impact on investment decisions.

Overall, text classification and sentiment analysis provide valuable tools for the analysis and prediction of the financial market by enabling the quick and efficient processing of textual data and providing additional insights to traditional quantitative data.

## Installation

```
source .venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
python3 -m pip install -e ".[docs]" [Only for docs]
```

## Make Documentation 

```
python3 -m mkdocs new . (if not initialized)
python3 -m mkdocs serve
```


<hr>
<!-- Citation -->
Heavily inspired from:

```bibtex
@article{madewithml,
    author       = {Goku Mohandas},
    title        = {Made With ML},
    howpublished = {\url{https://madewithml.com/}},
    year         = {2022}
}

@book{mlsystems,
    author       = {Chip Huyen},
    title        = {Designing Machine Learning Systems},
    publisher    = {O’Reilly},
    year         = {2022}
}

@book{mlops,
    author       = {Noah Gift and Alfredo Deza},
    title        = {Practical MLOps : Operationalizing Machine Learning Models},
    publisher    = {O’Reilly},
    year         = {2021}
}
```
