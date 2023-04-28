# MLOps - Tweet Classification for financial application

The goal of this project is to implement MLOps practices to develop, deploy and maintain production machine learning applications for our use case applied to finance.

## Context

Text classification and sentiment analysis are natural language processing subjects that can be used to aid in the analysis and prediction of the financial market. These methods allow for the processing of large amounts of textual data from various sources such as news, financial reports, social media, etc.

By automatically classifying this textual data into different categories, such as industry sectors or types of companies, it becomes possible to identify market trends and investment opportunities. Similarly, sentiment analysis can evaluate the opinions of investors and analysts towards a company or sector, which can have an impact on investment decisions.

Overall, text classification and sentiment analysis provide valuable tools for the analysis and prediction of the financial market by enabling the quick and efficient processing of textual data and providing additional insights to traditional quantitative data.

## Installation

```
make venv
```

## Serving Documentation 

```
make doc
```

## View MLflow experiments 

```
make mlflow
```

## Serving REST API 

### Dev

```
make rest-dev
```

### Prod 

Enable parallelism and can deal with meaningful traffic. 

```
make rest-prod
```

You can get the documentation of the REST API at ```http://0.0.0.0:8000/docs```


## Testing 

### Code

We will use pytest to test ours functions, even those that interact with our data and models !!!

```
python -m pytest -m 'not training'
python -m pytest -m 'training'
```

### Data

We use the **great expectations** library to create expectations as to what our data should look like in a standardized way for data validity.

**All the following commands need to be done in the tests folder !!!**

To setup the folder for data test we run the command ```great_expectations init```.

The first step is then to establish our datasource which tells Great Expectations where our data lives with the command ```great_expectations datasource new```

Create expectations manually, interactively or automatically and save them as suites (a set of expectations for a particular data asset).

```great_expectations suite new```

For now only supports tsv, csv, and parquet file extensions. 

Create Checkpoints where a Suite of Expectations are applied to a specific data asset.

```great_expectations checkpoint new <CHECKPOINT_NAME>```


```great_expectations checkpoint run <CHECKPOINT_NAME>```

Great Expectations automatically generates documentation for our tests. It also stores information about validation runs and their results. 

```great_expectations docs build```

Note : The advantage of using a library such as great expectations in production is that we can :

- reduce redundant efforts for creating tests across data modalities

- automatically create testing checkpoints to execute as our dataset grows

- automatically generate documentation on expectations and report on runs

- easily connect with backend data sources

We will try these features later.

### Model

Behavioral testing is the process of  testing input data and expected outputs while treating the model as a black box.

The main tests are : 

- invariance test: Changes should not affect outputs.

- directional test : Change should affect outputs.

- minimum functionality test : Simple combination of inputs and expected outputs.

To run all tests (expect training one) run the command : ```make test```
##
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
