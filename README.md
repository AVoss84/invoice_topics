# Invoice topic modeling project

This is a blueprint of a generic end-to-end data science project, i.e. building a Python package along the usual steps: data preprocessing, model training, prediction, postprocessing, REST API construction (for real-time model serving) and containerization for final deployment as a microservice.

## Package structure

```
├── environment.yml
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── invoice_topics
    │   ├── config
    │   │   ├── config.py
    │   │   ├── global_config.py
    │   │   ├── __init__.py
    │   │   ├── input_output.yaml
    │   │   ├── preproc_txt.yaml
    │   │   └── stopwords.json
    │   ├── data
    │   ├── resources
    │   │   ├── __init__.py
    │   │   ├── postprocessor.py
    │   │   ├── predictor.py
    │   │   ├── preprocessor.py
    │   │   ├── README.md
    │   │   └── trainer.py
    │   ├── services
    │   │   ├── file.py
    │   │   ├── __init__.py
    │   │   ├── pipelines.py
    │   │   └── README.md
    │   └── utils
    │       ├── __init__.py
    │       └── utils.py
    ├── notebooks
    │   ├── topic_modeling.ipynb
    │   └── word2vec.ipynb
    └── setup.py
```

## Use Case description

**Business goal**: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 

**Business stakeholders**: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

**Input data description**: Iris data set

**Business impact KPI**: Faster STP (in hours/days)


## Package installation and application develoment

Create conda virtual environment with required packages 
```bash
conda env create -f environment.yml 
conda activate invoice_clustering
```

Install your package
```bash
pip install -e src
``` 
