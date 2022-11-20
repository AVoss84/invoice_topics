# Invoice topic modeling project

Python package *claims_topics* for invoice data topic exploration

## Package structure

```
├── environment.yml
├── README.md
├── requirements.txt                   # in case no conda being used
└── src
    ├── claims_topics
    │   ├── config
    │   │   ├── config.py
    │   │   ├── global_config.py
    │   │   ├── __init__.py
    │   │   ├── input_output.yaml
    │   │   ├── preproc_txt.yaml
    │   │   └── stopwords.txt
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
    │   │   └── README.md
    │   └── utils
    │       ├── __init__.py
    │       └── utils.py
    ├── __init__.py
    ├── notebooks
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

*Optional: if you wish to use JupyterHub on DP*
```bash
python -m ipykernel install --user --name invoice_clustering
```

Install your package
```bash
python -m spacy download de_core_news_lg      # install large Glove engl. word embeddings
pip install -e src
``` 
