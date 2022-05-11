
 ![mlfow](https://github.com/TatyanaMardvilko/forest_cover/blob/master/mlFlow.png)
 
Homework for RS School Machine Learning course.
9_evaluation_selection

This task uses [Forest Cover](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv) dataset.

## Usage
This package allows you to train model for  predict an integer classification for the forest cover type.
1. Clone this repository to your machine.
2. Download [Forest Cover](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Only train use CLI. Other script such as experiments.py and Nested_K-Fold.py
I run from PyCharm.
```sh
open Edit Configuration in upper part of window PyCharm
add Name
add Script Path
run
```
all result you see in PyCharm console
8. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```



## Development
The code in this repository must be formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
Install black. I use
```
pip install git+https://github.com/psf/black
```
To start black
```
black {path of project}
```
Install  flake8. 
```
pip install flake8
```
To start flake8
```
flake8  {path of project}
```

