# Personal Identifiable Information Data Detection

The following Personally Identifiable Information (PII) Data Detection project was based on [this kaggle competition](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data). The task was to detect students’ personal data in essays written by students, i.e. for an input batch of essays, we would return a csv file with tokens labelled as being a particular kind of PII. We extend the original task, which demanded a csv file with detected PIIs’ document id, token ids and labels, into a web app (`app.py`), which given a text returns the same text with detected PII labelled accordingly. A command line version of the app is also available (`cli_app.py`), which takes a .json file with a batch of essays and saves the detected PII to a .csv file, much like in the original competition.

## Table of Contents

- [How to run the project](#how-to-run-the-project)
- [Project Structure](#project-structure)

## How to run the project

1. Create a new python virtual environment, activate it
2. In the newly created venv, run `pip install -r requirements.txt`

- to run `app.py`, use `streamlit run app.py`
- to run `cli_app.py`, use `python cli_app.py`

## Project Structure

- `data/test/ten_essays_test.json`: an example file for testing the cli_app.py
- `models/`: Directory for storing the fine-tuned models
- `notebooks/`: Jupyter notebooks
  - `baseline.ipynb`: an example file for the training procedure that was used
  - `exploration.ipynb`: a notebook containing dataset exploration
- `src/`: Source code used in this project
  - `cli_app.py`: an executable CLI version of the system
  - `preprocessing.py`: Methods used to load and preprocess the dataset
  - `prediction.py`: Methods used get the LLM embeddings and perform the NER prediction
  - `postprocessing.py`: Methods used to postprocess the dataset, retrieve original tokens and save predictions to a .csv file
  - `trainer.py`: Custom Trainer class created for the experiments with a custom Loss function
  - `utils.py`: Methods used for the training (such as methods for computing the metrics)
- `app.py`: an executable file with a GUI version of the system
- `requirements.txt`: a file containing all the dependencies that have to be installed to run the app.py and cli_app.py
