# Personal Identifiable Information Data Detection

The goal is to develop a model that detects personally identifiable information (PII) in student writing. Automate the detection and removal of PII from educational data will lower the cost of releasing educational datasets. This will support learning science research and the development of educational tools.

Reliable automated techniques could allow researchers and industry to tap into the potential that large public educational datasets offer to support the development of effective tools and interventions for supporting teachers and students.

## Table of Contents

- [Project Structure](#project-structure)

## Project Structure


- `models/`: Directory for storing the fine-tuned models
- `notebooks/`: Jupyter notebooks
  - `baseline.ipynb`: an example file for the training procedure that was used
  - `exploration.ipynb`: a notebook containing dataset exploration
- `src/`: Source code used in this project
  - `preprocessing.py`: Methods used to load and preprocess the dataset
  - `trainer.py`: Custom Trainer class created for the experiments with a custom Loss function
  - `utils.py`: Methods used for the training (such as the computing the metrics)
