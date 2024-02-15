# Personal Identifiable Information Data Detection

The goal is to develop a model that detects personally identifiable information (PII) in student writing. Automate the detection and removal of PII from educational data will lower the cost of releasing educational datasets. This will support learning science research and the development of educational tools.

Reliable automated techniques could allow researchers and industry to tap into the potential that large public educational datasets offer to support the development of effective tools and interventions for supporting teachers and students.

## Table of Contents

- [Project Structure](#project-structure)

## Project Structure

- `configs/`: Configuration, training parameters, loggin and parameters for each model
- `data/`: Directory for storing raw and processed datasets
  - `processed/`: The final, canonical datasets for modeling
  - `raw/`: The original, immutable dataseTt
- `models/`: Trained and serialized models, model predictions, or model summaries
- `notebooks/`: Jupyter notebooks
- `/src`: Source source for use in the project
  - `data/`: Scripts to download and generated data
  - `features/`: Scripts to turn raw data into features for modeling
  - `models/`: Scripts to train models and use them to make predictions
  - `visualization/`: Scripts to create explanatory and result-oriented visulaizations