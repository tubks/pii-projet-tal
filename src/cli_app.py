from preprocessing import get_dataset_from_path, preprocess_data
from prediction import predict, tokenizer, model, label2id
from postprocessing import postprocess_data
import numpy as np
import pandas as pd

# predict_pii.py: cmd version, takes a filepath with a batch of essays as an argument, does the preprocessing,
# predictions, postprocessing and generates a csv


if __name__ == "__main__":

    help = """
    Welcome to the Personally Identifiable Information detection system.\n
    0.5 Download the fine-tuned model (this will be done automatically)
    1. Type the path to a batch of student essays that contain some PII in .json or .csv format.
    2. The system will pass the essays through a fine-tuned LLM and assign a label to every word.
    3. The system will save all the tokens that are some kind of PII in the result_pii.csv file.
    The columns in the output file will correspond to the essay id, the token string and the assigned (non-O) label.
    """
    print(help)

    data_path = input("Input your path to the .json datafile: ")
    dataset = get_dataset_from_path(data_path)
    dataset = dataset.rename_column("tokens", "token_string")

    def add_token_ids(example):
        return {'token_id': [i for i in range(len(example['token_string']))]}
    data = dataset.map(add_token_ids)
    data_tokenized = preprocess_data(
        data, tokenizer, label2id=label2id, with_labels=False)
    predictions = predict(data_tokenized)
    final_df = postprocess_data(data_tokenized, predictions)
    ds_df = data.to_pandas()
    ds_df = ds_df.explode(['token_id', 'token_string'])
    df_back_to_token_string = pd.merge(
        ds_df, final_df, how='inner', on=['token_id'])
    aligned_pii = df_back_to_token_string[['token_string', 'label']]
    print(aligned_pii)
    print(f"Successfully saved predictions for your file in result_pii.csv")
