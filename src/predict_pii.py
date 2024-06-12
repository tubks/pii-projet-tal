from preprocessing import get_dataset_from_path, preprocess_data
from prediction import predict, tokenizer, model, label2id
import numpy as np
import pandas as pd

# predict_pii.py: cmd version, takes a filepath with a batch of essays as an argument, does the preprocessing,
# predictions and generates a csv


if __name__ == "__main__":

    help = """
    Welcome to the Personally Identifiable Information detection system.\n
    0.5 Download the fine-tuned model (this will be done automatically)
    1. Type the path to a batch of student essays that contain some PII in .json or .csv format.
    2. The system will pass the essays through a fine-tuned LLM and assign a label to every word.
    3. The system will save all the tokens that are some kind of PII in the result_pii.csv file.
    The columns in the output file will correspond to the essay id, the token id and the assigned non-O label.
    """
    print(help)

    data_path = input("Input your path to the .json datafile: ")

    data = get_dataset_from_path(data_path)
    data_tokenized = preprocess_data(
        data, tokenizer, label2id=label2id, with_labels=False)
    data_tokenized.set_format(
        type='pt', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    predictions = predict(data_tokenized)

    document_list = []
    token_id_list = []
    label_id_list = []
    for doc, token_id, pred in zip(data_tokenized['document'], data_tokenized['org_word_ids'], predictions):
        for j in range(len(pred)):
            current_word_id = token_id[j]
            if pred[j] != 14 and token_id[j] != None:
                document_list.append(doc)
                token_id_list.append(token_id[j])
                label_id_list.append(pred[j])
    pred_df = pd.DataFrame(
        {
            "document": document_list,
            "token_id": token_id_list,
            # "token": token_id_list, for kaggle submission this column name is required
            "label_id": label_id_list,
        }
    )
    print(pred_df)
    # map integer label to BIO format label
    pred_df["label"] = pred_df.label_id.map(model.config.id2label)
    no_duplicates_df = pred_df.drop_duplicates(
        subset=['token_id', 'document'], keep='first').reset_index(drop=True)
    final_df = no_duplicates_df.drop(
        columns=["label_id"])  # remove extra columns
    final_df = final_df.rename_axis(
        "row_id").reset_index()  # add `row_id` column
    final_df.to_csv("result_pii.csv", index=False)
    print(final_df)

    print(f"Successfully saved predictions for your file in result_pii.csv")
