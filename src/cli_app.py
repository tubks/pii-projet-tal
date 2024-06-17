from preprocessing import get_dataset_from_path, add_token_ids, preprocess_data, label2id
from prediction import predict, tokenizer
from postprocessing import postprocess_data

# cli_app.py: cmd version, takes a filepath with a batch of essays as an argument, does the preprocessing,
# the predictions, the postprocessing and generates a csv with all the retrieved PII


if __name__ == "__main__":

    help = """
    Welcome to the Personally Identifiable Information detection system.\n
    1. Type the path to a batch of student essays that contain some PII in .json or .csv format (usable example: data/test/ten_essays_test.json).
    2. The system will pass the essays through a fine-tuned LLM (RoBERTa) and label every token.
    3. The system will save all the tokens that are some kind of PII in the result_pii.csv file.
    The columns in the output file will correspond to the document (essay) id, the token id in the essay, the token string and the assigned (non-O) label.
    """
    print(help)

    data_path = input(
        "Input your path to the .json datafile (data/test/ten_essays_test.json or other): ")
    dataset = get_dataset_from_path(data_path)
    dataset = dataset.rename_column("tokens", "token_string")
    dataset = dataset.map(add_token_ids)
    data_tokenized = preprocess_data(
        dataset, tokenizer, label2id=label2id, with_labels=False)
    predictions = predict(data_tokenized)
    result_pii = postprocess_data(
        dataset, data_tokenized, predictions)
    result_pii.to_csv("result_pii.csv", index=False)
    print(f"Successfully saved predictions for your file in result_pii.csv")
