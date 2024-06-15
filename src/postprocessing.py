from prediction import model
import pandas as pd


def postprocess_data(data_tokenized, predictions):
    document_list = []
    token_id_list = []
    label_id_list = []
    for doc, token_id, pred in zip(data_tokenized['document'], data_tokenized['org_word_ids'], predictions):
        for j in range(len(pred)):
            if pred[j] != 14 and token_id[j] != None:
                document_list.append(doc)
                token_id_list.append(token_id[j])
                label_id_list.append(pred[j])
    print(label_id_list, token_id_list, document_list)
    pred_df = pd.DataFrame(
        {
            "document": document_list,
            "token_id": token_id_list,
            # "token": token_id_list, for kaggle submission this column name is required
            "label_id": label_id_list,
        }
    )
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
    return final_df
