from prediction import model
import pandas as pd

def postprocess_data(dataset, data_tokenized, predictions):
    document_list = []
    token_id_list = []
    label_id_list = []
    for doc, token_id, pred in zip(data_tokenized['document'], data_tokenized['org_word_ids'], predictions):
        for j in range(len(pred)):
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
    # map integer label to BIO format label
    pred_df["label"] = pred_df.label_id.map(model.config.id2label)
    no_duplicates_df = pred_df.drop_duplicates(
        subset=['token_id', 'document'], keep='first').reset_index(drop=True)
    final_df = no_duplicates_df.drop(
        columns=["label_id"])  # remove extra columns
    final_df = final_df.rename_axis(
        "row_id").reset_index()  # add `row_id` column
    ds_df = dataset.to_pandas()
    ds_df = ds_df.explode(['token_id', 'token_string', 'labels'])
    df_back_to_token_string = pd.merge(
        ds_df, final_df, how='inner', on=['token_id', 'document'])
    aligned_pii = df_back_to_token_string[[
        'document', 'token_id', 'token_string', 'label']]
    return aligned_pii
