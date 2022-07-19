"""Script version to run in the cloud."""

import datetime
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


MODEL_NAME = "bilal_full_yelp_finetune"
BATCH_SIZE = 16
EPOCHS = 4
MAX_LENGTH = 320


if __name__ == "__main__":

    # Code for helping save models locally after training:

    # Directory where models will be stored:
    MODEL_DIR = os.path.join(os.path.expanduser("~"), "models")

    # Make the directories for storing results if they don't exist yet:
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)


    def local_save_dir(*subdir: str, model_name: str = "test_model"):
        """Create timestamped directory local for storing checkpoints or models.

        Args:
            subdir: optional subdirectories of the main model directory
                (e.g. `checkpoints`, `final_model`, etc.)
            model_name: main name for directory specifying the model being saved.
        """
        model_dir = f"{MODEL_DIR}/{model_name}"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        for s in subdir:
            model_dir = f"{model_dir}/{s}"
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
        now = datetime.datetime.now()
        now_str = now.strftime("%Y_%m_%d__%H_%M_%S")
        dir_path = f"{model_dir}/{now_str}"
        os.mkdir(dir_path)
        print(f"Created dir: {dir_path}")
        return dir_path


    # Using the data downloaded from here: https://sites.google.com/view/review-helpfulness-prediction/datasets
    # Saved publicly in Google Drive.
    def get_google_drive_download_url(raw_url: str):
        return "https://drive.google.com/uc?id=" + raw_url.split("/")[-2]


    train_url = "https://drive.google.com/file/d/104W3CqRu4hUK1ht7wPfi8r8fDT7xdFCf/view?usp=sharing"
    valid_url = "https://drive.google.com/file/d/1--NRor8D2x5au59_B0LCk9wOHIc8Qh46/view?usp=sharing"
    test_url = "https://drive.google.com/file/d/1-3Czl0HdsMiVnnTQ4ckoAL0mcEDZGpsP/view?usp=sharing"

    yelp_train = pd.read_csv(get_google_drive_download_url(train_url), encoding="utf-8")
    yelp_valid = pd.read_csv(get_google_drive_download_url(valid_url), encoding="utf-8")
    yelp_test = pd.read_csv(get_google_drive_download_url(test_url), encoding="utf-8")

    x_train = yelp_train["text"]
    y_train = yelp_train["label"]
    x_val = yelp_valid["text"]
    y_val = yelp_valid["label"]
    x_test = yelp_test["text"]
    y_test = yelp_test["label"]

    print(f"Shape x_train: {x_train.shape}")
    print(f"Shape x_val: {x_val.shape}")
    print(f"Shape x_test: {x_test.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Shape y_val: {y_val.shape}")
    print(f"Shape y_test: {y_test.shape}")

    # Using BERT base uncased tokenizer as per the paper:
    print("Tokenizing ...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_encodings = bert_tokenizer(
        list(x_train.values),
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    valid_encodings = bert_tokenizer(
        list(x_val.values),
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    test_encodings = bert_tokenizer(
        list(x_test.values),
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    print("Training ...")


    def bilal_full_yelp_finetune():
        """Create a BERT model using the model and parameters specified in the Bilal paper:
        https://link.springer.com/article/10.1007/s10660-022-09560-w/tables/2

            - model: TFBertForSequenceClassification
            - learning rate: 2e-5
            - epsilon: 1e-8
        """
        # Using the TFBertForSequenceClassification as specified in the paper:
        bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # Don't freeze any layers:
        untrainable = []
        trainable = [w.name for w in bert_model.weights]

        for w in bert_model.weights:
            if w.name in untrainable:
                w._trainable = False
            elif w.name in trainable:
                w._trainable = True

        # Compile the model:
        bert_model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
        )

        return bert_model


    model = bilal_full_yelp_finetune()
    print(model.summary())

    # Create directory for storing checkpoints after each epoch:
    checkpoint_dir = local_save_dir("checkpoints", model_name=MODEL_NAME)
    checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"

    # Create a callback that saves the model's weights:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1)

    # Fit the model saving weights every epoch:
    history = model.fit(
        [train_encodings.input_ids, train_encodings.token_type_ids, train_encodings.attention_mask],
        y_train.values,
        validation_data=(
            [valid_encodings.input_ids, valid_encodings.token_type_ids, valid_encodings.attention_mask],
            y_val.values
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[cp_callback]
    )

    print("Saving model, scores, and predictions ...")
    # Save the entire model to GDrive:
    model_dir = local_save_dir("full_model", model_name=MODEL_NAME)
    model.save(model_dir)

    # Save scores on the test set:
    test_score = model.evaluate(
        [test_encodings.input_ids, test_encodings.token_type_ids, test_encodings.attention_mask], y_test)
    print("Test loss:", test_score[0])
    print("Test accuracy:", test_score[1])
    score_fp = os.path.join(model_dir, "test_score.txt")
    with open(score_fp, "w") as f:
        f.write(f"Test loss = {test_score[0]}\n")
        f.write(f"Test accuracy = {test_score[1]}\n")

    # Save predictions and classification_report:
    predictions = model.predict(
        [test_encodings.input_ids, test_encodings.token_type_ids, test_encodings.attention_mask])
    preds_fp = os.path.join(model_dir, "test_predictions.csv")
    pred_df = pd.DataFrame(predictions.to_tuple()[0], columns=["pred_prob_0", "pred_prob_1"])
    pred_df["yhat"] = pred_df[["pred_prob_0", "pred_prob_1"]].values.argmax(1)
    pred_df["y"] = y_test
    pred_df["category"] = np.where((pred_df["yhat"] == 1) & (pred_df["y"] == 1), "tp", None)
    pred_df["category"] = np.where((pred_df["yhat"] == 0) & (pred_df["y"] == 0), "tn", pred_df["category"])
    pred_df["category"] = np.where((pred_df["yhat"] == 1) & (pred_df["y"] == 0), "fp", pred_df["category"])
    pred_df["category"] = np.where((pred_df["yhat"] == 0) & (pred_df["y"] == 1), "fn", pred_df["category"])
    pred_df.to_csv(preds_fp, encoding="utf-8", index=False)
    report = classification_report(y_test, pred_df["yhat"])
    report_fp = os.path.join(model_dir, "classification_report.txt")
    with open(report_fp, "w") as f:
        for line in report.split("\n"):
            f.write(f"{line}\n")
    print(f"{MODEL_NAME} - test set results")
    print(report)

    print("Saving history ...")
    # Save the history file:
    hist_dir = local_save_dir("history", model_name=MODEL_NAME)
    with open(os.path.join(hist_dir, "hist_dict"), "wb") as f:
        pickle.dump(history.history, f)

    print("Finished!")
