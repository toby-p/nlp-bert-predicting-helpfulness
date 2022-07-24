"""Script to run fine-tuning transfer learning on all relevant models and
datasets."""

import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
    from transformers import BertTokenizer, TFBertForSequenceClassification

from utils import local_save_dir

# Hard-coded filepaths to relevant local directories and files:
EVAL_DIR = os.path.join(os.getcwd(), "data", "transfer_learning_evaluation")
if not os.path.exists(EVAL_DIR):
    os.mkdir(EVAL_DIR)
ENCODING_DIR = os.path.join(os.path.dirname(os.getcwd()), "encodings")
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), "models")
TRAINED_MODELS = {
    "amazon_finetune": os.path.join(MODEL_DIR, "amazon_finetune", "full_model", "2022_07_19__22_18_29"),
    "amazon_finetune_LARGE": os.path.join(MODEL_DIR, "amazon_finetune_LARGE", "full_model", "2022_07_21__10_46_09"),
}


def get_google_drive_download_url(raw_url: str):
    return "https://drive.google.com/uc?id=" + raw_url.split("/")[-2]


def shuffle(df: pd.DataFrame):
    """Make sure data is shuffled (deterministically)."""
    ix = list(df.index)
    random.seed(42)
    random.shuffle(ix)
    return df.loc[ix].reset_index(drop=True)


def base_model():
    """Create a BERT model with parameters specified in the Bilal paper:
    https://link.springer.com/article/10.1007/s10660-022-09560-w/tables/2

        - model: TFBertForSequenceClassification
        - learning rate: 2e-5
        - epsilon: 1e-8
    """
    # Using the TFBertForSequenceClassification as specified in the paper:
    bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Don't freeze any layers:
    untrainable = []
    trainable = [w.name for w in bert_model.weights]

    for w in bert_model.weights:
        if w.name in untrainable:
            w._trainable = False  # NOQA
        elif w.name in trainable:
            w._trainable = True  # NOQA

    # Compile the model:
    bert_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
    )

    return bert_model


def layers_trainable(model):
    """Identify which layers are trainable in the model."""
    return {w.name: w._trainable for w in model.weights}  # NOQA


def freeze_except_classifier(model):
    """Make all layers untrainable except the final classifier layers."""
    trainable, untrainable = 0, 0
    for w in model.weights:
        if w.name.split("/")[1] == "classifier":
            w._trainable = True
            trainable += 1
        else:
            w._trainable = False
            untrainable += 1
    print(f"Froze all layers except classifier; model now has {trainable} trainable "
          f"layers, {untrainable} untrainable.")
    return model


def freeze_except_classifier_and_pooler(model):
    """Make all layers untrainable except the final pooler and classifier layers."""
    trainable, untrainable = 0, 0
    for w in model.weights:
        if (w.name.split("/")[1] == "classifier") or (w.name.split("/")[2] == "pooler"):
            w._trainable = True
            trainable += 1
        else:
            w._trainable = False
            untrainable += 1
    print(f"Froze all layers except pooler and classifier; model now has {trainable} trainable "
          f"layers, {untrainable} untrainable.")
    return model


def finetune_classifier_layers(datasets: dict, dataset_name: str,
                               trained_model_name: str = None,
                               unfreeze_pooler: bool = False):
    """Perform fine-tuning of a pretrained model on a dataset.

    Args:
        datasets: dictionary containing all the required data for fine-tuning.
        dataset_name: top-level key from `datasets`.
        trained_model_name: dict key for `TRAINED_MODELS` (i.e. name of a model
            that was pre-trained on a different dataset. If not passed, the
            TFBertForSequenceClassification base model is used.
        unfreeze_pooler: whether or not to unfreeze the BERT pooler (4 layers to
            train, or just the classifier (2 layers to train).
    """
    # Create fresh base BERT model:
    model = base_model()

    # Load the trained weights into model:
    if trained_model_name is not None:
        model_fp = TRAINED_MODELS[trained_model_name]
        weights_fp = f"{model_fp}/variables/variables"
        model.load_weights(weights_fp)
        print(f"Loaded pre-trained model weights from: {weights_fp}")
    else:
        trained_model_name = "bert_base"

    # Freeze weights except classifier/pooler layers:
    if unfreeze_pooler:
        unfreeze_func = freeze_except_classifier_and_pooler
    else:
        unfreeze_func = freeze_except_classifier
    model = unfreeze_func(model)

    # Get the encodings for fine-tuning:
    train_encodings = datasets[dataset_name]["train_tokenized"]
    valid_encodings = datasets[dataset_name]["val_tokenized"]
    test_encodings = datasets[dataset_name]["test_tokenized"]

    # Create the model name for saving the fine-tuned version:
    if unfreeze_pooler:
        n_layers = 4
    else:
        n_layers = 2
    finetune_name = f"{trained_model_name}_{n_layers}_LAYERS_FINETUNED_ON_{dataset_name}"

    # Create directory for storing checkpoints after each epoch:
    checkpoint_dir = local_save_dir("checkpoints", model_name=finetune_name)
    checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"

    # Create a callback that saves the model's weights:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1)

    y_col = datasets[dataset_name]["y_col"]
    y_train = datasets[dataset_name]["train"][y_col]
    y_val = datasets[dataset_name]["val"][y_col]
    y_test = datasets[dataset_name]["test"][y_col]

    # Fit the model saving weights every epoch:
    history = model.fit(
        [train_encodings.input_ids, train_encodings.token_type_ids, train_encodings.attention_mask],
        y_train.values,
        validation_data=(
            [valid_encodings.input_ids, valid_encodings.token_type_ids, valid_encodings.attention_mask],
            y_val.values
        ),
        batch_size=16,
        epochs=4,
        callbacks=[cp_callback]
    )

    print("Saving model ...")
    model_dir = local_save_dir("full_model", model_name=finetune_name)
    model.save(model_dir)

    print("Saving history ...")
    hist_dir = local_save_dir("history", model_name=finetune_name)
    with open(os.path.join(hist_dir, "hist_dict"), "wb") as hist_file:
        pickle.dump(history.history, hist_file)

    # Save scores on the test set:
    print(f"Saving test scores and predictions ...")
    results_dir = local_save_dir("results", model_name=finetune_name)
    description = f"{trained_model_name} fine-tuned on {dataset_name}"
    test_score = model.evaluate(
        [test_encodings.input_ids, test_encodings.token_type_ids, test_encodings.attention_mask], y_test
    )
    score_fp = os.path.join(results_dir, f"test_score.txt")
    with open(score_fp, "w") as score_file:
        line1 = f"{description} loss = {test_score[0]}\n"
        print(line1)
        score_file.write(line1)
        line2 = f"{description} accuracy = {test_score[1]}\n"
        print(line2)
        score_file.write(line2)

    # Save predictions and classification report:
    predictions = model.predict(
        [test_encodings.input_ids, test_encodings.token_type_ids, test_encodings.attention_mask]
    )
    preds_fp = os.path.join(results_dir, "predictions.csv")
    pred_df = pd.DataFrame(predictions.to_tuple()[0], columns=["pred_prob_0", "pred_prob_1"])
    pred_df["yhat"] = pred_df[["pred_prob_0", "pred_prob_1"]].values.argmax(1)
    pred_df["y"] = y_test
    pred_df["category"] = np.where((pred_df["yhat"] == 1) & (pred_df["y"] == 1), "tp", "None")
    pred_df["category"] = np.where((pred_df["yhat"] == 0) & (pred_df["y"] == 0), "tn", pred_df["category"])
    pred_df["category"] = np.where((pred_df["yhat"] == 1) & (pred_df["y"] == 0), "fp", pred_df["category"])
    pred_df["category"] = np.where((pred_df["yhat"] == 0) & (pred_df["y"] == 1), "fn", pred_df["category"])
    pred_df.to_csv(preds_fp, encoding="utf-8", index=False)
    report = classification_report(y_test, pred_df["yhat"])
    report_fp = os.path.join(results_dir, f"classification_report.txt")
    with open(report_fp, "w") as report_file:
        for line in report.split("\n"):
            report_file.write(f"{line}\n")
    print(f"{description} test set results")
    print(report)

    print("Finished!")


if __name__ == "__main__":

    finetuning_datasets = dict()

    print("Loading bilal:")
    finetuning_datasets["bilal"] = dict()

    bilal_train_url = "https://drive.google.com/file/d/1i54O_JSAVtvP5ivor-ARJRkwSoBFdit1/view?usp=sharing"
    bilal_test_url = "https://drive.google.com/file/d/1boRdmasHB6JZDNBrlt6MRB1pUVnxxY-6/view?usp=sharing"

    bilal_train_val = pd.read_csv(get_google_drive_download_url(bilal_train_url), encoding="latin1")
    bilal_test = pd.read_csv(get_google_drive_download_url(bilal_test_url), encoding="latin1")
    # Split train into 90-10 split for train-validation as per the paper:
    bilal_train, bilal_val = train_test_split(bilal_train_val, test_size=0.1, random_state=42)

    finetuning_datasets["bilal"]["train"] = bilal_train
    finetuning_datasets["bilal"]["test"] = bilal_test
    finetuning_datasets["bilal"]["val"] = bilal_val

    finetuning_datasets["bilal"]["x_col"] = "sentence"
    finetuning_datasets["bilal"]["y_col"] = "label"
    print(f"> train={len(bilal_train):,}, test={len(bilal_test):,}, val={len(bilal_val):,}")

    print("Loading yelp:")
    finetuning_datasets["yelp"] = dict()

    yelp_train_url = "https://drive.google.com/file/d/104W3CqRu4hUK1ht7wPfi8r8fDT7xdFCf/view?usp=sharing"
    yelp_valid_url = "https://drive.google.com/file/d/1--NRor8D2x5au59_B0LCk9wOHIc8Qh46/view?usp=sharing"
    yelp_test_url = "https://drive.google.com/file/d/1-3Czl0HdsMiVnnTQ4ckoAL0mcEDZGpsP/view?usp=sharing"

    yelp_train = pd.read_csv(get_google_drive_download_url(yelp_train_url), encoding="utf-8")
    yelp_val = pd.read_csv(get_google_drive_download_url(yelp_valid_url), encoding="utf-8")
    yelp_test = pd.read_csv(get_google_drive_download_url(yelp_test_url), encoding="utf-8")

    finetuning_datasets["yelp"]["train"] = yelp_train
    finetuning_datasets["yelp"]["test"] = yelp_test
    finetuning_datasets["yelp"]["val"] = yelp_val

    finetuning_datasets["yelp"]["x_col"] = "text"
    finetuning_datasets["yelp"]["y_col"] = "label"
    print(f"> train={len(yelp_train):,}, test={len(yelp_test):,}, val={len(yelp_val):,}")

    # Load pre-saved encodings:
    for name, values in finetuning_datasets.items():
        dir_path = os.path.join(ENCODING_DIR, name)
        for key in ("train", "val", "test"):
            encodings_name = f"{key}_tokenized"
            fp = os.path.join(dir_path, f"{encodings_name}.obj")
            if not os.path.exists(fp):
                print(f"File not found (run make_encodings.py first):\n  {fp}")
            else:
                with open(fp, "rb") as f:
                    encodings = pickle.load(f)
                    finetuning_datasets[name][f"{key}_tokenized"] = encodings

    # Perform fine-tuning experiments on both Bilal dataset and our yelp dataset:
    for name in ("bilal", "yelp"):
        for unfreeze_p in (True, False):
            for pretrained_model in (None, "amazon_finetune", "amazon_finetune_LARGE"):
                finetune_classifier_layers(
                    datasets=finetuning_datasets, dataset_name=name,
                    trained_model_name=pretrained_model, unfreeze_pooler=unfreeze_p
                )
