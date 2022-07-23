"""Script to create and save BERT encodings for all the project datasets."""

import os
import pickle
import random
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from transformers import BertTokenizer, TFBertForSequenceClassification


EVAL_DIR = os.path.join(os.path.dirname(os.getcwd()), "transfer_learning_evaluation")
if not os.path.exists(EVAL_DIR):
    os.mkdir(EVAL_DIR)
    

def get_google_drive_download_url(raw_url: str):
    return "https://drive.google.com/uc?id=" + raw_url.split("/")[-2]


def shuffle(df: pd.DataFrame):
    "Make sure data is shuffled (deterministically)."
    ix = list(df.index)
    random.seed(42)
    random.shuffle(ix)
    return df.loc[ix].reset_index(drop=True)


# Load all the datasets into memory:
datasets = dict()

print("bilal")

datasets["bilal"] = dict()

bilal_train_url = "https://drive.google.com/file/d/1i54O_JSAVtvP5ivor-ARJRkwSoBFdit1/view?usp=sharing"
bilal_test_url = "https://drive.google.com/file/d/1boRdmasHB6JZDNBrlt6MRB1pUVnxxY-6/view?usp=sharing"

bilal_train_val = pd.read_csv(get_google_drive_download_url(bilal_train_url), encoding="latin1")
bilal_test = pd.read_csv(get_google_drive_download_url(bilal_test_url), encoding="latin1")
# Split train into 90-10 split for train-validation as per the paper:
bilal_train, bilal_val = train_test_split(bilal_train_val, test_size=0.1, random_state=42)

datasets["bilal"]["train"] = bilal_train
datasets["bilal"]["test"] = bilal_test
datasets["bilal"]["val"] = bilal_val

datasets["bilal"]["x_col"] = "sentence"
datasets["bilal"]["y_col"] = "label"

print(f"> train={len(bilal_train):,}, test={len(bilal_test):,}, val={len(bilal_val):,}")


print("yelp")

datasets["yelp"] = dict()

yelp_train_url = "https://drive.google.com/file/d/104W3CqRu4hUK1ht7wPfi8r8fDT7xdFCf/view?usp=sharing"
yelp_valid_url = "https://drive.google.com/file/d/1--NRor8D2x5au59_B0LCk9wOHIc8Qh46/view?usp=sharing"
yelp_test_url = "https://drive.google.com/file/d/1-3Czl0HdsMiVnnTQ4ckoAL0mcEDZGpsP/view?usp=sharing"

yelp_train = pd.read_csv(get_google_drive_download_url(yelp_train_url), encoding="utf-8")
yelp_val = pd.read_csv(get_google_drive_download_url(yelp_valid_url), encoding="utf-8")
yelp_test = pd.read_csv(get_google_drive_download_url(yelp_test_url), encoding="utf-8")

datasets["yelp"]["train"] = yelp_train
datasets["yelp"]["test"] = yelp_test
datasets["yelp"]["val"] = yelp_val

datasets["yelp"]["x_col"] = "text"
datasets["yelp"]["y_col"] = "label"

print(f"> train={len(yelp_train):,}, test={len(yelp_test):,}, val={len(yelp_val):,}")


print("amazon_small")

datasets["amazon_small"] = dict()

amazon_small_train_url = "https://raw.githubusercontent.com/toby-p/w266-final-project/main/data/amazon/train.csv"
amazon_small_test_url = "https://raw.githubusercontent.com/toby-p/w266-final-project/main/data/amazon/test.csv"
amazon_small_val_url = "https://raw.githubusercontent.com/toby-p/w266-final-project/main/data/amazon/val.csv"

amazon_small_train = shuffle(pd.read_csv(amazon_small_train_url, encoding="latin1"))
amazon_small_test = shuffle(pd.read_csv(amazon_small_test_url, encoding="latin1"))
amazon_small_val = shuffle(pd.read_csv(amazon_small_val_url, encoding="latin1"))

datasets["amazon_small"]["train"] = amazon_small_train
datasets["amazon_small"]["test"] = amazon_small_test
datasets["amazon_small"]["val"] = amazon_small_val

datasets["amazon_small"]["x_col"] = "reviewText"
datasets["amazon_small"]["y_col"] = "label"

print(f"> train={len(amazon_small_train):,}, test={len(amazon_small_test):,}, val={len(amazon_small_val):,}")


print("amazon_large")

datasets["amazon_large"] = dict()

amazon_large_train_fp = os.path.join(os.getcwd(), "data", "amazon", "train_LARGE.csv")
amazon_large_test_fp = os.path.join(os.getcwd(), "data", "amazon", "test_LARGE.csv")
amazon_large_val_fp = val_fp = os.path.join(os.getcwd(), "data", "amazon", "val_LARGE.csv")

amazon_large_train = shuffle(pd.read_csv(amazon_large_train_fp, encoding="latin1"))
amazon_large_test = shuffle(pd.read_csv(amazon_large_test_fp, encoding="latin1"))
amazon_large_val = shuffle(pd.read_csv(amazon_large_val_fp, encoding="latin1"))

datasets["amazon_large"]["train"] = amazon_large_train
datasets["amazon_large"]["test"] = amazon_large_test
datasets["amazon_large"]["val"] = amazon_large_val

datasets["amazon_large"]["x_col"] = "reviewText"
datasets["amazon_large"]["y_col"] = "label"

print(f"> train={len(amazon_large_train):,}, test={len(amazon_large_test):,}, val={len(amazon_large_val):,}")


# Make all the encodings:
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize(df: pd.DataFrame, name: str, x_col: str):
    tokenized = list()
    for dataframe in df:
        encodings = bert_tokenizer(
            list(df[x_col].values), 
            max_length=320,
            truncation=True,
            padding="max_length", 
            return_tensors="tf"
        )
        tokenized.append(encodings)
    return tokenized


# Make the encodings and save them if not already done:
for name, values in datasets.items():
    dir_path = os.path.join(EVAL_DIR, name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for key in ("train", "val", "test"):
        print(f"{name} - {key}")
        fp = os.path.join(dir_path, f"{key}_tokenized.obj")
        if not os.path.exists(fp):
            print(f"> encoding ... ", end="")
            x_col = values["x_col"]
            encodings = tokenize(values[key], name, x_col)
            with open(fp, "wb") as f:
                pickle.dump(encodings, f)
            print("> finished!")
        else:
            print("> already encoded!")
