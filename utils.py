
import datetime
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split


# Directory where models will be stored:
MODEL_DIR = os.path.join(os.path.expanduser("~"), "models")


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


def get_google_drive_download_url(raw_url: str):
    return "https://drive.google.com/uc?id=" + raw_url.split("/")[-2]


def shuffle(df: pd.DataFrame):
    """Make sure data is shuffled (deterministically)."""
    ix = list(df.index)
    random.seed(42)
    random.shuffle(ix)
    return df.loc[ix].reset_index(drop=True)


def load_full_datasets():
    """Helper function to load all datasets."""
    datasets = dict()
    
    print("Loading bilal:")
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
    
    print("Loading yelp:")
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

    print("Loading amazon_small:")
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
    
    print("Loading amazon_large:")
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
    
    return datasets


def confusion_matrix(df: pd.DataFrame, normalize: bool = False):
    pvt = pd.pivot_table(df, index="label", columns="yhat", values="text", aggfunc="count").fillna(0).astype(int)
    pvt[" "] = ["Actual = False", "Actual = True"]
    pvt = pvt.set_index(" ").rename(columns={0: "Pred = False", 1: "Pred = True"})
    pvt.columns.name = ""
    if normalize:
        return pvt / pvt.sum().sum()
    else:
        return pvt

    
def accuracy(df: pd.DataFrame):
    return (df["yhat"] == df["label"]).astype(int).sum() / len(df)


def most_confident_predictions(df: pd.DataFrame, category: str = "tp", top: int = 10):
    rank_col = {"tp": "pred_prob_1", "fp": "pred_prob_1", "tn": "pred_prob_0", "fn": "pred_prob_0"}[category]
    subset = df[df["category"] == category].copy()
    return subset.sort_values(by=[rank_col], ascending=False).iloc[:top][["category", "text", "num_words"]].reset_index(drop=True)
