
import datetime
import os

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
