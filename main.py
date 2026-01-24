# Standard library imports
import io
import os
import random
import re
from typing import Dict, List, Tuple
from collections import defaultdict

# Third-party imports
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset 
from transformers import AutoImageProcessor, ResNetForImageClassification

# Local application imports
from process_images import *
from classify_images import *
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

class Config:
    TEST_TRANS = False
    SHOW_DATA_SPLITS = False 
    NLS_DATA = r"PD|CTL"
    LABEL_1 = "PD"
    LABEL_2 = "CTL"
    RUN_COPRA = True # train on one and test on the rest of the datasets jointly
    RUN_LOO = False #train on 4 datasets test on one
    RANDOM_STATE = 42
    N_SPLITS = 5
    BATCH_SIZE = 16
    MODEL_NAME = "microsoft/resnet-50"
    PATIENCE = 10
    NUM_EPOCHS = 30
    BASE_PATH = "/../../projects/NLS_ADPIE/data/NLS/handwriting/clean/" 
    DATA_PATH = "/projects/NLS_ADPIE/data/"

random.seed(42)
np.random.seed(42)

def index_list(lst, idx):
    return [lst[i] for i in idx]

def train_and_get_model(
    d_trainval,
    task,
    model_name=Config.MODEL_NAME,
    batch_size=Config.BATCH_SIZE,
    num_epochs=Config.NUM_EPOCHS,
    n_splits=Config.N_SPLITS,
    patience=Config.PATIENCE,
    random_state=Config.RANDOM_STATE,
    testing_data=None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)

    X_full = np.array(d_trainval["X"])
    y_full = np.array(d_trainval["y"])
    groups = np.array(d_trainval["groups"])

    gkf = StratifiedGroupKFold( n_splits=Config.N_SPLITS, shuffle=True, random_state=random_state)

    all_metrics = {task: {"accuracy": [], "auc": [], "f1_score": []}}

    if Config.TEST_TRANS or Config.RUN_COPRA or Config.RUN_LOO:
        print("loading additional tests...")
        new_test_datasets = load_additional_tests(data=testing_data)

    for fold, (trainval_idx, test_idx) in enumerate(
        gkf.split(X_full, y_full, groups)
    ):
        print(f"\n────────── Fold {fold + 1}/{n_splits} ──────────")

        # Train/Val split (GROUP-AWARE)
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=random_state
        )

        train_idx, val_idx = next(
            gss.split(
                X_full[trainval_idx],
                y_full[trainval_idx],
                groups[trainval_idx]
            )
        )

        train_idx = index_list(trainval_idx, train_idx)
        val_idx   = index_list(trainval_idx, val_idx)

        # Extract raw data
        X_train_raw, y_train_raw = X_full[train_idx], y_full[train_idx]
        X_val_raw,   y_val_raw   = X_full[val_idx],   y_full[val_idx]
        X_test_raw,  y_test_raw  = X_full[test_idx],  y_full[test_idx]

        # Preprocess
        X_train, y_train = preprocess_images(processor, X_train_raw, y_train_raw)
        X_val,   y_val   = preprocess_images(processor, X_val_raw, y_val_raw)
        X_test,  y_test  = preprocess_images(processor, X_test_raw, y_test_raw)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train.long()),
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(X_val, y_val.long()),
            batch_size=batch_size,
            shuffle=False
        )

        # Train
        model = ResNetWithMLP().to(device)
        model = train_one_fold(
            train_loader,
            val_loader,
            device,
            num_epochs,
            model=model,
            patience=patience
        )

        # Test (in-fold)
        acc, auc, f1 = testCNN(model, X_test, y_test)
        all_metrics[task]["accuracy"].append(acc)
        all_metrics[task]["auc"].append(auc)
        all_metrics[task]["f1_score"].append(f1)

        # External tests
        if Config.TEST_TRANS or Config.RUN_COPRA or Config.RUN_LOO:
            # This assumes your dictionary is only composed of dataset names as keys and dictionary of folds as value
            # i.e this is a cascaded dictionary
            for test_name, test in new_test_datasets.items():
                fold_name = f"fold{fold+1}"
                test_fold = test[fold_name]
                X, y = preprocess_images(
                    processor,
                    test_fold["X"],
                    test_fold["y"]
                )
                acc, auc, f1 = testCNN(model, X, y)
                all_metrics.setdefault(test_name, {"accuracy": [], "auc": [], "f1_score": []})
                all_metrics[test_name]["accuracy"].append(acc)
                all_metrics[test_name]["auc"].append(auc)
                all_metrics[test_name]["f1_score"].append(f1)
    

        del model, train_loader, val_loader
        torch.cuda.empty_cache()

    return all_metrics

def process_data_and_train(
    task,
    processing_function=None,
    data=None,
    testing_data=None,
    results_csv="results.csv"
):
    """
    Processes data, trains the ResNet and SVM models, prints metrics,
    and appends results to a CSV file.
    """
    print(f"\nProcessing task: {task}")

    if processing_function is not None:
        data = processing_function(task)

    dtrainval = partitionData(data)
    all_metrics = train_and_get_model(
        dtrainval,
        task,
        testing_data=testing_data
    )

    #  TEST-TRANSFER MODE 
    if Config.TEST_TRANS or Config.RUN_LOO:
        for cur_task, metrics in all_metrics.items():
            avg_accuracy = np.mean(metrics["accuracy"])
            std_acc = np.std(metrics["accuracy"])
            avg_auc = np.mean(metrics["auc"])
            avg_f1 = np.mean(metrics["f1_score"])

            print(
                f"Task: {cur_task} - "
                f"Average Accuracy + std: {avg_accuracy:.4f} +- {std_acc:.4f}, "
                f"Average AUC: {avg_auc:.4f}, "
                f"Average F1 Score: {avg_f1:.4f}"
            )

        return all_metrics

    #  COPRA / MULTI-DATASET 
    elif Config.RUN_COPRA :
        metrics = all_metrics["multi_dataset"]

        avg_accuracy = np.mean(metrics["accuracy"])
        std_acc = np.std(metrics["accuracy"])
        avg_auc = np.mean(metrics["auc"])
        avg_f1 = np.mean(metrics["f1_score"])

        print(
            f"Task: Multi_dataset - "
            f"Average Accuracy + std: {avg_accuracy:.4f} +- {std_acc:.4f}, "
            f"Average AUC: {avg_auc:.4f}, "
            f"Average F1 Score: {avg_f1:.4f}"
        )

        return {
            "accuracy": avg_accuracy,
            "auc": avg_auc,
            "f1_score": avg_f1
        }

    #  SINGLE TASK 
    else:
        metrics = all_metrics[task]

        avg_accuracy = np.mean(metrics["accuracy"])
        std_acc = np.std(metrics["accuracy"])
        avg_auc = np.mean(metrics["auc"])
        avg_f1 = np.mean(metrics["f1_score"])

        print(
            f"Task: {task} - "
            f"Average Accuracy + std: {avg_accuracy:.4f} +- {std_acc:.4f}, "
            f"Average AUC: {avg_auc:.4f}, "
            f"Average F1 Score: {avg_f1:.4f}"
        )

        return {
            "accuracy": avg_accuracy,
            "auc": avg_auc,
            "f1_score": avg_f1
        }


def load_additional_tests(data_name=None, data=None, random_state=42):
    new_test_datasets = {}
    if data is None:
        data_sets = ["HandPD", "NewHandPD", "Parkinson_Drawings", "PaHaW", "spirals", "nls_all"]
        for set_name in data_sets:
                data = gatherData(set_name)
                test = partitionData(data)
                gkf = StratifiedGroupKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=random_state)
                test_fold = {}
                for fold, (_,test_idx) in enumerate(gkf.split(test["X"], test["y"], test['groups'])):
                    X_test, y_test = test["X"][test_idx],test["y"][test_idx]
                    test_fold["fold" + str(fold+1)] = {"X": X_test, "y": y_test}
                
                new_test_datasets[set_name] = test_fold

    elif Config.RUN_COPRA:
        test = partitionData(data)
        gkf = StratifiedGroupKFold( n_splits=Config.N_SPLITS, shuffle=True, random_state=random_state)
        test_fold = {}

        for fold, (_,test_idx) in enumerate(gkf.split(test["X"], test["y"], test['groups'])):
            X_test, y_test = np.array(test["X"])[test_idx], np.array(test["y"])[test_idx]
            test_fold["fold" + str(fold+1)] = {"X": X_test, "y": y_test}
        
        new_test_datasets["multi_dataset"] = test_fold
    else:
        test = partitionData(data)
        gkf = StratifiedGroupKFold( n_splits=Config.N_SPLITS, shuffle=True, random_state=random_state)
        test_fold = {}
        
        for fold, (_,test_idx) in enumerate(gkf.split(test["X"], test["y"], test['groups'])):
            X_test, y_test = np.array(test["X"])[test_idx], np.array(test["y"])[test_idx]
            test_fold["fold" + str(fold+1)] = {"X": X_test, "y": y_test}
        
        new_test_datasets[data_name] = test_fold

    return  new_test_datasets 

def display_metrics_table(all_metrics):
    """
    Display averaged metrics in a neatly formatted table (no external libraries).

    Parameters
    ----------
    all_metrics : dict
        Expected format:
        {task: {"accuracy": float, "auc": float, "f1_score": float}}
    """
    headers = ["Task", "Accuracy", "AUC", "F1 Score"]

    # Prepare rows
    rows = []
    for task, metrics in all_metrics.items():
        rows.append([
            task,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['auc']:.4f}",
            f"{metrics['f1_score']:.4f}"
        ])

    # Compute column widths (max length across header + rows)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build row formatter
    row_fmt = " | ".join("{:<" + str(w) + "}" for w in col_widths)

    # Print header
    print(row_fmt.format(*headers))
    print("-+-".join("-" * w for w in col_widths))

    # Print rows
    for row in rows:
        print(row_fmt.format(*row))

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    # Ensure this is the correct path for the user's system
    # It's better to use an absolute path or a robust relative path
    print("Starting...")
    try:
        os.chdir(Config.BASE_PATH)
        print(f"Current working directory: {os.getcwd()}")
    except FileNotFoundError:
        print(f"Warning: The base path {Config.BASE_PATH} was not found. Please set the correct path.")
        exit()
    
    if Config.RUN_COPRA:
        name = ["HandPD", "NewHandPD", "Parkinson_Drawings", "PaHaW", "spirals"]

        dataHPD = gatherData("HandPD")
        dataNHPD = gatherData("NewHandPD")
        dataPD = gatherData("Parkinson_Drawings")
        dataPW = gatherData("PaHaW")
        dataNLS = gatherDataNLS("spirals")

        data = {
            "HandPD" : dataHPD,
            "NewHandPD": dataNHPD,
            "Parkinson_Drawings": dataPD,
            "PaHaW": dataPW,
            "spirals" : dataNLS
            }

        for i in range(5):
            print(f"TRAINING ON {name[i]}")
            DO_NOT_USE = {name[i]}
            dataset_to_use = list(DO_NOT_USE ^ set(name))
            final_data = {}
            for key in dataset_to_use:
                final_data = final_data | data[key]
            
            process_data_and_train(task=name[i], data=data[name[i]], testing_data=final_data)
    elif Config.RUN_LOO:

        name = ["HandPD", "NewHandPD", "Parkinson_Drawings", "PaHaW", "spirals"]

        dataHPD = gatherData("HandPD")
        dataNHPD = gatherData("NewHandPD")
        dataPD = gatherData("Parkinson_Drawings")
        dataPW = gatherData("PaHaW")
        dataNLS = gatherDataNLS("spirals")

        data = {
            "HandPD" : dataHPD,
            "NewHandPD": dataNHPD,
            "Parkinson_Drawings": dataPD,
            "PaHaW": dataPW,
            "spirals" : dataNLS
            }

        for i in range(5):
            print(f"TESTING ON {name[i]}")
            DO_NOT_USE = {name[i]}
            dataset_to_train_on = list(DO_NOT_USE ^ set(name))
            final_data = {}
            for key in dataset_to_train_on:
                final_data = final_data | data[key]
            
            process_data_and_train(task=name[i], data=final_data, testing_data=data[name[i]])

    else:
        print("Begining Expriements")
        all_tasks = ["points", "spirals", "numbers", "writing", "drawing", "all"]
        individual_tasks = [
            "point_DOM", "point_NONDOM", "point_sustained",
            "spiral_DOM", "spiral_NONDOM", "spiral_pataka",
            "numbers", "copytext", "copyreadtext", "freewrite",
            "drawclock", "copycube", "copymage"
        ]
        data_sets = ["HandPD", "NewHandPD", "Parkinson_Drawings", "PaHaW"]

        large_metric = {}
        # Process each task
        print("-" * 60)
        print(f"{'ALL TASKS':^60}")
        print("-" * 60)
        for task in all_tasks:
            print(f"{task:*^60}")
            metrics = process_data_and_train(task, gatherDataNLS)
            large_metric[task] = metrics
        
        # Process individual tasks
        print("-" * 60)
        print(f"{'INDIVIDUAL TASKS':^60}")
        print("-" * 60)
        for task in individual_tasks:
            print(f"{task:*^60}")
            metrics = process_data_and_train(task, gatherDataNLS)
            large_metric[task] = metrics
        
        # process datasets
        print("-" * 60)
        print(f"{'DATASETS':^60}")
        print("-" * 60)
        for dataset in data_sets:
            print(f"{dataset:*^60}")
            metrics = process_data_and_train(dataset, gatherData)
            large_metric[dataset] = metrics
 
        display_metrics_table(large_metric)
