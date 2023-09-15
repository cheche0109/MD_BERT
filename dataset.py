
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
import torch.nn.functional as F

# For data preprocess
import numpy as np
import csv
import os
import json

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import re

def prep_seq(seq):
    """
    Adding spaces between AAs and replace rare AA [UZOB] to X.
    ref: https://huggingface.co/Rostlab/prot_bert.

    Args
        seq: a string of AA sequence.

    Returns:
        String representing the input sequence where U,Z,O and B has been replaced by X.
    """
    seq_spaced = " ".join(seq)
    seq_input = re.sub(r"[UZOB]", "X", seq_spaced)
    return seq_input

class SequenceDatasetWithTarget(Dataset):
    """Intended for all sequence-only models."""

    def __init__(self, sequences, labels, tokenizer=None, preprocess=True):
        """Initializes the dataset
        Args:
            sequences: list of strings
            labels: tensor of labels [n_samples, n_labels]
            tokenizer: BertTokenizer
            preprocess: Bool. Wheather or not to process the sequences.

        Return:
            None
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        if preprocess:
            self._preprocess()

    def _preprocess(self):
        """Preprocess sequences to input_ids and attention_mask

        Args:

        Return:
            None
        """
        print("Preprocessing seqeuence data...")
        self.sequences = [prep_seq(seq) for seq in self.sequences]
        encodings = self.tokenizer(
            self.sequences, return_tensors="pt", padding=True
        )
        self.encodings = {
            key: val
            for key, val in encodings.items()
            if key in ("input_ids", "attention_mask")
        }

    def __getitem__(self, idx):
        """Retrieve protein information by index.

        Args:
            idx: Integer representing the position of the protein.

        Return:
            Dictionary with `input_ids`, `attention_mask` and `labels`
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }

    def __len__(self):
        """Lenght of the dataset.

        Args:

        Return:
            Integer representing the length of the dataset.
        """
        return len(self.sequences)

DATA_ROOT_DIR = "/data1"

def load_label(filename):
    """
    Loads the GO annotations.

    Args:
        filename: String representing the path to the GO annotations file.
    Returns
        Quatruple where elements are
            1/ a dict of dict with protein annotations: {protein: {'cc': np.array([...])}}
            2/ a dict with metadata of GO terms: {'cc': [goterm1, ...]}
            3/ a dict with metadata of GO names: {'cc': [goname1, ...]}
            4/ a dict with protein counts of GO terms: {'cc': np.array(...)}
    """
    # Load GO annotations
    prot_label = {}
    with open(filename, "r") as f:        
        for row in f:
            row = row.strip().split('\t')
            prot, label = row[0], row[1]
            prot_label[prot] = label
    return prot_label

def load_avgfp_data(
    #gvp_data_dir="{}/gvp-datasets".format(DATA_ROOT_DIR),
    task="MD",
    split="train",
    seq_only=False,
):
    """
    Retrun:
        Dictionary containing the GVP dataset of proteins.
    """
    filename = os.path.join(DATA_ROOT_DIR,task, f"mini_avgfp_{split}.json")
    dataset = json.load(open(filename, "rb"))
    if seq_only:
        # delete the "coords" in data objects
        for obj in dataset:
            obj.pop("coords", None)
    return dataset


def preprocess_seqs(tokenizer, dataset):
    """Preprocess seq in dataset and bind the input_ids, attention_mask.

    Args:
        tokenizer: hugging face artifact. Tokenization to be used in the sequence.
        dataset: Dictionary containing the GVP dataset of proteins.

    Return:
        Input dataset with `input_ids` and `attention_mask`
    """
    seqs = [prep_seq(rec["seq"]) for rec in dataset]
    encodings = tokenizer(seqs, return_tensors="pt", padding=True)
    # add input_ids, attention_mask to the json records
    for i, rec in enumerate(dataset):
        rec["input_ids"] = encodings["input_ids"][i]
        rec["attention_mask"] = encodings["attention_mask"][i]
    return dataset


def load_labels():
    """Load the labels in the GO dataset

    Return:
        Tuple where the first element is a dictionary mapping proteins to their target, second element is an integer with the number of outputs of the task and the third element is a matrix with the weight of each target.
    """
    prot_label = load_label(
        os.path.join(
            DATA_ROOT_DIR,
            "chenchen/DeepFRI_GO/score_avgfp.txt",
        )
    )
    return prot_label


def add_labels(dataset, prot_label):
    """
    Add labels to a dataset
    """
    for rec in dataset:
        rec["target"] = float(prot_label[rec["name"]])
    return dataset


def get_dataset(model_type="", split="train"):
    """Load data from files, then transform into appropriate
    Dataset objects.
    
    Return:
        Torch dataset.
    """
    seq_only = True if model_type == "seq" else False

    tokenizer = None
    if model_type != "struct":
        # need to add BERT
        print("Loading BertTokenizer...")
        tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False
        )

    # Load data from files
    # load labels
    num_outputs = 1
    prot_label = load_labels()
    # load features
    dataset = load_avgfp_data(
        task="chenchen/DeepFRI_GO", split=split, seq_only=seq_only
    )
    add_labels(dataset, prot_label)

    # Convert data into Dataset objects
    if model_type == "seq":
        if num_outputs == 1:
            targets = torch.tensor(
                [obj["target"] for obj in dataset], dtype=torch.float32
            ).unsqueeze(-1)
        else:
            targets = [obj["target"] for obj in dataset]
        dataset = SequenceDatasetWithTarget(
            [obj["seq"] for obj in dataset],
            targets,
            tokenizer=tokenizer,
            preprocess=True,
        )
        
    dataset.num_outputs = num_outputs
    return dataset

        
