import numpy as np
import csv
import os
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

data_path = '/data1/yuzhe/avgfp_fasta/'
def plot_score(data_path):
    name_seq = {}
    md_list = os.listdir(data_path)
    for md in md_list:
        score_path = data_path+md
        with open (score_path) as f:
            for line in f:
                if not line.startswith('>'):
                    seq = line

                    ID = ('_').join(md.split(",")).split('.')[0]
                    ID = ID + '-A'
                    name_seq[ID] = seq
    return name_seq


DATA_ROOT_DIR = "/data1"     
             
def load_avgfp_data(split="train", seq_only=True):
    filename = os.path.join(DATA_ROOT_DIR,'chenchen/DeepFRI_GO', f"avgfp_{split}.json")
    dataset = json.load(open(filename, "rb"))
    if seq_only:
        # delete the "coords" in data objects
        for obj in dataset:
            obj.pop("coords", None)
    return dataset
# avgfp: min:22.xxx, max:57.xxx
# bgl3: min:82.xxx, max:91.xxx

def change_seqs(name_seq, dataset):
    """
    Add GO labels to a dataset

    Args:
        dataset: list of dict (output from `load_gvp_data`)
        prot2annot: output from `load_GO_labels`
        go_ont: String. GO ontology/task to be used. One of: 'cc', 'bp', 'mf'

    Return:
        Dataset formatted as a list. Where, for each element (dictionary), a `target` field has been added.

    """
    for rec in dataset:
        rec["seq"] = name_seq[rec["name"]]
    return dataset


split = "valid"
name_seq = plot_score(data_path)
dataset = load_avgfp_data(split=split)
dataset = change_seqs(name_seq, dataset)

with open(f"/data1/chenchen/avgfp_{split}.json", "w") as outfile:
    json.dump(dataset , outfile)