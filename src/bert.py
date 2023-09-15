
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
import torch.nn.functional as F
# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# esm: esm_2
class BertFinetuneModel(nn.Module):
    """Sequence-only baseline: Bert + linear layer on pooler_output"""

    def __init__(self):
        """Initializes the module"""

        super(BertFinetuneModel, self).__init__()
        self.bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
        # freeze the embeddings
        _freeze_bert(self.bert_model, freeze_bert=False, freeze_layer_count=-1)

        self.output = nn.Sequential(
            nn.Linear(self.bert_model.pooler.dense.out_features, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 序列（esm2）
        # 结构 (GVP)
        # 序列 + 结构
        # 可以改一下分类层： ResNet

    def _forward(self, input_ids, attention_mask):
        """Helper function to perform the forward pass.

        Args:
            batch: torch_geometric.data.Data
            input_ids: IDs of the embeddings to be used in the model.
            attention_mask: Masking to use durinig BERT's self-attention.

        Returns:
            logits
        """
        x = self.bert_model(
            input_ids, attention_mask=attention_mask
        ).pooler_output

        outputs = self.output(x)

        return outputs

    def forward(self, input_ids, attention_mask):
        """Performs the forward pass.

        Returns:
            logits
        """
        outputs = self._forward(input_ids, attention_mask)
        return outputs


    
def _freeze_bert(
    bert_model: BertModel, freeze_bert=False, freeze_layer_count=-1
):
    """Freeze parameters in BertModel (in place)

    Args:
        bert_model: HuggingFace bert model
        freeze_bert: Bool whether or not to freeze the bert model
        freeze_layer_count: If freeze_bert, up to what layer to freeze.

    Returns:
        bert_model
    """
    if freeze_bert:
        # freeze the entire bert model
        for param in bert_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            # freeze layers in bert_model.encoder
            for layer in bert_model.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
    return None