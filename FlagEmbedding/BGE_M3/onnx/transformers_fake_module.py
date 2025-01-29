# modified from https://huggingface.co/aapot/bge-m3-onnx/blob/main/export_onnx.py
# Do NOT change the name prefix, otherwise onnx exporter will refuse to acknowledge the library name is transformers :D

import os
from typing import Dict

import torch
from huggingface_hub import snapshot_download
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel


class BGEM3InferenceModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        colbert_dim: int = -1,
    ) -> None:
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name)
        self.colbert_linear = torch.nn.Linear(
            in_features=self.model.config.hidden_size,
            out_features=(
                self.model.config.hidden_size if colbert_dim == -1 else colbert_dim
            ),
        )
        self.sparse_linear = torch.nn.Linear(
            in_features=self.model.config.hidden_size, out_features=1
        )
        colbert_state_dict = torch.load(
            os.path.join(model_name, "colbert_linear.pt"), map_location="cpu"
        )
        sparse_state_dict = torch.load(
            os.path.join(model_name, "sparse_linear.pt"), map_location="cpu"
        )
        self.colbert_linear.load_state_dict(colbert_state_dict)
        self.sparse_linear.load_state_dict(sparse_state_dict)

    def dense_embedding(self, last_hidden_state: Tensor) -> Tensor:
        return last_hidden_state[:, 0]

    def sparse_embedding(self, last_hidden_state: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.relu(self.sparse_linear(last_hidden_state))

    def colbert_embedding(
        self, last_hidden_state: Tensor, attention_mask: Tensor
    ) -> Tensor:
        with torch.no_grad():
            colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * attention_mask[:, 1:][:, :, None].float()
        return colbert_vecs

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad():
            last_hidden_state = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state

        output = {}
        dense_vecs = self.dense_embedding(last_hidden_state)
        output["dense_vecs"] = torch.nn.functional.normalize(dense_vecs, dim=-1)

        sparse_vecs = self.sparse_embedding(last_hidden_state)
        output["sparse_vecs"] = sparse_vecs

        colbert_vecs = self.colbert_embedding(last_hidden_state, attention_mask)
        output["colbert_vecs"] = torch.nn.functional.normalize(colbert_vecs, dim=-1)

        return output
