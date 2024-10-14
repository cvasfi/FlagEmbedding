import logging
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import faiss
import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import ndcg_score, roc_auc_score
from tqdm import tqdm
from transformers import HfArgumentParser

from FlagEmbedding import BGEM3FlagModel, FlagModel

logger = logging.getLogger(__name__)


@dataclass
class Args:
    encoder: str = field(
        metadata={"help": "The encoder peft adapter path."},
    )
    fp16: bool = field(default=False, metadata={"help": "Use fp16 in inference?"})
    add_instruction: bool = field(
        default=False, metadata={"help": "Add query-side instruction?"}
    )

    corpus_data: str = field(
        default="namespace-Pt/msmarco", metadata={"help": "candidate passages"}
    )
    query_data: str = field(
        default="namespace-Pt/msmarco-corpus",
        metadata={"help": "queries and their positive passages for evaluation"},
    )

    max_query_length: int = field(default=32, metadata={"help": "Max query length."})
    max_passage_length: int = field(
        default=128, metadata={"help": "Max passage length."}
    )
    batch_size: int = field(default=256, metadata={"help": "Inference batch size."})
    index_factory: str = field(
        default="Flat", metadata={"help": "Faiss index factory."}
    )
    k: int = field(default=100, metadata={"help": "How many neighbors to retrieve?"})

    save_embedding: bool = field(
        default=False, metadata={"help": "Save embeddings in memmap at save_dir?"}
    )
    load_embedding: bool = field(
        default=False, metadata={"help": "Load embeddings from save_dir?"}
    )
    save_path: str = field(
        default="embeddings.memmap", metadata={"help": "Path to save embeddings."}
    )

    peft: bool = field(default="False")
    quantize_base: bool = field(default="False")


def index(
    corpus_embeddings,
    index_factory: str = "Flat",
):
    """
    1. Encode the entire corpus into dense embeddings;
    2. Create faiss index;
    3. Optionally save embeddings.
    """

    dim = corpus_embeddings.shape[-1]

    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    if faiss.get_num_gpus() > 0:
        # co = faiss.GpuClonerOptions()
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index


def search(
    query_embeddings,
    faiss_index: faiss.Index,
    k: int = 100,
    batch_size: int = 256,
):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """

    query_size = len(query_embeddings)

    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i:j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices


def evaluate(preds, preds_scores, labels, cutoffs=[1, 10, 100]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}

    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall
    recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / max(min(len(recall), len(label)), 1)
    recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall

    # AUC
    pred_hard_encodings = []
    for pred, label in zip(preds, labels):
        pred_hard_encoding = np.isin(pred, label).astype(int).tolist()
        pred_hard_encodings.append(pred_hard_encoding)

    from sklearn.metrics import ndcg_score, roc_auc_score, roc_curve

    pred_hard_encodings1d = np.asarray(pred_hard_encodings).flatten()
    preds_scores1d = preds_scores.flatten()
    auc = roc_auc_score(pred_hard_encodings1d, preds_scores1d)

    metrics["AUC@100"] = auc

    # nDCG
    for k, cutoff in enumerate(cutoffs):
        nDCG = ndcg_score(pred_hard_encodings, preds_scores, k=cutoff)
        metrics[f"nDCG@{cutoff}"] = nDCG

    return metrics


def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    if args.query_data == "namespace-Pt/msmarco-corpus":
        assert args.corpus_data == "namespace-Pt/msmarco"
        eval_data = datasets.load_dataset("namespace-Pt/msmarco", split="dev")
        corpus = datasets.load_dataset("namespace-Pt/msmarco-corpus", split="train")
    else:
        eval_data = datasets.load_dataset(
            "json", data_files=args.query_data, split="train"
        )
        corpus = datasets.load_dataset(
            "json", data_files=args.corpus_data, split="train"
        )

    model = BGEM3FlagModel(
        "BAAI/bge-m3", use_fp16=True
    )  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    model.model.model = PeftModel.from_pretrained(
        model.model.model, args.encoder, is_trainable=False
    )

    query_embeddings = model.encode(
        eval_data["query"],
        batch_size=args.batch_size,
        max_length=args.max_query_length,
        return_sparse=False,
        return_colbert_vecs=True,
        return_dense=True,
    )
    corpus_embeddings = model.encode(
        corpus["content"],
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        return_sparse=False,
        return_colbert_vecs=True,
        return_dense=True,
    )

    faiss_index = index(
        corpus_embeddings["dense_vecs"], index_factory=args.index_factory
    )

    scores, indices = search(
        query_embeddings["dense_vecs"],
        faiss_index=faiss_index,
        k=args.k,
        batch_size=args.batch_size,
    )

    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(corpus[indice]["content"])

    ground_truths = []
    for sample in eval_data:
        ground_truths.append(sample["positive"])

    print("doing dense eval: ")
    metrics = evaluate(
        retrieval_results, scores, ground_truths, cutoffs=[1, 10, args.k]
    )

    print(metrics)
    print()
    print(f"Reranking top {args.k} and evaluating again: ")
    reranked_retrieval_results = []
    reranked_scores = []

    # rerank based on the unified score.
    for indice, query_dense, query_colbert in zip(
        indices,
        query_embeddings["dense_vecs"],
        # query_embeddings["lexical_weights"],
        query_embeddings["colbert_vecs"],
    ):
        query_dense: torch.Tensor
        # filter invalid indices
        indice = indice[indice != -1]

        def apply_on_query(query_sample, corpus_data, func):
            return np.stack(
                [
                    func(
                        query_sample,
                        corpus_data[corpus_idx],
                    )
                    for corpus_idx in indice
                ]
            )

        dense_score = apply_on_query(
            query_dense, corpus_embeddings["dense_vecs"], np.matmul
        )

        colbert_score = apply_on_query(
            query_colbert, corpus_embeddings["colbert_vecs"], model.colbert_score
        )
        # sparse_score = apply_on_query(
        #     query_sparse,
        #     corpus_embeddings["lexical_weights"],
        #     model.compute_lexical_matching_score,
        # )

        weights = [0.4, 0.2, 0.4]
        rankings = (
            weights[0] * dense_score
            # + weights[1] * sparse_score
            + weights[2] * colbert_score
        )
        ordered_rankings_indices = np.flip(np.argsort(rankings))
        ordered_indices = indice[ordered_rankings_indices]
        reranked_scores.append(rankings[ordered_rankings_indices])
        reranked_retrieval_results.append(corpus[ordered_indices]["content"])

    metrics = evaluate(
        reranked_retrieval_results,
        np.stack(reranked_scores),
        ground_truths,
        cutoffs=[1, 10, args.k],
    )

    print(metrics)


if __name__ == "__main__":
    print("custom")
    main()
