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
    batch_size_rr: int = field(default=256, metadata={"help": "Inference batch size."})

    peft: bool = field(default="False")
    quantize_base: bool = field(default="False")


def index(
    model: BGEM3FlagModel,
    corpus: datasets.Dataset,
    batch_size: int = 256,
    max_length: int = 512,
    index_factory: str = "Flat",
    save_path: str = None,
    save_embedding: bool = False,
    load_embedding: bool = False,
):
    """
    1. Encode the entire corpus into dense embeddings;
    2. Create faiss index;
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(save_path, mode="r", dtype=dtype).reshape(-1, dim)

    else:
        corpus_embeddings = model.encode(
            corpus["content"], batch_size=batch_size, max_length=max_length
        )["dense_vecs"]

        dim = corpus_embeddings.shape[-1]

        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype,
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(
                    range(0, length, save_batch_size),
                    leave=False,
                    desc="Saving Embeddings",
                ):
                    j = min(i + save_batch_size, length)
                    memmap[i:j] = corpus_embeddings[i:j]
            else:
                memmap[:] = corpus_embeddings

    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    if model.device == torch.device("cuda"):
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
    model: BGEM3FlagModel,
    queries: datasets,
    faiss_index: faiss.Index,
    k: int = 100,
    batch_size: int = 256,
    max_length: int = 512,
):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = model.encode(
        queries["query"], batch_size=batch_size, max_length=max_length
    )["dense_vecs"]
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


class RRDataset(datasets.Dataset):
    def __init__(self, data_in):
        self._data = data_in

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def make_rr_dataset(queries: datasets, preds: List, labels: List):
    dataset_list = []
    rr_labels = []
    for query, pred, label in zip(queries, preds, labels):
        for pred_el in pred:
            dataset_list.append([query["query"], pred_el])
            rr_labels.append(pred_el in label)
    return RRDataset(dataset_list), rr_labels


def rr_predict(
    model: BGEM3FlagModel,
    rr_dataset: RRDataset,
    batch_size: int,
    max_query_len: int,
    max_passage_len: int,
    k: int,
    rr_labels: List[bool],
):
    similarities = model.compute_score(
        sentence_pairs=rr_dataset._data,
        batch_size=batch_size,
        max_query_length=max_query_len,
        max_passage_length=max_passage_len,
        weights_for_different_modes=[0.4, 0.2, 0.4],
    )["colbert+sparse+dense"]
    similarities = np.array(similarities, dtype=np.float32)
    rr_labels = np.array(rr_labels, dtype=bool)
    num_queries = int(len(similarities) / k)
    similarities = similarities.reshape((num_queries, k))
    rr_labels = rr_labels.reshape((num_queries, k))

    return similarities, rr_labels


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


def evaluate_unified(
    similarities: np.ndarray, labels: np.ndarray, cutoffs=[1, 10, 100]
):
    metrics = {}
    mrrs = np.zeros(len(cutoffs))
    recalls = np.zeros(len(cutoffs))
    for i, (sim, gt) in enumerate(zip(similarities, labels)):
        indices = np.flip(np.argsort(sim))
        similarities[i] = sim[indices]  # k
        labels[i] = gt[indices]

    for sim, gt in zip(similarities, labels):
        # MRR
        for i, ordered_label in enumerate(gt, 1):
            if ordered_label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                break

        # Recall
        for k, cutoff in enumerate(cutoffs):
            recall = np.count_nonzero(gt[:cutoff])
            total = np.count_nonzero(gt)
            recalls[k] += recall / max(min(recall, total), 1)

    mrrs /= len(similarities)
    for i, cutoff in enumerate(cutoffs):
        metrics[f"MRR@{cutoff}"] = mrrs[i]

    recalls /= len(similarities)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall

    metrics["AUC"] = roc_auc_score(labels.flatten().astype(int), similarities.flatten())
    for k, cutoff in enumerate(cutoffs):
        nDCG = ndcg_score(labels, similarities, k=cutoff)
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

    faiss_index = index(
        model=model,
        corpus=corpus,
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding,
    )

    scores, indices = search(
        model=model,
        queries=eval_data,
        faiss_index=faiss_index,
        k=args.k,
        batch_size=args.batch_size,
        max_length=args.max_query_length,
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
    metrics = evaluate(retrieval_results, scores, ground_truths)

    print(metrics)
    print()
    print("doing unified eval: ")

    rr_dataset, rr_labels = make_rr_dataset(eval_data, retrieval_results, ground_truths)

    similarities, rr_labels = rr_predict(
        model=model,
        rr_dataset=rr_dataset,
        batch_size=args.batch_size_rr,
        max_passage_len=args.max_passage_length,
        max_query_len=args.max_query_length,
        k=args.k,
        rr_labels=rr_labels,
    )
    metrics = evaluate_unified(similarities, rr_labels)

    print(metrics)


if __name__ == "__main__":
    print("custom")
    main()
