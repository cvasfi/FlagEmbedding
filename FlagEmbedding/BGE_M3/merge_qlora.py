import argparse
import logging
import os
import random
from pathlib import Path

import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoTokenizer

parser = argparse.ArgumentParser()


from .modeling import BGEM3Model

logger = logging.getLogger(__name__)


def main():
    parser.add_argument(
        "--output_dir", help="where to save output", type=str, required=True
    )
    parser.add_argument(
        "--input_dir", help="Saved peft adapters", type=str, required=True
    )
    args = parser.parse_args()
    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        ("BAAI/bge-m3"),
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        "BAAI/bge-m3",
        num_labels=num_labels,
    )
    logger.info("Config: %s", config)

    model = BGEM3Model(
        model_name="BAAI/bge-m3",
        normlized=True,
        temperature=0.02,
        unified_finetuning=True,
        use_self_distill=True,
        quantized=True,
    )
    model.model = PeftModel.from_pretrained(
        model.model, args.input_dir, is_trainable=True
    )
    for param in model.model.parameters():
        param.data = param.data.float()
    model.model = model.model.merge_and_unload()

    model.save(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
