import argparse

from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer

parser = argparse.ArgumentParser()


from .modeling import BGEM3Model


def main():
    parser.add_argument(
        "--output_dir", help="where to save output", type=str, required=True
    )
    parser.add_argument(
        "--input_dir", help="Saved peft adapters", type=str, required=True
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        ("BAAI/bge-m3"),
        use_fast=False,
    )
    # The idea is that, loading model in full precision and applying the lora weights is a good enough dequantization from 8 bits.
    model = BGEM3Model(
        model_name="BAAI/bge-m3",
        normlized=True,
        temperature=0.02,
        unified_finetuning=True,
        use_self_distill=True,
        quantized=False,
    )
    model.model = PeftModel.from_pretrained(
        model.model, args.input_dir, is_trainable=False
    )
    model.model = model.model.merge_and_unload()
    model.save(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
