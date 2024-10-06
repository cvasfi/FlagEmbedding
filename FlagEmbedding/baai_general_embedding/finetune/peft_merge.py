import argparse

from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="BAAI/bge-m3", type=str)
    parser.add_argument("--peft_model_path", default=None, type=str)
    parser.add_argument("--out_model_path", default=None, type=str)
    args = parser.parse_args()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Load the LoRA adapters
    base_model = AutoModel.from_pretrained(
        args.base_model, quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()
    # Save the full model and tokenizer to the checkpoint directory
    model.save_pretrained(args.out_model_path)
    tokenizer.save_pretrained(args.out_model_path)

    word_embedding_model = models.Transformer(args.out_model_path)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls"
    )

    normlize_layer = models.Normalize()
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, normlize_layer], device="cpu"
    )

    model.save(args.out_model_path)


if __name__ == "__main__":
    main()
