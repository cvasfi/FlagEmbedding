import logging
import os
import random
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from .arguments import DataArguments, LoRAArguments, ModelArguments
from .data import GroupCollator, TrainDatasetForCE
from .modeling import CrossEncoder
from .trainer import CETrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoRAArguments)
    )
    model_args, data_args, training_args, lora_args = (
        parser.parse_args_into_dataclasses()
    )
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    lora_args: LoRAArguments
    transformers.logging.set_verbosity_error()
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    _model_class = CrossEncoder
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["classifier", "pre_classifier"],
    )

    if model_args.resume_peft and model_args.lora:
        raise SystemError("Cannot resume and apply lora at the same time.")

    if model_args.resume_peft:
        base_model = _model_class.from_pretrained(
            model_args,
            data_args,
            training_args,
            "BAAI/bge-reranker-v2-m3",
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            quantization_config=bnb_config,
        )
        base_model = prepare_model_for_kbit_training(base_model)
        base_model.gradient_checkpointing_enable()
        model = PeftModel.from_pretrained(
            base_model, model_args.model_name_or_path, is_trainable=True
        )
    else:
        model = _model_class.from_pretrained(
            model_args,
            data_args,
            training_args,
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            quantization_config=bnb_config,
        )

    def print_trainable_parameters(m):
        print("Trainable parameters:")
        for name, param in m.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.size()}")

    # print_trainable_parameters(model)
    # Maybe train with LoRA
    if model_args.lora is True:
        lora_config = LoraConfig(
            r=lora_args.r,
            lora_alpha=lora_args.alpha,
            target_modules=[
                "query",
                "key",
                "value",
                "dense",
                "out_proj",
            ],  # module names specific to bert (small, base, and large)
            lora_dropout=lora_args.dropout,
            bias="none",
            # task_type="SEQ_CLS",
        )
        logger.info("LoRA config: %s", lora_config)

        model = prepare_model_for_kbit_training(model, lora_config)
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print_trainable_parameters(model)

    train_dataset = TrainDatasetForCE(data_args, tokenizer=tokenizer)
    train_dataset.dataset = train_dataset.dataset.shuffle(random.randint(0, 10000))
    _trainer_class = CETrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
        tokenizer=tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
