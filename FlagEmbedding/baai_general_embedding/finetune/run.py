import logging
import os
import random
from pathlib import Path

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.mapping import inject_adapter_in_model
from peft.tuners.lora import LoraModel
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed

from .arguments import DataArguments, LoRAArguments, ModelArguments
from .arguments import RetrieverTrainingArguments as TrainingArguments
from .data import EmbedCollator, TrainDatasetForEmbedding
from .modeling import BiEncoderModel
from .trainer import BiTrainer

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

    # Set seed
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
    logger.info("Config: %s", config)

    model = BiEncoderModel(
        model_name=model_args.model_name_or_path,
        normlized=training_args.normlized,
        sentence_pooling_method=training_args.sentence_pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        use_inbatch_neg=training_args.use_inbatch_neg,
        peft=model_args.peft,
    )

    def print_trainable_parameters(m):
        print("Trainable parameters:")
        for name, param in m.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.size()}")

    lora_config = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.alpha,
        target_modules=[
            "word_embeddings",
            "query",
            "key",
            "value",
            "dense",
        ],  # module names specific to bert (small, base, and large)
        lora_dropout=lora_args.dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    # Maybe train with LoRA
    if training_args.lora is True:
        logger.info("LoRA config: %s", lora_config)

        model.model = prepare_model_for_kbit_training(model.model, lora_config)
        model.model.gradient_checkpointing_enable()
        model.model = get_peft_model(model.model, lora_config)
        model.model.print_trainable_parameters()
        # print_trainable_parameters(model.model)

    if training_args.reapply_lora and not model_args.peft:
        raise ValueError("can only reapply lora on a peft model")

    if training_args.reapply_lora is True:
        new_lora_config = LoraConfig(
            r=lora_args.r,
            lora_alpha=lora_args.alpha,
            target_modules=[
                "word_embeddings",
            ],  # module names specific to bert (small, base, and large)
            lora_dropout=lora_args.dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        logger.info("LoRA config: %s", lora_config)
        print(model.model.modules_to_save)
        model.model = LoraModel(model.model, new_lora_config, "embeddings")
        model.model.add_weighted_adapter(["default", "embeddings"])
        # Step 3: Use BasicTuner to merge adapters
        # tuner = BaseTunerLayer(model, {"embeddings": new_lora_config})
        # tuner.merge_adapter(["embeddings", "default"])

        #        print(model.model.active_adapters)
        # model.model.print_trainable_parameters()
        print_trainable_parameters(model.model)

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    train_dataset.dataset = train_dataset.dataset.shuffle(random.randint(0, 10000))
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
        ),
        tokenizer=tokenizer,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    print("NEW FINETUNE")
    main()
