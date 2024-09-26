from peft import AutoPeftModelForFeatureExtraction
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
from transformers.trainer import *

from FlagEmbedding import BGEM3FlagModel


def save_ckpt_for_sentence_transformers(
    ckpt_dir, pooling_mode: str = "cls", normlized: bool = True
):
    word_embedding_model = models.Transformer(ckpt_dir)

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode
    )
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, normlize_layer], device="cpu"
        )
    else:
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device="cpu"
        )
    model.save(ckpt_dir)


def merge_and_save_ckpt_for_sentence_transformers(
    base_model: str, ckpt_dir, pooling_mode: str = "cls", normalized: bool = True
):

    # Load the LoRA adapters
    base_model = AutoModel.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # Merge the adapters into the base model
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model = model.merge_and_unload()
    # Save the full model and tokenizer to the checkpoint directory
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode
    )
    if normalized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, normlize_layer], device="cpu"
        )
    else:
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device="cpu"
        )
    model.save(ckpt_dir)


class BiTrainer(Trainer):
    def __init__(
        self, base_model: str = "BAAI/bge-m3", peft: bool = False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._base_model = base_model
        self._peft = peft

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, "save"):
            raise NotImplementedError(
                f"MODEL {self.model.__class__.__name__} "
                f"does not support save interface"
            )
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Possibly enable it later.

        # # save the checkpoint for sentence-transformers library
        # if self.is_world_process_zero():
        #     if self._peft:
        #         merge_and_save_ckpt_for_sentence_transformers(
        #             self._base_model,
        #             output_dir,
        #             pooling_mode=self.args.sentence_pooling_method,
        #             normalized=self.args.normlized,
        #         )
        #     else:
        #         save_ckpt_for_sentence_transformers(
        #             output_dir,
        #             pooling_mode=self.args.sentence_pooling_method,
        #             normalized=self.args.normlized,
        #         )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
