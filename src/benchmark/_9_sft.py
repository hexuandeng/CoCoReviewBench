#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Purpose
Fine-tune causal language models for text or chat-style datasets, with optional LoRA and int8 training.

Notes
- Supports HuggingFace datasets or local CSV/JSON/JSONL/TXT inputs.
- Chat-format data can truncate user prompts to fit `block_size` while preserving assistant responses.
- Reports the ratio of overlength samples when enabled.

Source
- Adapted from HuggingFace Transformers language-modeling example scripts.

License
- Copyright 2020 The HuggingFace Inc. team.
- Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
"""

import logging
import math
import time
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
import copy
import json
from transformers.utils import add_start_docstrings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# Label value for tokens that should be ignored by the loss.
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


# LoRA-specific training arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    ignore_overlength: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, filter out samples whose untruncated length exceeds block_size before preprocessing."
            )
        },
    )
    report_overlength_ratio: bool = field(
        default=True,
        metadata={"help": "Whether to report the ratio of samples exceeding block_size during preprocessing."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class LoRATrainingArguments(TrainingArguments):
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA."}
    )
    use_int8_training: bool = field(
        default=False,
        metadata={"help": "Whether to use int8 training."}
    )
    lora_config: Optional[str] = field(
        default=None,
        metadata={"help": "LoRA config file."},
    )


# Save PEFT adapters when a checkpoint is written.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        return control


# Save PEFT adapters at the end of training.
class SavePeftModelAtEndCallback(TrainerCallback):
    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        peft_model_path = os.path.join(args.output_dir, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        return control


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Parse model/data/training arguments from CLI or JSON.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoRATrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        ext = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if ext in ["json", "jsonl"]:
            extension = "json"
        elif ext == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        else:
            extension = ext
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if data_args.validation_split_percentage and "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if training_args.use_int8_training:
            device_map = "auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                load_in_8bit=True,  # Enable 8-bit loading.
                device_map=device_map,  # Required for 8-bit loading.
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        
    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    tokenizer.padding_side = "right"

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    if training_args.use_lora:
        if training_args.use_int8_training:
            model = prepare_model_for_kbit_training(model)
        lora_hyper = json.load(open(training_args.lora_config))
        for key, value in lora_hyper.items():
            logger.info("{} : {}".format(key, value))
        lora_config = LoraConfig(
            r=lora_hyper['lora_r'],
            lora_alpha=lora_hyper['lora_alpha'],
            target_modules=lora_hyper['lora_target_modules'],
            lora_dropout=lora_hyper['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA configs: {lora_config}")
        # Ensure inputs require grads when gradient checkpointing is enabled.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # Helpers for building chat prompts and truncating user text.
    def _build_chat_texts(system: str, user: str, assistant: str, enable_thinking: bool = True):
        msgs = []
        if system and len(system) > 0:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user if user is not None else ""})
        prefix = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        full = tokenizer.apply_chat_template(
            msgs + [{"role": "assistant", "content": assistant if assistant is not None else ""}],
            tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking
        )
        return prefix, full

    def _truncate_user_to_fit(system: str, user: str, assistant: str, enable_thinking: bool, block_size: int):
        """
        If (prefix + assistant) tokens exceed block_size, truncate user text first
        to preserve the assistant response.
        Returns: trimmed_user, was_overlength, prefix_text, full_text
        """
        # Initial build.
        prefix, full = _build_chat_texts(system, user or "", assistant or "", enable_thinking)
        full_ids = tokenizer(full, truncation=False, padding=False)["input_ids"]
        prefix_ids = tokenizer(prefix, truncation=False, padding=False)["input_ids"]
        was_over = len(full_ids) > block_size
        if not was_over:
            return user, False, prefix, full

        # Assistant token length.
        a_len = len(full_ids) - len(prefix_ids)

        # Empty-user baseline.
        prefix_empty, full_empty = _build_chat_texts(system, "", assistant or "", enable_thinking)
        prefix_empty_ids = tokenizer(prefix_empty, truncation=False, padding=False)["input_ids"]

        # Allowed prefix length and user token budget (relative to empty prefix).
        allowed_prefix = max(0, block_size - a_len)
        user_budget = max(0, allowed_prefix - len(prefix_empty_ids))

        if user_budget <= 0:
            # No room for user text; clear it.
            prefix, full = _build_chat_texts(system, "", assistant or "", enable_thinking)
            return "", True, prefix, full

        # Truncate user text by token count.
        user_ids = tokenizer(user or "", add_special_tokens=False)["input_ids"]
        if len(user_ids) > user_budget:
            user_ids = user_ids[:user_budget]
        trimmed_user = tokenizer.decode(user_ids, skip_special_tokens=True)

        # Rebuild with trimmed user and return.
        prefix, full = _build_chat_texts(system, trimmed_user, assistant or "", enable_thinking)
        return trimmed_user, True, prefix, full

    def preprocess_function(examples):
        was_truncated_flags = []  # Track whether the original sample exceeded block_size.

        if "assistant" in examples:
            assistants = examples["assistant"]
            systems = examples["system"] if "system" in examples else [""] * len(assistants)
            users = examples["user"] if "user" in examples else [""] * len(assistants)
            enable_thinking = examples["enable_thinking"]

            prefixes = []
            full_texts = []
            for s, u, a, k in zip(systems, users, assistants, enable_thinking):
                # Check length; if over and not ignoring, truncate user text.
                if model_args.ignore_overlength:
                    # Filtering is handled earlier; compute was_over for reporting.
                    _, full0 = _build_chat_texts(s, u or "", a or "", k)
                    was_over = len(tokenizer(full0, truncation=False, padding=False)["input_ids"]) > block_size
                    was_truncated_flags.append(was_over)
                    prefix, full = _build_chat_texts(s, u or "", a or "", k)
                else:
                    trimmed_user, was_over, prefix, full = _truncate_user_to_fit(s, u or "", a or "", k, block_size)
                    was_truncated_flags.append(was_over)

                prefixes.append(prefix)
                full_texts.append(full)

            prefix_tokenized = tokenizer(prefixes, truncation=True, max_length=block_size, padding=False)
            text_tokenized = tokenizer(full_texts, truncation=True, max_length=block_size, padding=False)

            # Labels: train on assistant tokens only (mask prefix with IGNORE_INDEX).
            labels = copy.deepcopy(text_tokenized["input_ids"])
            prefix_lengths = [len(p) for p in prefix_tokenized["input_ids"]]
            for label, prefix_len in zip(labels, prefix_lengths):
                label[: prefix_len] = [IGNORE_INDEX] * prefix_len

            text_tokenized["labels"] = labels
            text_tokenized["was_truncated"] = was_truncated_flags
            return text_tokenized

        with CaptureLogger(tok_logger) as cl:
            _text = examples[text_column_name]  # May be multiple entries.
            if "prefix" in column_names:
                _prefix = examples["prefix"]

                prefixes = []
                full_texts = []
                for p, t in zip(_prefix, _text):
                    # Treat prefix as user prompt; preserve response text.
                    if model_args.ignore_overlength:
                        # Pre-filtering done; only compute the flag.
                        pref = tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                        )
                        full0 = pref + (t or "") + tokenizer.eos_token
                        was_over = len(tokenizer(full0, truncation=False, padding=False)["input_ids"]) > block_size
                        was_truncated_flags.append(was_over)
                        prefix_text = pref
                        full_text = full0
                    else:
                        # Estimate assistant length and truncate prefix tokens to fit.
                        pref_empty = tokenizer.apply_chat_template(
                            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
                        )
                        full_try = tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                        ) + (t or "") + tokenizer.eos_token

                        full_ids = tokenizer(full_try, truncation=False, padding=False)["input_ids"]
                        pref_ids = tokenizer(pref_empty, truncation=False, padding=False)["input_ids"]  # Empty-user baseline.
                        # Estimate assistant length using the non-empty prefix variant for accuracy.
                        pref_full = tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                        )
                        pref_full_ids = tokenizer(pref_full, truncation=False, padding=False)["input_ids"]
                        a_len = len(full_ids) - len(pref_full_ids)
                        was_over = len(full_ids) > block_size
                        was_truncated_flags.append(was_over)

                        allowed_prefix = max(0, block_size - a_len)
                        user_budget = max(0, allowed_prefix - len(pref_ids))
                        if user_budget <= 0:
                            prefix_text = pref_empty
                        else:
                            user_ids = tokenizer(p or "", add_special_tokens=False)["input_ids"]
                            user_ids = user_ids[:user_budget] if len(user_ids) > user_budget else user_ids
                            p_trim = tokenizer.decode(user_ids, skip_special_tokens=True)
                            prefix_text = tokenizer.apply_chat_template(
                                [{"role": "user", "content": p_trim}],
                                tokenize=False, add_generation_prompt=True
                            )
                        full_text = prefix_text + (t or "") + tokenizer.eos_token

                    prefixes.append(prefix_text)
                    full_texts.append(full_text)

                prefix_tokenized = tokenizer(prefixes, truncation=True, max_length=block_size, padding=False)
                text_tokenized = tokenizer(full_texts, truncation=True, max_length=block_size, padding=False)
                labels = copy.deepcopy(text_tokenized["input_ids"])
                prefix_lengths = [len(p) for p in prefix_tokenized["input_ids"]]
                for label, prefix_len in zip(labels, prefix_lengths):
                    label[:prefix_len] = [IGNORE_INDEX] * prefix_len
            else:
                # Plain-text LM: only track overlength; apply standard truncation.
                was_truncated_flags = []
                for t in _text:
                    full_len = len(tokenizer(t or "", truncation=False, padding=False)["input_ids"])
                    was_truncated_flags.append(full_len > block_size)
                text_tokenized = tokenizer(_text, truncation=True, max_length=block_size, padding=False)
                labels = copy.deepcopy(text_tokenized["input_ids"])

            text_tokenized["labels"] = labels
            text_tokenized["was_truncated"] = was_truncated_flags

        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Ignore the warning above; overlength inputs will be truncated before modeling."
            )
        return text_tokenized

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="example per line with padding"):
        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Tokenize with padding",
            )
        else:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
            )

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    # Report overlength ratio when ignore_overlength is disabled.
    if model_args.report_overlength_ratio and not model_args.ignore_overlength and not data_args.streaming:
        if training_args.do_train and "was_truncated" in train_dataset.features:
            flags = train_dataset["was_truncated"]
            ratio = (sum(1 for x in flags if x) / len(flags)) if len(flags) > 0 else 0.0
            logger.info(
                f"[train] Overlength samples: {ratio:.2%} "
                f"({sum(1 for x in flags if x)} / {len(flags)})"
            )
        if training_args.do_eval and "was_truncated" in eval_dataset.features:
            flags = eval_dataset["was_truncated"]
            ratio = (sum(1 for x in flags if x) / len(flags)) if len(flags) > 0 else 0.0
            logger.info(
                f"[validation] Overlength samples: {ratio:.2%} "
                f"({sum(1 for x in flags if x)} / {len(flags)})"
            )

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

    # Load pretrained adapter weights from a checkpoint, if present.
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        if checkpoint:
            peft_model_name = os.path.join(checkpoint, "adapter_model/adapter_model.bin")
            if os.path.exists(peft_model_name):
                logger.info(f"Loading pretrained adapter weights from {peft_model_name}")
                adapters_weights = torch.load(peft_model_name)
                set_peft_model_state_dict(model, adapters_weights)
                logger.info("Verifying trainable parameters...")
                model.print_trainable_parameters()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                          padding=True, label_pad_token_id=IGNORE_INDEX),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval else None,
        callbacks=[SavePeftModelCallback, SavePeftModelAtEndCallback] if training_args.use_lora else None,
    )
    # Disable cache for LoRA training.
    if training_args.use_lora:
        model.config.use_cache = False

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
