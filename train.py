import sys
import logging
try:
    local_rank = int(sys.argv[1].split("=")[-1])
except:
    local_rank = -1
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if local_rank<=0 else logging.WARN,
)
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random
import json

from datasets import load_dataset, set_progress_bar_enabled
set_progress_bar_enabled(False)

from pathlib import Path
import torch.distributed as dist

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from diffaug.models import RobertaForCL, BertForCL
from diffaug.trainers import CLTrainer
from utils import write_eval_args, TEMPLATES, change_templates, get_encoded_bs_and_es

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_train: bool = field(
        default=True,
        metadata={
            "help": "Whether use MLP during training"
        }   
    )
    mlp_eval: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }   
    )

    # PromptBERT's args
    apply_prompt: bool = field(
        default=False,
        metadata={
            "help": "Whether use prompt to get sentence embedding"
        }   
    )
    prompt_template_id: str = field(
        default="0",
        metadata={
            "help": "template id"
        }   
    )
    prompt_template: str = field(
        init=False,
        metadata={"help": "Will be initialized in __post_init__"}   
    )
    apply_template_delta_train: bool = field(
        default=False,
        metadata={"help": "Whether use delta during training."}   
    )
    apply_template_delta_infer: bool = field(
        default=False,
        metadata={"help": "Whether use delta during evaluation."}   
    )

    # Ours
    use_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix-tuning style deep prompt in the model"}   
    )
    prefix_len: int = field(
        default=0,
        metadata={"help": "length of prefix"}
    )
    sup_label_num: int = field(
        default=0,
        metadata={"help": "decide the output shape of sup classifier"}
    )
    use_aux_loss: bool = field(
        default=False,
        metadata={"help": "Whether use CE when doing CL"}   
    )
    aux_weight: float = field(
        default=0.001,
        metadata={"help": "CL loss + alpha * CE loss"}   
    )

    def __post_init__(self):
        if self.apply_prompt:
            self.prompt_template = TEMPLATES[self.prompt_template_id]
        else:
            self.prompt_template = None
            self.apply_template_delta_train = False
            self.apply_template_delta_infer = False


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    # Ours
    sup_data_sample_ratio: float = field(
        default=1., 
        metadata={"help": "How many percent of nli data will be sampled"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )
    use_two_optimizers: bool = field(
        default=False,
        metadata={"help": "Necessary for adversarial training"}
    )
    use_two_datasets: bool = field(
        default=False,
        metadata={"help": "for semi-supervised learning"}
    )
    sup_learning_rate: float=field(
        default=1e-5,
        metadata={"help": "learning rate for optimizer2."}
    )
    per_device_sup_train_batch_size: int=field(
        default=8,
        metadata={"help": "batch size for labeled data"}
    )
    phase1_steps: int=field(
        default=0,
        metadata={"help": ""}
    )
    sup_bs_multi: int=field(
        default=1,
        metadata={"help": ""}
    )
    lr1_decay_steps: int=field(
        default=0,
        metadata={"help": "0 means use default"}
    )
    lr2_decay_steps: int=field(
        default=0,
        metadata={"help": "0 means use default"}
    )
    num_train_sup_epochs: int=field(
        default=0,
        metadata={"help": "0 means use default"}
    )
    data_split_seed: int=field(
        default=0,
        metadata={"help": "0 means use default"}
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    use_json = False
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        use_json = True
        all_args = json.loads(Path(sys.argv[-1]).read_text())
        model_args, data_args, training_args = parser.parse_dict(args=all_args)
    elif len(sys.argv) == 3 and sys.argv[-1].endswith(".json"):
        use_json = True
        all_args = json.loads(Path(sys.argv[-1]).read_text())
        all_args["local_rank"] = int(sys.argv[1].split("=")[-1])
        model_args, data_args, training_args = parser.parse_dict(args=all_args)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    random.seed(training_args.data_split_seed)

    #--------------------------
    # Prepare dataset
    #--------------------------
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]

    if training_args.use_two_datasets:
        data_files["train"] = data_args.train_file
        if extension == "txt":
            datasets = load_dataset("text", data_files=data_files, cache_dir="./data/")
        elif extension == "csv":
            datasets = load_dataset("csv", data_files=data_files, cache_dir="./data/")
        else:
            raise NotImplementedError

        nli_data_files = {}
        nli_data_files["train"] = "data/nli_for_simcse.csv"
        sup_datasets = load_dataset("csv", data_files=nli_data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
        
        sample_ratio = data_args.sup_data_sample_ratio
        if sample_ratio <= 1:
            picked_idx = list()
            for i in range(len(sup_datasets["train"])):
                if random.random() < sample_ratio:
                    picked_idx.append(i)
            sup_datasets = sup_datasets["train"].select(picked_idx)
            if extension == "csv":
                datasets["train"] = datasets["train"].select(picked_idx)

    else:
        if extension == "txt":
            extension = "text"
        if extension == "csv":
            datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
            sample_ratio = data_args.sup_data_sample_ratio
            if sample_ratio <= 1:
                picked_idx = list()
                for i in range(len(datasets["train"])):
                    if random.random() < sample_ratio:
                        picked_idx.append(i)
                datasets["train"] = datasets["train"].select(picked_idx)
        else:
            datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")
    #--------------------------

    # Set seed before initializing model.
    set_seed(training_args.seed)

    #--------------------------
    # Prepare tokenizer
    #--------------------------
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
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
    #--------------------------


    #--------------------------
    # Prepare hard prompt
    #--------------------------
    if model_args.apply_prompt:

        bs_tokens, es_tokens, template_tokens, model_args.enc_bs, model_args.enc_es, model_args.enc_template = \
            get_encoded_bs_and_es(model_args.prompt_template, tokenizer)
        model_args.mask_token_id = tokenizer.mask_token_id
        model_args.pad_token_id = tokenizer.pad_token_id

        logger.info(
            f"\n\ntemplate: {model_args.prompt_template}\n" + 
            f"template tokens: {template_tokens}\n" + 
            f"template (encoded): {model_args.enc_template}\n" +
            f"template (decoded): {tokenizer.decode(model_args.enc_template)}\n\n" + 
            f"bs tokens: {tokenizer.cls_token + bs_tokens}\n" + 
            f"bs (encoded): {model_args.enc_bs}\n" + 
            f"es tokens: {es_tokens + tokenizer.sep_token}\n" + 
            f"es (encoded): {model_args.enc_es}\n\n"
        )
        assert len(model_args.enc_template) == len(model_args.enc_bs) + len(model_args.enc_es)
        assert model_args.enc_template == model_args.enc_bs + model_args.enc_es
        # assert len(model_args.prompt_template.split("_"))-1 == len(model_args.enc_template)
    #--------------------------


    #--------------------------
    # Prepare model
    #--------------------------
    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    #--------------------------

    #--------------------------
    # Prepare features
    column_names = datasets["train"].column_names
    if training_args.use_two_datasets:
        sup_column_names = sup_datasets.column_names
        if len(sup_column_names) > 3:
            sup_column_names = sup_column_names[:3]
    else:
        sup_column_names = None

    def prepare_features(examples):
        return prepare_features_(examples, column_names)

    def prepare_features_sup(examples):
        return prepare_features_(examples, sup_column_names)

    def prepare_features_(examples, column_names):
        sent2_cname = None
        if len(column_names) == 2:
            sent0_cname = column_names[0]
            sent1_cname = column_names[1]
        elif len(column_names) == 3:
            sent0_cname = column_names[0]
            sent1_cname = column_names[1]
            sent2_cname = column_names[2]
        elif len(column_names) == 1:
            sent0_cname = column_names[0]
            sent1_cname = column_names[0]
        else:
            raise NotImplementedError

        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        
        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        if model_args.apply_prompt:
            sent_features = {'input_ids': [], 'attention_mask': []}
            for i, s in enumerate(sentences):
                s = tokenizer.encode(s, add_special_tokens=False)[:data_args.max_seq_length]
                sent_features['input_ids'].append(model_args.enc_bs+s+model_args.enc_es)

            ml = max([len(i) for i in sent_features['input_ids']])
            for i in range(len(sent_features['input_ids'])):
                enc_sent = sent_features['input_ids'][i]
                sent_features['input_ids'][i] = enc_sent + [tokenizer.pad_token_id]*(ml-len(enc_sent))
                sent_features['attention_mask'].append(len(enc_sent)*[1] + (ml-len(enc_sent))*[0])
        else:
            sent_features = tokenizer(
                sentences,
                max_length=data_args.max_seq_length,
                truncation=True,
                # padding="max_length" if data_args.pad_to_max_length else False,
                padding="max_length",
            )

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            
        return features
    #--------------------------


    if training_args.do_train:
        logger.info("Preparing for main dataset...\n")
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if training_args.use_two_datasets:
            logger.info("Preparing for additional dataset...\n")
            sup_train_dataset = sup_datasets.map(
                prepare_features_sup,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=sup_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        else:
            sup_train_dataset = None

    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        
        def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        train_sup_dataset=sup_train_dataset if training_args.use_two_datasets else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args

    # Write evaluation JSON
    if os.path.exists(training_args.output_dir) and use_json:
        if is_main_process(training_args.local_rank):
            json_log = dict()
            json_log["train"] = all_args
            # json_log["eval"] = write_eval_args(model_args, training_args)
            json_log["results"] = dict()
            json_log_path = os.path.join(training_args.output_dir, "exp_log.json")
            with open(json_log_path, "w") as f:
                json.dump(json_log, f, ensure_ascii=False, indent=4)

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results, sts_task_names, sts_scores = trainer.evaluate(eval_senteval_transfer=False, predict_mode=True)

        if dist.is_initialized():
            output_ls = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(output_ls, sts_scores)

            if float(output_ls[0][-1]) > float(output_ls[1][-1]):
                sts_scores = output_ls[0]
            else:
                sts_scores = output_ls[1]

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results



if __name__ == "__main__":
    main()