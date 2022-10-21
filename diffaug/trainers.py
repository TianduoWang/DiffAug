import collections
import inspect
import math
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)

from transformers.utils import logging
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from prettytable import PrettyTable
import random

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

# from transformers.trainer import _model_unwrap
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.integrations import WandbCallback, rewrite_logs
import copy
# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock

logger = logging.get_logger(__name__)



class CLTrainer(Trainer):


    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        train_sup_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        ):

        super(CLTrainer, self).__init__(
            model = model, 
            args = args, 
            data_collator = data_collator, 
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            model_init = model_init,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            optimizers = optimizers
            )
        self.train_sup_dataset = train_sup_dataset

    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:

        def batcher(params, batch):
            input_sentences = [' '.join(s) for s in batch]

            #--------------------
            # Add template to eval sentences
            #--------------------
            if self.model_args.apply_prompt:
                eval_template = self.model_args.prompt_template
                template = eval_template.replace('*mask*', self.tokenizer.mask_token )\
                                .replace('_', ' ').replace('*sep+*', '').replace('*cls*', '').strip()
                prompt_sents = []
                for s in input_sentences:
                    prompt_sents.append(template.replace('*sent0*', s).strip())
                sentences = prompt_sents
            else:
                sentences = input_sentences
            #--------------------

            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        def print_table(task_names, scores):
            tb = PrettyTable()
            tb.field_names = task_names
            tb.add_row(scores)
            print(tb)

        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        se = senteval.engine.SE(params, batcher, mode='dev')
        tasks = ['STSBenchmark']
        
        self.model.eval()
        results = se.eval(tasks)
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        metrics = {"eval_stsb_spearman": round(stsb_spearman, 5)}
        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
    
    def _save_checkpoint(self, model, trial, metrics=None):
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]
            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                output_dir = self.args.output_dir
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir
                # Only save model when it is the best one
                self.save_model(output_dir)
                if self.deepspeed:
                    self.deepspeed.save_checkpoint(output_dir)
                # Save the Trainer state
                if self.is_world_process_zero():
                    self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        else:
            # Save model checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is not None and trial is not None:
                if self.hp_search_backend == HPSearchBackend.OPTUNA:
                    run_id = trial.number
                else:
                    from ray import tune
                    run_id = tune.get_trial_id()
                run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
                output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
            else:
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                self.store_flos()
            self.save_model(output_dir)
            if self.deepspeed:
                self.deepspeed.save_checkpoint(output_dir)
            # Save the Trainer state
            if self.is_world_process_zero():
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
    

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps)


    def create_optimizer(self):
        """
        Setup the optimizer.
        Different from the original implementation, we create 2 optimizers in this function
        """
        pnames = [n for n, _ in self.model.named_parameters()]
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                                 if (n.startswith("bert.") or n.startswith("roberta.") or n.startswith("mlp.") 
                                  or n.startswith("prefix_encoder.") or n.startswith("sup_class"))],
                    "weight_decay": self.args.weight_decay,
                }
            ]
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        self.optimizer2 = None
        if self.args.use_two_optimizers:
            optimizer_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                                 if (n.startswith("prefix_encoder.") or n.startswith("sup_class"))],
                    "weight_decay": self.args.weight_decay,
                },
            ]
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            optimizer_kwargs["lr"] = self.args.sup_learning_rate
            self.optimizer2 = AdamW(optimizer_parameters, **optimizer_kwargs)

            n1 = [p for n, p in self.model.named_parameters() if (n.startswith("bert.") or n.startswith("roberta.") or n.startswith("mlp"))]
            n2 = [n for n, _ in self.model.named_parameters() if (n.startswith("prefix_encoder.") or n.startswith("sup_class"))]
            assert len(n1) + len(n2) == len(pnames)

        return self.optimizer


    def create_scheduler(self, num_training_steps: int):
        """
          num_training_steps (int): The number of training steps to do (including stage1 + stage2).
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps-self.args.phase1_steps \
                                    if self.args.lr1_decay_steps == 0 else
                                    self.args.lr1_decay_steps,
            )

        self.lr_scheduler2 = None
        if self.args.use_two_optimizers:
            self.lr_scheduler2 = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer2,
                num_warmup_steps=0,
                num_training_steps=num_training_steps-self.args.phase1_steps \
                                    if self.args.lr2_decay_steps == 0 else
                                    self.args.lr2_decay_steps,
            )

        return self.lr_scheduler


    def _get_train_sampler(self, dataset) -> Optional[torch.utils.data.Sampler]:
        if not isinstance(dataset, collections.abc.Sized):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(dataset, datasets.Dataset):
                lengths = (
                    dataset[self.args.length_column_name]
                    if self.args.length_column_name in dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size,
                dataset=dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=generator,
            )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(dataset, generator=generator)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader [`~torch.utils.data.DataLoader`].
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        train_sampler = self._get_train_sampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        if self.args.use_two_datasets:
            assert self.train_sup_dataset is not None
            sup_train_sampler = self._get_train_sampler(self.train_sup_dataset)
            train_sup_dataloader = DataLoader(
                self.train_sup_dataset,
                batch_size=self.args.per_device_sup_train_batch_size,
                sampler=sup_train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            return train_dataloader, train_sup_dataloader
        else:
            return train_dataloader


    def training_step(
        self, 
        model: nn.Module, 
        inputs: Dict[str, Union[torch.Tensor, Any]],
        stage: str = "1") -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        
        assert stage in ["1", "2"]
        inputs["stage"] = stage

        with self.autocast_smart_context_manager():
            outputs = model(**inputs)
            loss = outputs["loss"]
            if self.state.global_step > 0 and \
                self.state.global_step % (50*self.args.gradient_accumulation_steps) == 0:
                if stage == "1":
                    self.log({"ce_loss": loss.item()})
                else:
                    self.log({"encoder_loss": loss.item()})

        if self.args.gradient_accumulation_steps > 1 and stage == "2":
            # assume only stage 2 needs large batch size
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()
    
    
    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        
        The main difference between ours and Huggingface's original implementation is that we 
        also load model_args when reloading best checkpoints for evaluation.
        """
        self._memory_tracker.start()

        args = self.args
        
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)
            if not self.is_model_parallel:
                model = model.to(self.args.device)

            self.model = model
            self.model_wrapped = model

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        
        # Data loader and number of training steps
        if self.args.use_two_datasets:
            train_dataloader, train_sup_dataloader = self.get_train_dataloader()
        else:
            train_dataloader = self.get_train_dataloader()

        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        
        max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
        if self.args.lr1_decay_steps > 0:
            max_steps = min(max_steps, self.args.lr1_decay_steps)
        # carefully adjust here
        max_steps += self.args.phase1_steps
        num_train_epochs = math.ceil(self.args.num_train_epochs)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        model = self.model_wrapped

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )

        if model is not self.model:
            self.model_wrapped = model

        # Train!
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        if self.args.use_two_datasets:
            logger.info(f"  Num labeled examples = {self.num_examples(train_sup_dataloader)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        if self.args.phase1_steps > 0:
            logger.info(f"  Stage-1 optimization steps = {self.args.phase1_steps}")
            logger.info(f"  Stage-2 optimization steps = {max_steps - self.args.phase1_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        sup_epoch=0
        #------------------------
        # Stage-1
        #------------------------
        if self.args.phase1_steps > 0:
            while True:
                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # Prepare sup data
                sup_inputs_ls = list()
                for i in range(args.sup_bs_multi):
                    try:
                        sup_inputs_ls.append(next(sup_data_iterator))
                    except:
                        if isinstance(train_sup_dataloader.sampler, DistributedSampler):
                            train_sup_dataloader.sampler.set_epoch(sup_epoch)
                        sup_epoch += 1
                        sup_data_iterator = iter(train_sup_dataloader)
                        sup_inputs_ls.append(next(sup_data_iterator))

                if len(sup_inputs_ls) == 1:
                    sup_inputs = sup_inputs_ls[0]
                else:
                    sup_inputs = dict()
                    for key in ["input_ids", "attention_mask"]:
                        sup_inputs[key] = torch.cat([item[key] for item in sup_inputs_ls], dim=0)

                # Make input batch
                inputs = dict()
                inputs["sup_input_ids"] = sup_inputs["input_ids"]
                inputs["sup_attention_mask"] = sup_inputs["attention_mask"]

                loss = self.training_step(model, inputs, stage="1")

                # Prepare optimizer and lr_scheduler
                if self.args.use_two_optimizers:
                    optim = self.optimizer2
                    lr_sche = self.lr_scheduler2
                else:
                    optim = self.optimizer
                    lr_sche = self.lr_scheduler

                # Gradient clipping
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    self.scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                # Optimizer step
                self.scaler.step(optim)
                self.scaler.update()
                lr_sche.step()
                model.zero_grad()

                self.state.global_step += 1
                self.state.epoch = 0
                self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, 0, ignore_keys_for_eval=None)

                if self.state.global_step >= self.args.phase1_steps:
                    break

        #------------------------
        # Stage-2
        #------------------------
        for epoch in range(num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
                if self.args.use_two_datasets:
                    train_sup_dataloader.sampler.set_epoch(sup_epoch)
                    sup_epoch += 1
                    sup_data_iterator = iter(train_sup_dataloader)

            steps_in_epoch = len(train_dataloader) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            assert train_dataset_is_sized, "currently we only support sized dataloader!"

            inputs = None
            last_inputs = None
            for step, inputs in enumerate(train_dataloader):
                
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if self.args.use_two_datasets:
                    try:
                        sup_inputs = next(sup_data_iterator)
                    except:
                        sup_epoch += 1
                        # logger.info(f"sup_epoch: {sup_epoch}")
                        if isinstance(train_dataloader.sampler, DistributedSampler):
                            train_sup_dataloader.sampler.set_epoch(sup_epoch)
                        sup_epoch += 1
                        sup_data_iterator = iter(train_sup_dataloader)
                        sup_inputs = next(sup_data_iterator)

                    if self.model_args.use_aux_loss:
                        self.model.model_args.use_aux_loss = True
                        inputs["sup_input_ids"] = sup_inputs["input_ids"]
                        inputs["sup_attention_mask"] = sup_inputs["attention_mask"]
                # ------------------------- #
                # Train BERT
                # ------------------------- #
                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        loss = self.training_step(model, inputs, stage="2")
                else:
                    loss = self.training_step(model, inputs, stage="2")
                # loss = self.training_step(model, inputs, stage="2")

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):

                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval=None)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    logger.info("Early stop due to long no improvements")
                    break   

                if self.state.global_step > self.args.max_steps:
                    logger.info("Stop due to reaching max steps")
                    break 

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval=None)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed!!!\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint, model_args=self.model_args)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)

#---------------
# End of file
#---------------