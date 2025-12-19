'''
Author: zhangshd
Date: 2024-08-16 11:18:15
LastEditors: zhangshd
LastEditTime: 2024-08-17 19:34:58
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import R2Score, MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError
from torchmetrics import Accuracy, MatthewsCorrCoef, F1Score, AUROC
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import warnings
from module.module_utils import plot_confusion_matrix, plot_roc_curve, plot_scatter
from module.module_utils import group_model_params, DWALoss
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, balanced_accuracy_score
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    
)

# PCGrad imports
from CGCNN_MT.pcgrad import PCGrad, get_shared_params

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")

class MInterface(pl.LightningModule):
    def __init__(self, model: nn.Module, normalizers, **kwargs):
        super().__init__()
        self.model = model
        self.normalizers = normalizers
        self.save_hyperparameters(ignore=['model', 'datamodule'])
        # focal_alpha = self.hparams.get('focal_alpha', 0.25)
        # focal_gamma = self.hparams.get('focal_gamma', 4)
        num_atoms_per_crystal = 18

        self.example_input_array = {
                'input': {
                    'extra_fea': torch.Tensor(self.hparams.batch_size, self.hparams.orig_extra_fea_len),
                }
            }
        if hasattr(self.hparams, 'orig_atom_fea_len'):

            self.example_input_array['input'].update({
                    'atom_fea': torch.Tensor(self.hparams.batch_size*num_atoms_per_crystal, self.hparams.orig_atom_fea_len),
                    'nbr_fea': torch.Tensor(self.hparams.batch_size*num_atoms_per_crystal, self.hparams.max_num_nbr, self.hparams.nbr_fea_len),
                    'nbr_fea_idx': torch.LongTensor(self.hparams.batch_size*num_atoms_per_crystal*([list(range(self.hparams.max_num_nbr))])),
                    'crystal_atom_idx': [torch.LongTensor(list(range(num_atoms_per_crystal))) for _ in range(self.hparams.batch_size)],
                })
        if self.hparams.model_name in ['cgcnn_uni_atom']:

            self.example_input_array['input'].update({
                    'uni_idx': self.hparams.batch_size*[[[0 for _ in range(num_atoms_per_crystal//2)], [1 for _ in range(num_atoms_per_crystal//2)]]],  # 2 unique atom per crystal
                    'uni_count': self.hparams.batch_size*[np.array([num_atoms_per_crystal/2, num_atoms_per_crystal/2])],
                })
            self.example_input_array['input']['atom_fea'] = torch.randint(0, 118, size=(self.hparams.batch_size*num_atoms_per_crystal,))

        self.task_weights = [w * len(self.hparams.tasks)/sum(self.hparams.task_weights) for w in self.hparams.task_weights] if self.hparams.task_weights is not None else [1.0 for _ in self.hparams.tasks]

        self.collections_init(split='val')
        self.collections_init(split='test')
        self.best_metric = 0.0
        self.best_epoch = 0
        self.best_model_path = None
        
        if self.hparams.ckpt_path is not None:
            ckpt = torch.load(self.hparams.ckpt_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {self.hparams.ckpt_path}")

        self.configure_criterion()
        
        # PCGrad configuration
        self._pcgrad_enabled = getattr(self.hparams, 'pcgrad_enabled', False)
        self._pcgrad_on_shared_only = getattr(self.hparams, 'pcgrad_on_shared_only', True)
        self._pcgrad_head_names = getattr(self.hparams, 'pcgrad_head_names', ["fc_outs", "task_attentions"])
        self._pcgrad_generator = None  # Will be initialized on first use with random_seed
        self._pcgrad_helper = None  # Will be initialized when shared params are known
        
        if self._pcgrad_enabled:
            print(f"[PCGrad] Enabled. on_shared_only={self._pcgrad_on_shared_only}")


    def collections_init(self, split='val'):

        if split == 'test':
            self.test_logits = [[] for _ in range(len(self.hparams.tasks))]
            self.test_preds = [[] for _ in range(len(self.hparams.tasks))]
            self.test_labels = [[] for _ in range(len(self.hparams.tasks))]
            self.test_cifids = [[] for _ in range(len(self.hparams.tasks))]

        elif split == 'val':
            self.val_logits = [[] for _ in range(len(self.hparams.tasks))]
            self.val_preds = [[] for _ in range(len(self.hparams.tasks))]
            self.val_labels = [[] for _ in range(len(self.hparams.tasks))]
            self.val_cifids = [[] for _ in range(len(self.hparams.tasks))]
        else:
            raise ValueError(f"Unsupported split: {split}")

    def configure_criterion(self):
        
        self.criterions = {}
        for task_tp, task in zip(self.hparams.task_types, self.hparams.tasks):
            if 'classification' in task_tp:
                try:
                    n_classes = int(task_tp.split('_')[-1])
                except Exception:
                    n_classes = 2

                self.criterions[task] = {"loss": nn.NLLLoss()}
                self.criterions[task]["metrics"] = [
                    Accuracy(num_classes=n_classes, task="multiclass"),
                    MatthewsCorrCoef(num_classes=n_classes, task="multiclass"),
                    F1Score(num_classes=n_classes, task="multiclass"),
                    AUROC(num_classes=n_classes, task="multiclass"),
                ]
            elif task_tp =='regression':
                self.criterions[task] = {"loss": nn.MSELoss()}
                self.criterions[task]["metrics"] = [
                    R2Score(),
                    MeanAbsoluteError(),
                    MeanAbsolutePercentageError(),
                    MeanSquaredError(),
                ]

            else:
                raise ValueError(f"Unsupported task type: {task_tp}")
        if self.hparams.loss_aggregation == 'dwa':
            self.dwaloss = DWALoss(len(self.hparams.tasks), temp=self.hparams.dwa_temp, 
                                   alpha=self.hparams.dwa_alpha, init_weights=None)
            self.dwaloss.to(self.device)
            self.task_weights = self.dwaloss.weights

    def on_epoch_end(self):
        for k, v in self.criterions.items():
            for metric in v["metrics"]:
                metric.reset()

    def forward(self, input):
        return self.model(**input)
    
    @property
    def automatic_optimization(self) -> bool:
        """
        Disable automatic optimization when PCGrad is enabled to allow
        manual gradient manipulation while keeping baseline unchanged when OFF.
        """
        return not self._pcgrad_enabled

    def _init_pcgrad_helper(self):
        """Initialize PCGrad helper with shared parameters (lazy initialization)."""
        if self._pcgrad_helper is None:
            shared_params, shared_names = get_shared_params(self.model, self._pcgrad_head_names)
            
            # Initialize generator with the random seed for reproducibility
            seed = getattr(self.hparams, 'random_seed', 42)
            self._pcgrad_generator = torch.Generator()
            self._pcgrad_generator.manual_seed(seed)
            
            self._pcgrad_helper = PCGrad(shared_params, generator=self._pcgrad_generator)
            self._shared_param_names = shared_names
            print(f"[PCGrad] Initialized with {len(shared_params)} shared parameters")

    def training_step(self, batch, batch_idx):
        self.model.train()
        
        if not self._pcgrad_enabled:
            # Baseline path: unchanged behavior
            loss = self._step(batch, batch_idx, split='train')
            return loss
        else:
            # PCGrad path: manual optimization with gradient surgery
            return self._training_step_pcgrad(batch, batch_idx)
    
    def _training_step_pcgrad(self, batch, batch_idx):
        """
        Training step with PCGrad gradient surgery.
        
        This method:
        1. Computes per-task weighted losses
        2. Gets per-task gradients on shared parameters
        3. Applies PCGrad projection to resolve conflicts
        4. Runs normal backward for full loss (keeps head grads intact)
        5. Overwrites shared param grads with PCGrad-combined grads
        6. Performs optimizer step
        """
        # Initialize PCGrad helper if needed
        self._init_pcgrad_helper()
        
        # Get optimizer and scheduler
        opt = self.optimizers()
        
        # Forward pass
        outputs, last_layer_feas = self.model(**batch)
        task_ids = batch['task_id']
        targets = batch['targets']
        
        # Compute per-task weighted losses and collect gradients
        per_task_weighted_losses = []
        per_task_grads = []
        valid_task_indices = []
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for task_id, task in enumerate(self.hparams.tasks):
            mask = (task_ids == task_id)
            if mask.sum() == 0:
                continue
            
            valid_task_indices.append(task_id)
            target_i = targets[mask]
            output_i = outputs[task_id][mask]
            
            # Normalize target for loss computation
            if "classification" in self.hparams.task_types[task_id]:
                target_i_normed = target_i.squeeze(-1).long()
            else:
                target_i_normed = self.normalize(target_i, task_id)
            
            # Compute raw loss
            loss_i = self.criterions[task]["loss"](output_i, target_i_normed)
            
            # Log individual task loss
            self.log(f'{task}/train_loss', loss_i, 
                     prog_bar=True, on_step=True, on_epoch=True, 
                     sync_dist=True, batch_size=self.hparams.batch_size)
            
            # Compute weighted loss (Option A: use weighted losses for PCGrad)
            if self.hparams.loss_aggregation == "trainable_weight_sum" and len(self.hparams.task_types) > 1:
                precision = torch.exp(-self.model.log_vars[task_id])
                weighted_loss_i = precision * loss_i + torch.log(torch.exp(self.model.log_vars[task_id]) + 1)
                self.log(f'{task}/train_loss_weight', self.model.log_vars[task_id], 
                         prog_bar=False, on_step=True, on_epoch=True, 
                         sync_dist=True, batch_size=self.hparams.batch_size)
            elif self.hparams.loss_aggregation == "sample_weight_sum" and len(self.hparams.task_types) > 1:
                weighted_loss_i = loss_i * (mask.sum() / mask.shape[0])
            elif self.hparams.loss_aggregation in ["fixed_weight_sum", "dwa"]:
                weighted_loss_i = loss_i * self.task_weights[task_id]
                self.log(f'{task}/train_loss_weight', self.task_weights[task_id], 
                         prog_bar=False, on_step=True, on_epoch=True, 
                         sync_dist=True, batch_size=self.hparams.batch_size)
            else:
                weighted_loss_i = loss_i
            
            per_task_weighted_losses.append(weighted_loss_i)
            total_loss = total_loss + weighted_loss_i
            
            # Compute gradients on shared parameters for this task
            if self._pcgrad_on_shared_only:
                task_grads = self._pcgrad_helper.compute_task_grads(
                    weighted_loss_i, retain_graph=True, create_graph=False
                )
                per_task_grads.append(task_grads)
        
        # Log total loss
        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True, 
                 sync_dist=True, batch_size=self.hparams.batch_size)
        
        # Apply PCGrad if we have multiple tasks with gradients
        if len(per_task_grads) > 1 and self._pcgrad_on_shared_only:
            # Combine gradients using PCGrad
            combined_grads, pcgrad_stats = self._pcgrad_helper.combine_grads(per_task_grads)
            
            # Log PCGrad statistics
            self.log('pcgrad/conflict_rate', pcgrad_stats['conflict_rate'], 
                     prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            self.log('pcgrad/mean_cosine_sim', pcgrad_stats['mean_cosine_sim'], 
                     prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            
            # Log per-task gradient norms
            if 'grad_norms' in pcgrad_stats:
                for i, (task_idx, norm) in enumerate(zip(valid_task_indices, pcgrad_stats['grad_norms'])):
                    task = self.hparams.tasks[task_idx]
                    self.log(f'pcgrad/{task}_grad_norm', norm, 
                             prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            
            # Zero gradients before backward
            opt.zero_grad()
            
            # Run normal backward for full loss (this sets head grads correctly)
            self.manual_backward(total_loss)
            
            # Overwrite shared parameter gradients with PCGrad-combined gradients
            self._pcgrad_helper.set_grads(combined_grads)
        else:
            # Single task or PCGrad disabled on shared: normal backward
            opt.zero_grad()
            self.manual_backward(total_loss)
        
        # Gradient clipping if configured
        if hasattr(self.hparams, 'gradient_clip_val') and self.hparams.gradient_clip_val is not None:
            self.clip_gradients(opt, gradient_clip_val=self.hparams.gradient_clip_val)
        
        # Optimizer step
        opt.step()
        
        # Scheduler step (if using step-based scheduler)
        sch = self.lr_schedulers()
        if sch is not None:
            if isinstance(sch, dict):
                sch = sch['scheduler']
            # Only step for schedulers that operate on step (not epoch)
            if hasattr(self, 'scheduler') and isinstance(self.scheduler, dict):
                if self.scheduler.get('interval', 'epoch') == 'step':
                    sch.step()
        
        return total_loss

    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self._step(batch, batch_idx, split='val')
    
    def test_step(self, batch, batch_idx):
        self.model.eval()
        self._step(batch, batch_idx, split='test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        outputs, last_layer_feas = self.model(**batch)
        processed_outputs = {}
        for i, output in enumerate(outputs):
            if self.hparams.task_types[i] == 'regression':
                processed_outputs[f"{self.hparams.tasks[i]}_pred"] = self.denormalize(output, task_id=i)
            elif 'classification' in self.hparams.task_types[i]:
                output = torch.exp(output)
                processed_outputs[f"{self.hparams.tasks[i]}_pred"] = torch.argmax(output, dim=1, keepdim=True)
                processed_outputs[f"{self.hparams.tasks[i]}_prob"] = output
            processed_outputs[f"{self.hparams.tasks[i]}_last_layer_fea"] = last_layer_feas[i]
        return processed_outputs
    
    def normalize(self, input, task_id):
        self.normalizer = self.normalizers[task_id]
        if self.normalizer.device.type != self.device.type:
            self.normalizer.to(self.device)
        input_norm = self.normalizer.norm(input)
        return input_norm

    def denormalize(self, output, task_id):
        self.normalizer = self.normalizers[task_id]
        if self.normalizer.device.type != self.device.type:
            self.normalizer.to(self.device)
        output_denorm = self.normalizer.denorm(output)
        return output_denorm

    def _step(self, batch, batch_idx, split='val'):
        
        outputs, last_layer_feas = self.model(**batch)
        task_ids = batch['task_id']
        targets = batch['targets']
        cifids = batch['cif_id']

        metrics = {}
        merged_metric = torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        valid_task_indices = []
        for task_id, task in enumerate(self.hparams.tasks):   
            mask = (task_ids == task_id)
            if mask.sum() == 0:
                continue
            valid_task_indices.append(task_id)
            target_i = targets[mask]
            output_i = outputs[task_id][mask]
            cifids_i = np.array(cifids)[mask.tolist()]
            
            output_i_prob = torch.exp(torch.clamp(output_i, min=-20, max=20))
            if "classification" in self.hparams.task_types[task_id]:
                target_i_normed = target_i.squeeze(-1).long()
                target_i = target_i.squeeze(-1).long()
                output_i_denorm = torch.argmax(output_i, dim=1)
            else:
                target_i_normed = self.normalize(target_i, task_id)
                output_i_denorm = self.denormalize(output_i, task_id)
            loss_i = self.criterions[task]["loss"](output_i, target_i_normed)
            self.log(f'{task}/{split}_loss', loss_i, 
                     prog_bar=True, on_step=True, on_epoch=True, 
                     sync_dist=True, batch_size=self.hparams.batch_size)
            
            if self.hparams.loss_aggregation == "trainable_weight_sum" and len(self.hparams.task_types) > 1:
                
                precision = torch.exp(-self.model.log_vars[task_id])
                loss += (precision * loss_i + torch.log(torch.exp(self.model.log_vars[task_id]) + 1))
                
                self.log(f'{task}/{split}_loss_weight', self.model.log_vars[task_id], 
                     prog_bar=False, on_step=True, on_epoch=True, 
                     sync_dist=True, batch_size=self.hparams.batch_size)
            elif self.hparams.loss_aggregation == "sample_weight_sum" and len(self.hparams.task_types) > 1:
                loss += loss_i*(mask.sum()/mask.shape[0])
            elif self.hparams.loss_aggregation in ["fixed_weight_sum", "dwa"]:
                loss += loss_i*self.task_weights[task_id]
                self.log(f'{task}/{split}_loss_weight', self.task_weights[task_id], 
                     prog_bar=False, on_step=True, on_epoch=True, 
                     sync_dist=True, batch_size=self.hparams.batch_size)
            else:
                loss += loss_i
            
            if split == 'train':
                continue
            for metric_func in self.criterions[task]["metrics"]:
                metric_name = metric_func._get_name()
                metric_func.to(self.device)
                if mask.sum() < 2:
                    metrics[f"{task}_{metric_name}"] = torch.tensor(0.0, device=self.device)
                    continue
                if metric_name.endswith("AUROC"):
                    metric_value = metric_func(output_i_prob, target_i).float()
                else:
                    metric_value = metric_func(output_i_denorm, target_i).float()
                metrics[f"{task}/{split}_{metric_name}"] = metric_value
                self.log(f'{task}/{split}_{metric_name}', metric_value, 
                         prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
                if split == 'val' and metric_name.endswith("R2Score"):
                    merged_metric += metric_value * mask.sum()/mask.shape[0]
                    # print(f"{task}/{split}_{metric_name}: {metric_value:.4f}")
                elif split == 'val' and metric_name.endswith("Accuracy"):
                    merged_metric += metric_value * mask.sum()/mask.shape[0]
                    # print(f"{task}/{split}_{metric_name}: {metric_value:.4f}")
                    
            ## collect predictions and labels for test set
            if split == 'test':
                self.test_logits[task_id] += output_i_prob.tolist()
                self.test_preds[task_id] += output_i_denorm.tolist()
                self.test_labels[task_id] += target_i.tolist()
                self.test_cifids[task_id] += cifids_i.tolist()
            elif split == 'val':
                self.val_logits[task_id] += output_i_prob.tolist()
                self.val_preds[task_id] += output_i_denorm.tolist()
                self.val_labels[task_id] += target_i.tolist()
                self.val_cifids[task_id] += cifids_i.tolist()
        # if self.hparams.loss_aggregation == "dwa":
        #     loss = self.dwaloss(losses, valid_task_indices, split)
        #     for i, task_id in enumerate(valid_task_indices):
        #         task = self.hparams.tasks[task_id]
        #         if split == 'train':
        #             self.log(f'{task}/{split}_loss_weight', self.dwaloss.weights[i], 
        #                     prog_bar=False, on_step=True, on_epoch=True, 
        #                     sync_dist=True, batch_size=self.hparams.batch_size)

        self.log(f'{split}_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        if split == 'val':
            self.log(f'{split}_MergedMetric', merged_metric, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
            # print(f"{split}_MergedMetric: {merged_metric:.4f}")
            # print(f'{split}_loss: {loss:.4f}')
            # print('-'*50)
        return loss
    
    def configure_optimizers(self):
        if hasattr(self.hparams, 'optim_config') and self.hparams.optim_config == "fine":
            return self.configure_optimizers_fine()
        else:
            return self.configure_optimizers_coarse()
            
    def configure_optimizers_coarse(self):
        if hasattr(self.hparams, 'weight_decay'):
            self.weight_decay = self.hparams.weight_decay
        else:
            self.weight_decay = 0
        if hasattr(self.hparams, 'group_lr') and self.hparams.group_lr:
            optimizer_grouped_parameters = group_model_params(self)
        else:
            optimizer_grouped_parameters = self.model.parameters()

        if self.hparams.optim.lower() == 'sgd':
            self.optimizer = optim.SGD(optimizer_grouped_parameters, self.hparams.lr,
                                momentum=self.hparams.momentum,
                                weight_decay=self.weight_decay)
        elif self.hparams.optim.lower() == 'adam':
            self.optimizer = optim.Adam(optimizer_grouped_parameters, self.hparams.lr,
                                weight_decay=self.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')
        
        if hasattr(self.hparams, 'lr_record') and len(self.hparams.lr_record.keys()) == 1:
            print("Using lr_record from hparams")
            def lr_lambda(epoch):
                lr_record = list(self.hparams.lr_record.values())[0]
                lr = lr_record[epoch+1]
                lr_lambda = lr/self.hparams.lr
                return lr_lambda
            self.scheduler = {
            'scheduler': lrs.LambdaLR(self.optimizer, lr_lambda, verbose=True),
            'interval': 'epoch',
            'frequency': 1
                    }
            
            return [self.optimizer], [self.scheduler]

        elif self.hparams.lr_scheduler is None:
            return self.optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                self.scheduler = lrs.StepLR(self.optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate, verbose=True)
            elif self.hparams.lr_scheduler == 'cosine':
                self.scheduler = lrs.CosineAnnealingLR(self.optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr, verbose=True)
            elif self.hparams.lr_scheduler == 'multi_step':
                self.scheduler = lrs.MultiStepLR(self.optimizer, milestones=self.hparams.lr_milestones,
                            gamma=self.hparams.lr_decay_rate, verbose=True)
            elif self.hparams.lr_scheduler == 'reduce_on_plateau':
                self.scheduler = lrs.ReduceLROnPlateau(self.optimizer, mode='max', factor=self.hparams.lr_decay_rate, 
                                                       patience=self.hparams.lr_decay_steps, min_lr=self.hparams.lr_decay_min_lr, verbose=True)
                return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "val_Metric"}
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [self.optimizer], [self.scheduler]
        
    def configure_optimizers_fine(self):
        
        lr = self.hparams.lr
        wd = self.hparams.weight_decay

        lr_end = self.hparams.lr_decay_min_lr
        lr_mult = self.hparams.lr_mult
        decay_power = self.hparams.decay_power
        
        if hasattr(self.hparams, 'group_lr') and self.hparams.group_lr:
            optimizer_grouped_parameters = group_model_params(self)
        else:
            optimizer_grouped_parameters = self.model.parameters()

        if self.hparams.optim == "adamw":
            optimizer = optim.AdamW(
                optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
            )
        elif self.hparams.optim == "adam":
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif self.hparams.optim == "sgd":
            optimizer = optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optim}")

        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps

        warmup_steps = self.hparams.warmup_steps
        if isinstance(self.hparams.warmup_steps, float):
            warmup_steps = int(max_steps * warmup_steps)

        print(
            f"max_epochs: {self.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
            f"lr_mult : {lr_mult} | weight_decay : {wd} | decay_power : {decay_power}"
        )

        if decay_power == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        elif decay_power == "constant":
            scheduler = get_constant_schedule(
                optimizer,
            )
        elif decay_power == "constant_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                lr_end=lr_end,
                power=decay_power,
            )

        sched = {"scheduler": scheduler, "interval": "step"}
        self.scheduler = sched
        self.optimizer = optimizer

        return (
            [self.optimizer],
            [self.scheduler],
        )

    def on_train_epoch_end(self):
        ## if using reduce_on_plateau, update lr_scheduler
        if isinstance(self.scheduler, lrs.ReduceLROnPlateau):
            self.scheduler.step(self.trainer.callback_metrics['val_Metric'])
        if self.hparams.loss_aggregation == "dwa":
            if all(self.trainer.callback_metrics[f'{task}/train_loss_epoch'] for task in self.hparams.tasks):
                epoch_losses = [self.trainer.callback_metrics[f'{task}/train_loss_epoch'].item() for task in self.hparams.tasks]
                self.task_weights = self.dwaloss(epoch_losses)

    def on_validation_epoch_end(self):
        self._epoch_eval(self.val_labels, self.val_preds, self.val_logits, self.val_cifids, 'val')
        self.collections_init(split='val')

    def on_test_epoch_end(self):
        self._epoch_eval(self.test_labels, self.test_preds, self.test_logits, self.test_cifids, 'test')
        self.collections_init(split='test')

    def log_regression_results(self, labels, preds, cifids, split, task_id, r2, mae):
        
        task = self.hparams.tasks[task_id]
        if self.logger.log_dir is None:
            return
        
        csv_file = os.path.join(self.logger.log_dir, f"{split}_results_{task}.csv")
        df_results = pd.DataFrame(
                {
                    "CifId": cifids[task_id],
                    "GroundTruth": np.array(labels[task_id]).squeeze().tolist(),
                    "Predicted": np.array(preds[task_id]).squeeze().tolist(),
                }
            )
        df_results["Error"] = (df_results["GroundTruth"] - df_results["Predicted"]).abs()
        df_results.sort_values(by="Error", inplace=True, ascending=False)
        df_results.to_csv(csv_file, index=False)

        img_file = os.path.join(self.logger.log_dir, f"{split}_scatter_{task}.png")
        ax = plot_scatter(
            np.array(labels[task_id]),
            np.array(preds[task_id]),
            title=f"{split}/scatter_{task}(epoch_{self.current_epoch})",
            metrics={"R2": r2, "MAE": mae},
            outfile=img_file,
        )
        fig = ax.figure
        self.logger.experiment.add_figure(f'{split}/scatter_{task}', fig, self.current_epoch)

    def log_classification_results(self, labels, preds, logits, cifids, split, task_id, auc_score):
        task = self.hparams.tasks[task_id]
        if self.logger.log_dir is None:
            return
        csv_file = os.path.join(self.logger.log_dir, f"{split}_results_{task}.csv")
        df_results = pd.DataFrame(
                {
                    "CifId": cifids[task_id],
                    "GroundTruth": np.array(labels[task_id]).squeeze().tolist(),
                    "Predicted": np.array(preds[task_id]).squeeze().tolist(),
                    "Prob": np.array(logits[task_id]).squeeze().tolist(),
                }
            )
        df_results.to_csv(csv_file, index=False)
        
        cm = confusion_matrix(np.array(labels[task_id]), np.array(preds[task_id]))
        img_file = os.path.join(self.logger.log_dir, f"{split}_confusion_matrix_{task}.png")
        ax = plot_confusion_matrix(
            cm,
            title=f"{split}/confusion_matrix_{task}(epoch_{self.current_epoch})",
            outfile=img_file,
        )
        fig = ax.figure
        self.logger.experiment.add_figure(f'{split}/confusion_matrix_{task}', fig, self.current_epoch)
        
        if len(logits[task_id][0]) == 2:
            fpr, tpr, thresholds = roc_curve(
                np.array(labels[task_id]), np.array(logits[task_id])[:,1], 
                drop_intermediate=False
            )
            img_file = os.path.join(self.logger.log_dir, f"{split}_roc_curve_{task}.png")
            ax = plot_roc_curve(
                fpr,
                tpr,
                auc_score,
                title=f"{split}/roc_curve_{task}(epoch_{self.current_epoch})",
                outfile=img_file,
            )
            fig = ax.figure
            self.logger.experiment.add_figure(f'{split}/roc_curve_{task}', fig, self.current_epoch)


    def _epoch_eval(self, labels, preds, logits, cifids, split):
        
        monitor_metric = 0
        all_metrics = {}
        total_num = sum([len(p) for p in preds])
        # print("-"*50)
        # print(f"Epoch {self.current_epoch}: Evaluating {split} ({total_num})...")
        for task_id, task in enumerate(self.hparams.tasks):
            # print(f"Evaluating {split}_{task} ({len(preds[task_id])})...")
            if len(preds[task_id]) == 0:
                continue
            if self.hparams.task_types[task_id] == "regression":
                r2 = r2_score(np.array(labels[task_id]), np.array(preds[task_id]))
                mae = mean_absolute_error(np.array(labels[task_id]), np.array(preds[task_id]))
                
                ## use R2 score as the main metric for regression tasks
                metric = r2

                all_metrics.update({
                    f"{task}/{split}_R2Score": r2,
                    f"{task}/{split}_MeanAbsoluteError": mae,
                                    })
                if split == 'test':
                    self.log_regression_results(labels, preds, cifids, split, task_id, r2, mae)

            elif "classification" in self.hparams.task_types[task_id]:

                acc = accuracy_score(np.array(labels[task_id]), np.array(preds[task_id]))
                bacc = balanced_accuracy_score(np.array(labels[task_id]), np.array(preds[task_id]))
                f1 = f1_score(np.array(labels[task_id]), np.array(preds[task_id]), average='macro')
                mcc = matthews_corrcoef(np.array(labels[task_id]), np.array(preds[task_id]))
                if len(logits[task_id][0]) == 2:
                    # print(self.hparams.tasks[task_id], labels[task_id])
                    try:
                        auc_score = roc_auc_score(np.array(labels[task_id]), np.array(logits[task_id])[:,1])
                    except Exception:   ## for binary classification, only one class is present in the dataset
                        auc_score = 0.0
                else:
                    try:
                        auc_score = roc_auc_score(np.array(labels[task_id]), np.array(logits[task_id]), multi_class='ovo', average='macro')
                    except Exception:   ## for multi-class classification, only one class is present in the dataset
                        auc_score = 0.0

                ## use accuracy as the main metric for classification tasks
                metric = bacc

                all_metrics.update({
                    f"{task}/{split}_AUROC": auc_score,
                    f"{task}/{split}_Accuracy": acc,
                    f"{task}/{split}_BalancedAccuracy": bacc,
                    f"{task}/{split}_F1Score": f1,
                    f"{task}/{split}_MatthewsCorrCoef": mcc,
                                    })
                if split == 'test':
                    self.log_classification_results(labels, preds, logits, cifids, split, task_id, auc_score)
            
            ## calculate the weighted metric
            if self.hparams.loss_aggregation == "sample_weight_sum" and len(self.hparams.task_types) > 1:
                monitor_metric += metric*(len(preds[task_id])/total_num)
            # elif self.hparams.loss_aggregation == "trainable_weight_sum" and len(self.hparams.task_types) > 1:
            #     precision = torch.exp(-self.model.log_vars[task_id])
            #     monitor_metric += (precision * metric + self.model.log_vars[task_id])
            elif self.hparams.loss_aggregation in ["trainable_weight_sum", "fixed_weight_sum", "dwa"] and len(self.hparams.task_types) > 1:
                monitor_metric += metric*self.hparams.task_weights[task_id]
            else:
                # monitor_metric += (metric/len(self.hparams.tasks))
                monitor_metric += metric*self.hparams.task_weights[task_id]

        if (split == 'val' and monitor_metric - self.best_metric > self.hparams.min_delta) :
            print(f"current_epoch({self.current_epoch}): metric={monitor_metric:.4f} > best_metric={self.best_metric:.4f}, log val results..")
            self.best_metric = monitor_metric
            self.best_epoch = self.current_epoch
            for task_id, task in enumerate(self.hparams.tasks):
                if len(preds[task_id]) == 0:
                    continue
                if self.hparams.task_types[task_id] == "regression":
                    self.log_regression_results(labels, preds, cifids, split, task_id, 
                                                all_metrics[f'{task}/{split}_R2Score'], 
                                                all_metrics[f'{task}/{split}_MeanAbsoluteError']
                                                )
                elif "classification" in self.hparams.task_types[task_id]:
                    self.log_classification_results(labels, preds, logits, cifids, split, task_id, 
                                                    all_metrics[f'{task}/{split}_AUROC']
                                                    )

        self.log(f'{split}_Metric', monitor_metric, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)

        ## save all metrics to a csv file
        if self.current_epoch > 0 and self.logger.log_dir is not None:
            df_metrics = pd.DataFrame(all_metrics, index=[self.current_epoch])
            metrics_file = os.path.join(self.logger.log_dir, f"{split}_metrics.csv")
            df_metrics.to_csv(metrics_file, mode='a', header=not os.path.exists(metrics_file))
            

    