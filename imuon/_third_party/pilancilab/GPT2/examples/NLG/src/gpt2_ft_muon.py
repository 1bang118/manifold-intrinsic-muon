#  ------------------------------------------------------------------------------------------
#  Modified version of gpt2_ft.py to use custom Muon optimizer
#  This allows testing if the optimizer is causing the BLEU score discrepancy
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import warnings
import numpy as np
import itertools
import pickle

import torch
import random
from torch.utils.data import DataLoader
torch.set_printoptions(threshold=100000)

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)

# Import original optimizer utilities for scheduler
from optimizer import (
    create_optimizer_scheduler,
    add_optimizer_params,
)

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

import loralib as lora

# Locate the iMuon optimizer package. REPO_ROOT defaults to 5 levels up from
# this file, which is correct for the layout in this repository:
#   <REPO_ROOT>/_third_party/pilancilab/GPT2/examples/NLG/src/gpt2_ft_muon.py
# Override with the REPO_DIR env var if running from a different layout.
REPO_ROOT = os.environ.get(
    'REPO_DIR',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')),
)
sys.path.insert(0, os.path.join(REPO_ROOT, 'swift', 'trainers'))
from optimizers import Muon

parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script with Muon optimizer')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')
parser.add_argument('--trial_name', default="default_trial", type=str, help="trial names for storing info")
parser.add_argument('--valid_data', required=True, help='location of validation data corpus')
parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')
parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')
parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg', 'gpt2.smallGPU'], 
                    help='model names')
parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')
parser.add_argument('--fp16', action='store_true', help='train model with fp16')
parser.add_argument('--log_interval', type=int, default=100, help='log interval')
parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')
parser.add_argument('--save_interval', type=int, default=500, help='save interval')
parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder.')
parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')
parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')
parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], 
                    help='language model training objective')
parser.add_argument('--lora_dropout', default=0.0, type=float, 
                    help='dropout probability for lora layers')
parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')
parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')
parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')
parser.add_argument('--roll_step', type=int, default=100, help='rolling step')
parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')

# Muon-specific arguments
parser.add_argument('--muon_momentum', type=float, default=0.95, help='Muon momentum')
parser.add_argument('--muon_nesterov', action='store_true', help='Use Nesterov momentum (default: enabled)')
parser.add_argument('--no_muon_nesterov', dest='muon_nesterov', action='store_false', help='Disable Nesterov')
parser.set_defaults(muon_nesterov=True)
parser.add_argument('--muon_ns_steps', type=int, default=5, help='Newton-Schulz iteration steps')
parser.add_argument('--muon_lora_precond', action='store_true', help='Enable LoRA Riemannian preconditioning')
parser.add_argument('--muon_lora_precond_eps', type=float, default=1e-6, help='LoRA precond epsilon')
parser.add_argument('--muon_lora_riemannian_muon', action='store_true', help='Enable LoRA Riemannian Muon update')
parser.add_argument('--muon_lora_riemannian_ortho_method', type=str, default='ns', 
                    choices=['ns', 'svd'], help='Orthogonalization method for Riemannian update')
parser.add_argument('--muon_lora_riemannian_adjust_lr', action='store_true',
                    help='Apply Muon LR adjustment to Riemannian updates (default: enabled)')
parser.add_argument('--no_muon_lora_riemannian_adjust_lr', dest='muon_lora_riemannian_adjust_lr',
                    action='store_false', help='Disable LR adjustment')
parser.set_defaults(muon_lora_riemannian_adjust_lr=True)
parser.add_argument('--muon_lora_riemannian_variant', type=str, default='full',
                    choices=['full', 'v2', 'v3', 'v4', 'v5', 'v5_warmup', 'v5_compact', 'v5_compact_warmup'],
                    help='Riemannian Muon variant: full (original), v2 (Ortho then precond), '
                         'v3 (precond then Ortho), v4 (horizontal projection), '
                         'v5 (projected QR variant), v5_warmup (V1 for first 50 steps then V5), '
                         'v5_compact (compact-QR form, Section 3.3), '
                         'v5_compact_warmup (V1 for first 50 steps then V5 compact)')

# Weights & Biases (optional)
parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/team')
parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
parser.add_argument('--wandb_group', type=str, default=None, help='W&B run group')
parser.add_argument('--wandb_tags', type=str, default=None, help='Comma-separated W&B tags')
parser.add_argument('--wandb_mode', type=str, default=None, choices=['online', 'offline', 'disabled'],
                    help='W&B mode (online/offline/disabled)')
parser.add_argument('--wandb_log_interval', type=int, default=None, help='Log interval for W&B (defaults to log_interval)')
parser.add_argument('--deterministic', action='store_true',
                    help='Enable deterministic training (slower, may require CUBLAS_WORKSPACE_CONFIG)')


def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


def _safe_wandb_import():
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    return wandb


def _wandb_config_from_args(args):
    config = {}
    for key, value in args.__dict__.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
    return config


def setup_wandb(args):
    if args.rank != 0:
        return None
    project = args.wandb_project or os.environ.get('WANDB_PROJECT')
    if not project:
        return None
    wandb = _safe_wandb_import()
    if wandb is None:
        print('W&B requested but wandb is not installed. Skipping W&B logging.')
        return None
    if args.wandb_mode:
        os.environ['WANDB_MODE'] = args.wandb_mode
    tags = [t.strip() for t in (args.wandb_tags or '').split(',') if t.strip()]
    run_name = args.wandb_run_name or args.trial_name
    return wandb.init(
        project=project,
        entity=args.wandb_entity or os.environ.get('WANDB_ENTITY'),
        name=run_name,
        group=args.wandb_group,
        tags=tags or None,
        config=_wandb_config_from_args(args),
    )


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update=True):
    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    if is_update:
        if args.clip > 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()        
        _optimizer.zero_grad()

    if _schedule is not None:
        _schedule.step()


def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0
    start_time = time.time()

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)

    
            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk) 
            loss = _loss.mean() 
            
            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
    model, 
    optimizer, 
    scheduler, 
    train_loader, 
    valid_loader, 
    args, 
    train_step=0, 
    epoch=0
):
    loss = []
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    train_loader.sampler.set_epoch(epoch)

    wandb_run = getattr(args, 'wandb_run', None)
    wandb_log_interval = args.wandb_log_interval or args.log_interval

    for idx, data in enumerate(train_loader):
        data = {key: value for key, value in data.items()}

        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)
        
        _lm_logits, _lm_loss = model(
            _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
        ) 

        _lm_loss = _lm_loss.mean() 
        loss.append(_lm_loss.item())

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        avg_lm_loss.update(_lm_loss.item())
        optimizer_step(
            _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
        )
        if train_step % args.log_interval == 0: 
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | { idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

            if args.rank == 0: 
                print(log_str)
                if wandb_run and (train_step % wandb_log_interval == 0):
                    wandb_run.log({
                        'train/loss': avg_lm_loss.val,
                        'train/avg_loss': avg_lm_loss.avg,
                        'train/ppl': math.exp(avg_lm_loss.avg),
                        'train/lr': lr,
                        'train/ms_per_batch': elapsed * 1000 / args.log_interval,
                        'train/epoch': epoch,
                    }, step=train_step)
            log_start_time = time.time()
            avg_lm_loss.reset()
        
        if train_step % args.save_interval == 0: 
            if args.rank == 0:
                model_path = os.path.join(args.work_dir, f'model_%s.{train_step}.pt'%args.trial_name)
                print('saving checkpoint', model_path)
                # Save full model state (base + LoRA) for proper inference
                torch.save({'model_state_dict': model.state_dict()}, model_path)
            distributed_sync(args)

        # evaluation interval
        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()

            valid_loss, valid_ppl = evaluate(model, valid_loader, args)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl
                
            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '

            if args.rank == 0:
                print('-' * 100)
                print(log_str)
                print('-' * 100)
                if wandb_run:
                    wandb_run.log({
                        'eval/loss': valid_loss,
                        'eval/ppl': valid_ppl,
                        'eval/best_ppl': best_val_ppl,
                        'eval/epoch': epoch,
                    }, step=train_step)

            model.train()
            distributed_sync(args)

        if train_step == args.max_step:
            break

    if args.rank == 0:
        model_path = os.path.join(args.work_dir, f'model_%s.{train_step}.pt'%args.trial_name)
        print('saving checkpoint', model_path)
        # Save full model state (base + LoRA) for proper inference
        torch.save({'model_state_dict': model.state_dict()}, model_path) 
    distributed_sync(args)
    return train_step, loss


def create_muon_optimizer(model, args):
    """Create Muon optimizer with LoRA parameters properly configured."""
    
    # Collect trainable parameters (all LoRA parameters)
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    
    # All LoRA parameters are 2D matrices, so they all use Muon
    muon_params = [p for p in trainable_params if p.ndim >= 2]
    adamw_params = [p for p in trainable_params if p.ndim < 2]
    
    if args.rank == 0:
        print(f'Muon optimizer: {len(muon_params)} params with Muon, {len(adamw_params)} with AdamW')
    
    # Collect LoRA A/B pairs for optional preconditioning/riemannian updates
    lora_pairs = []
    if args.muon_lora_precond or args.muon_lora_riemannian_muon:
        # Iterate through modules to find LoRA layers
        for name, module in model.named_modules():
            # Check for loralib's LoRA layers (used by PilanciLab)
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # loralib stores A and B as parameters directly
                if hasattr(module.lora_A, 'weight') and hasattr(module.lora_B, 'weight'):
                    lora_pairs.append((module.lora_A.weight, module.lora_B.weight))
                elif isinstance(module.lora_A, torch.nn.Parameter) and isinstance(module.lora_B, torch.nn.Parameter):
                    lora_pairs.append((module.lora_A, module.lora_B))
        
        if args.rank == 0:
            print(f'Found {len(lora_pairs)} LoRA A/B pairs for Riemannian preconditioning')
    
    # Create Muon optimizer
    optimizer = Muon(
        lr=args.lr,
        wd=args.weight_decay,
        muon_params=muon_params,
        adamw_params=adamw_params,
        momentum=args.muon_momentum,
        nesterov=args.muon_nesterov,
        ns_steps=args.muon_ns_steps,
        adamw_betas=(args.adam_beta1, args.adam_beta2),
        adamw_eps=args.adam_epislon,
        lora_precond=args.muon_lora_precond,
        lora_precond_eps=args.muon_lora_precond_eps,
        lora_pairs=lora_pairs,
        lora_riemannian_muon=args.muon_lora_riemannian_muon,
        lora_riemannian_ortho_method=args.muon_lora_riemannian_ortho_method,
        lora_riemannian_adjust_lr=args.muon_lora_riemannian_adjust_lr,
        lora_riemannian_variant=args.muon_lora_riemannian_variant,
    )
    
    return optimizer


if __name__ == '__main__':
    args = parser.parse_args()
    if args.deterministic:
        # Ensure deterministic CUDA behavior when requested.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        os.environ.setdefault('PYTHONHASHSEED', str(args.random_seed))
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        np.random.seed(args.random_seed)

    device_number = args.device
    torch.cuda.set_device(int(args.device))
    parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len, 
        joint_lm=args.obj=='jlm'
    )     

    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed)
    )

    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed)
    )

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )

    lm_net = GPT2LMModel(config)
    for n,p in lm_net.named_parameters():
        print('full parameter: ', n, p.shape)

    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        print(torch.load(args.init_checkpoint)['wte.weight'].shape)
        lm_net.load_weight(torch.load(args.init_checkpoint)) 

    lm_net = lm_net.to('cuda:'+str(device_number))

    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)
    
    # Create custom Muon optimizer instead of AdamW
    if args.rank == 0:
        print('=' * 100)
        print('Using CUSTOM MUON OPTIMIZER from ms-swift')
        print(f'  momentum={args.muon_momentum}, nesterov={args.muon_nesterov}, ns_steps={args.muon_ns_steps}')
        print(f'  lora_precond={args.muon_lora_precond}, lora_riemannian_muon={args.muon_lora_riemannian_muon}')
        if args.muon_lora_riemannian_muon:
            print(f'  riemannian_variant={args.muon_lora_riemannian_variant}')
        print('=' * 100)
    
    optimizer = create_muon_optimizer(lm_net, args)

    if args.max_step is None:
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)

    scheduler = create_optimizer_scheduler(optimizer, args)
    
    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")
    
    lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)

    args.wandb_run = setup_wandb(args)

    try:
        train_step = 0
        train_loss = []
        for epoch in itertools.count(start=1):
            train_step, epoch_loss = train_validate(
                lm_net, optimizer, scheduler, train_loader, valid_loader, args, 
                train_step=train_step, epoch=epoch
            )
            train_loss += epoch_loss
            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                if args.rank == 0:
                    print('-' * 100)
                    print('End of training')
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print('-' * 100)
            print('Exiting from training early')

    distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)
    if args.rank == 0 and getattr(args, 'wandb_run', None):
        args.wandb_run.finish()
