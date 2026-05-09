#  ------------------------------------------------------------------------------------------
#  GPT-2 E2E fine-tuning with Riemannion optimizer — SGD, no-momentum variant.
#  Fork of _third_party/RiemanianFinetune/experiments/e2e/gpt2_ft_riemannian.py with:
#    - default riemannian_optimizer = "sgd" (no Adam second-moment normalization)
#    - default riemannian_sgd_momentum = 0.0 (no Riemannian momentum either)
#  This fills the no-momentum block of Table 2 (apples-to-apples with SGD / Scaled GD / Muon / V1 / V5 nomom).
#  Uses pilancilab's data loading, model, and training loop.
#  Evaluation via pilancilab's beam search + e2e-metrics for comparable BLEU scores.
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

# Locate REPO_ROOT. Defaults to 3 levels up from this file, which matches the
# layout in this repository:
#   <REPO_ROOT>/examples/riemannian_muon_lora/pilancilab_pipeline/gpt2_ft_riemannion_sgd.py
# Override with the REPO_DIR env var if running from a different layout.
REPO_ROOT = os.environ.get(
    'REPO_DIR',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
)
PILANCILAB_SRC = os.path.join(REPO_ROOT, '_third_party', 'pilancilab', 'GPT2', 'examples', 'NLG', 'src')
sys.path.insert(0, PILANCILAB_SRC)

from gpu import (
    add_gpu_params,
    parse_gpu,
    distributed_opt,
    distributed_gather,
    distributed_sync,
    cleanup
)
from optimizer import (
    create_optimizer_scheduler,
    add_optimizer_params,
)
from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

import loralib as lora

# Import Riemannion optimizer
RIEMANNION_SRC = os.path.join(REPO_ROOT, '_third_party', 'RiemanianFinetune', 'src', 'optimizers')
sys.path.insert(0, RIEMANNION_SRC)
from RiemannianLoRA import RiemannianLora as _RiemannianLora, RiemannianSGD as _RiemannianSGD, FixedRank


def _riemannian_step_single(A, B, lr, beta, state, apply_momentum_fn, apply_second_momentum_fn):
    """Run one Riemannion optimizer step on a single (A, B) pair.

    A: shape (2r, m) — top r orthogonal, bottom r zeros
    B: shape (n, 2r) — left r active, right r auxiliary

    Returns updated (new_A_packed, new_B_packed) with same shapes.
    """
    r = A.shape[0] // 2

    if 'manifold' not in state:
        state['step'] = 0
        state['manifold'] = FixedRank(r=r)

    manifold = state['manifold']

    # Extract point and Riemannian gradient
    dot_A = A.grad.T if A.grad is not None else torch.zeros_like(A.T)
    dot_B = B.grad.T if B.grad is not None else torch.zeros_like(B.T)

    dot_A_r = dot_A[:, r:]
    dot_B_r = dot_B[:r, :]
    A_point = A.T[:, :r]
    B_point = B.T[:r, :]
    point = [A_point, B_point]
    vec = manifold.euclidian_to_reimanian(point, [dot_A_r, dot_B_r])

    # Apply momentum
    momentum_vec = apply_momentum_fn(point=point, vec=vec, state=state,
                                     manifold=manifold, beta=beta[0])

    # Apply second momentum (Adam) or passthrough (SGD)
    dot_A_final, dot_B_final = apply_second_momentum_fn(
        point=point, grad_vec=vec, momentum_vec=momentum_vec,
        state=state, beta=beta, manifold=manifold)

    # Retraction back to manifold
    new_A, new_B = manifold.retraction(
        point=point,
        tangent_vec=[-lr * dot_A_final, point[1] - lr * dot_B_final])

    # Rebuild doubled-rank structure
    A_zero = torch.zeros_like(new_A)
    B_orth = torch.linalg.qr(new_B.T).Q.T
    new_A_packed = torch.cat([new_A, A_zero], dim=1).T   # (2r, m)
    new_B_packed = torch.cat([new_B, B_orth], dim=0).T    # (n, 2r)

    state['step'] += 1
    return new_A_packed, new_B_packed


class LoralibRiemannianOptimizer(torch.optim.Optimizer):
    """Riemannion optimizer adapted for loralib, handling MergedLinear unpacking.

    Each param group contains one loralib module's (lora_A, lora_B).
    For MergedLinear (num_enabled > 1), we unpack into per-module pairs,
    run Riemannion on each, and repack.
    """

    def __init__(self, param_groups, lr=0.001, betas=(0.9, 0.999), use_adam=True):
        self.use_adam = use_adam
        defaults = {'lr': lr, 'betas': list(betas)}
        # param_groups is a list of dicts with 'lora_A', 'lora_B', 'num_modules', 'r'
        # We need to register actual parameters with the base Optimizer
        opt_groups = []
        for pg in param_groups:
            opt_groups.append({
                'params': [pg['lora_A'], pg['lora_B']],
                'lr': lr,
                'betas': list(betas),
                'num_modules': pg['num_modules'],
                'r': pg['r'],
            })
        super().__init__(opt_groups, defaults)
        self._momentum_fn = _RiemannianLora.apply_momentum
        self._second_momentum_fn = (
            _RiemannianLora.apply_second_momentum if use_adam
            else lambda self_, **kw: kw['momentum_vec']
        )
        # Create a dummy parent for bound method calls
        self._dummy = _RiemannianLora.__new__(_RiemannianLora)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            A_packed = group['params'][0]  # lora_A
            B_packed = group['params'][1]  # lora_B
            num_modules = group['num_modules']
            r = group['r']  # model rank (doubled)
            lr = group['lr']
            beta = group['betas']

            if num_modules == 1:
                # Standard Linear: A (r, m), B (n, r) — treat as single pair
                state = self.state[A_packed]
                new_A, new_B = _riemannian_step_single(
                    A_packed.data, B_packed.data, lr, beta, state,
                    lambda **kw: _RiemannianLora.apply_momentum(self._dummy, **kw),
                    lambda **kw: self._second_momentum_fn(self._dummy, **kw)
                        if self.use_adam else lambda **kw: kw['momentum_vec'],
                )
                A_packed.data.copy_(new_A)
                B_packed.data.copy_(new_B)
            else:
                # MergedLinear: unpack per-module, step each, repack
                A_chunks = A_packed.data.chunk(num_modules, dim=0)  # each (r, m)
                out_per_mod = B_packed.shape[0] // num_modules
                B_chunks = [B_packed.data[i * out_per_mod:(i + 1) * out_per_mod, :]
                            for i in range(num_modules)]

                # Split gradients too
                if A_packed.grad is not None:
                    A_grad_chunks = A_packed.grad.chunk(num_modules, dim=0)
                else:
                    A_grad_chunks = [torch.zeros_like(c) for c in A_chunks]
                if B_packed.grad is not None:
                    B_grad_chunks = [B_packed.grad[i * out_per_mod:(i + 1) * out_per_mod, :]
                                     for i in range(num_modules)]
                else:
                    B_grad_chunks = [torch.zeros_like(c) for c in B_chunks]

                new_A_parts = []
                new_B_parts = []

                for mod_i in range(num_modules):
                    state_key = f'{id(A_packed)}_mod{mod_i}'
                    if state_key not in self.state:
                        self.state[state_key] = {}
                    state = self.state[state_key]

                    # Create temporary tensors with gradients
                    A_i = A_chunks[mod_i].clone()
                    B_i = B_chunks[mod_i].clone()
                    A_i.grad = A_grad_chunks[mod_i]
                    B_i.grad = B_grad_chunks[mod_i]

                    new_A_i, new_B_i = _riemannian_step_single(
                        A_i, B_i, lr, beta, state,
                        lambda **kw: _RiemannianLora.apply_momentum(self._dummy, **kw),
                        (lambda **kw: _RiemannianLora.apply_second_momentum(self._dummy, **kw))
                            if self.use_adam else (lambda **kw: kw['momentum_vec']),
                    )
                    new_A_parts.append(new_A_i)
                    new_B_parts.append(new_B_i)

                A_packed.data.copy_(torch.cat(new_A_parts, dim=0))
                B_packed.data.copy_(torch.cat(new_B_parts, dim=0))

parser = argparse.ArgumentParser(description='GPT2 E2E fine-tuning with Riemannion optimizer')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True)
parser.add_argument('--trial_name', default="riemannion", type=str)
parser.add_argument('--valid_data', required=True)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--valid_batch_size', type=int, default=4)
parser.add_argument('--grad_acc', type=int, default=1)
parser.add_argument('--clip', type=float, default=0.0)
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'])
parser.add_argument('--init_checkpoint', default=None)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--eval_interval', type=int, default=2000)
parser.add_argument('--save_interval', type=int, default=500)
parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'))
parser.add_argument('--lora_dim', type=int, default=4, help='effective LoRA rank (model uses 2x for doubled-rank trick)')
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'])
parser.add_argument('--lora_dropout', default=0.1, type=float)
parser.add_argument('--label_smooth', default=0.1, type=float)
parser.add_argument('--roll_interval', type=int, default=-1)
parser.add_argument('--roll_lr', type=float, default=0.00001)
parser.add_argument('--roll_step', type=int, default=100)
parser.add_argument('--eval_epoch', type=int, default=1)

# Riemannion-specific arguments
parser.add_argument('--riemannian_optimizer', type=str, default='sgd',
                    choices=['adam', 'sgd'],
                    help='Riemannian optimizer: adam (with momentum) or sgd (no momentum). '
                         'This driver defaults to sgd for no-momentum Table 2 baseline.')
parser.add_argument('--riemannian_beta1', type=float, default=0.9)
parser.add_argument('--riemannian_beta2', type=float, default=0.999)
parser.add_argument('--riemannian_sgd_momentum', type=float, default=0.0,
                    help='Momentum for RiemannianSGD (0.0 = no momentum)')

parser.add_argument('--deterministic', action='store_true')


def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


class AverageMeter(object):
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
                _grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else:
                _grad_norm = torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

        _optimizer.step()
        _optimizer.zero_grad()

    if _schedule is not None:
        _schedule.step()


def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    total_token = 0

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value.to(args.device) for key, value in data.items()}
            _input = data['input']
            _target = data['target']
            _msk = data['mask']

            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk,
                                      label_smooth=args.label_smooth)
            _loss = _loss.mean()
            _token = _msk.float().sum().item()
            total_loss += _loss.item() * _token
            total_token += _token

    total_loss /= total_token
    model.train()
    return total_loss


def train_validate(model, optimizer, scheduler, train_loader, valid_loader, args,
                   train_step=0, epoch=0):
    model.train()
    avg_lm_loss = AverageMeter()
    train_loss = []

    log_start_time = time.time()

    best_val_ppl = None

    train_loader.sampler.set_epoch(epoch)

    for idx, data in enumerate(train_loader):
        data = {key: value.to(args.device) for key, value in data.items()}

        _input = data['input']
        _target = data['target']
        _msk = data['mask']

        _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk,
                                  label_smooth=args.label_smooth)
        _loss = _loss.mean()

        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False
        avg_lm_loss.update(_loss.item())
        optimizer_step(_loss / (args.grad_acc), optimizer, model, scheduler, args, is_update=is_update)

        if train_step % args.log_interval == 0:
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | {idx + 1:>6d} batches | ' \
                      f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                      f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
                      f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

            if args.rank == 0:
                print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()

        if train_step % args.save_interval == 0:
            if args.rank == 0:
                model_path = os.path.join(args.work_dir, f'model.{args.trial_name}.{train_step}.pt')
                print(f'saving checkpoint {model_path}')
                torch.save({'model_state_dict': model.state_dict()}, model_path)

        if train_step % args.eval_interval == 0:
            eval_start_time = time.time()
            val_loss = evaluate(model, valid_loader, args)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - eval_start_time

            if args.rank == 0:
                print(f'-' * 100)
                print(f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | '
                      f'time: {elapsed:5.2f}s | valid loss {val_loss:5.2f} | valid ppl {val_ppl:5.2f}')
                print(f'-' * 100)

                if best_val_ppl is None or val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    model_path = os.path.join(args.work_dir, f'model.{args.trial_name}.best.pt')
                    torch.save({'model_state_dict': model.state_dict()}, model_path)

            train_loss.append(val_loss)

        if train_step >= args.max_step:
            break

    if args.rank == 0:
        model_path = os.path.join(args.work_dir, f'model.{args.trial_name}.final.pt')
        torch.save({'model_state_dict': model.state_dict()}, model_path)
    distributed_sync(args)
    return train_step, train_loss


def init_riemannian_loralib(model, effective_rank):
    """Apply Riemannian doubled-rank initialization to loralib parameters.

    For each LoRA module within a layer:
      A_i (2r x m): top r rows orthogonal, bottom r rows zeros
      B_i (n_i x 2r): left r cols orthogonal (scaled), right r cols = copy of left

    For MergedLinear, A and B are packed: we init each module's slice separately.
    """
    r_model = 2 * effective_rank  # doubled rank used in model

    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue
        if not isinstance(module.lora_A, torch.nn.Parameter):
            continue

        A = module.lora_A  # (r_model * num_modules, in_features)
        B = module.lora_B  # (out_per_mod * num_modules, r_model)

        # Determine number of packed modules
        num_modules = A.shape[0] // r_model

        for mod_i in range(num_modules):
            # A slice for this module
            a_start = mod_i * r_model
            a_half = r_model // 2

            A_top = torch.empty(a_half, A.shape[1], device='cpu')
            torch.nn.init.orthogonal_(A_top, gain=1.0)
            A.data[a_start:a_start + a_half, :] = A_top.to(A.device)
            A.data[a_start + a_half:a_start + r_model, :] = 0.0

            # B slice for this module
            out_per_mod = B.shape[0] // num_modules
            b_start = mod_i * out_per_mod
            b_half = r_model // 2
            n_i = out_per_mod

            B_left = torch.empty(n_i, b_half, device='cpu')
            torch.nn.init.orthogonal_(B_left, gain=n_i ** (-0.5))
            B_left_dev = B_left.to(B.device)
            B.data[b_start:b_start + n_i, :b_half] = B_left_dev
            B.data[b_start:b_start + n_i, b_half:r_model] = B_left_dev

    print(f'Applied Riemannian initialization (effective_rank={effective_rank}, '
          f'model_rank={r_model})')


def create_riemannian_optimizer(model, args):
    """Create Riemannion optimizer with loralib param groups."""
    r_model = 2 * args.lora_dim  # doubled rank
    param_groups = []

    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')):
            continue
        if not isinstance(module.lora_A, torch.nn.Parameter):
            continue

        num_modules = module.lora_A.shape[0] // r_model
        param_groups.append({
            'lora_A': module.lora_A,
            'lora_B': module.lora_B,
            'num_modules': num_modules,
            'r': r_model,
        })

    if args.rank == 0:
        total_pairs = sum(pg['num_modules'] for pg in param_groups)
        print(f'Riemannion optimizer: {len(param_groups)} layers, '
              f'{total_pairs} total LoRA module pairs')

    use_adam = (args.riemannian_optimizer == 'adam')
    betas = [args.riemannian_beta1, args.riemannian_beta2] if use_adam else [args.riemannian_sgd_momentum, 0.0]

    optimizer = LoralibRiemannianOptimizer(
        param_groups,
        lr=args.lr,
        betas=betas,
        use_adam=use_adam,
    )

    return optimizer


if __name__ == '__main__':
    args = parser.parse_args()
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        os.environ.setdefault('PYTHONHASHSEED', str(args.random_seed))
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        np.random.seed(args.random_seed)

    device_number = args.device
    torch.cuda.set_device(int(args.device))
    parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception:
            warnings.warn('Could not import amp, apex may not be installed')

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    train_data = FT_Dataset(args.train_data, args.train_batch_size, args.seq_len,
                            joint_lm=args.obj == 'jlm')
    valid_data = FT_Dataset(args.valid_data, args.valid_batch_size, args.seq_len)

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, num_workers=0,
                              shuffle=False, pin_memory=False, drop_last=True,
                              sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed))
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, num_workers=0,
                              shuffle=False, pin_memory=False, drop_last=False,
                              sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed))

    # Use DOUBLED lora_dim for the Riemannian trick
    effective_rank = args.lora_dim
    model_lora_dim = 2 * effective_rank

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(n_embd=768, n_layer=12, n_head=12,
                            lora_attn_dim=model_lora_dim, lora_attn_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout)
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(n_embd=1024, n_layer=24, n_head=16,
                            lora_attn_dim=model_lora_dim, lora_attn_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout)
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(n_embd=1280, n_layer=36, n_head=20,
                            lora_attn_dim=model_lora_dim, lora_attn_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout)

    lm_net = GPT2LMModel(config)

    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        lm_net.load_weight(torch.load(args.init_checkpoint))

    lm_net = lm_net.to('cuda:' + str(device_number))

    if model_lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net)

    # Apply Riemannian initialization (doubled-rank layout)
    init_riemannian_loralib(lm_net, effective_rank)

    if args.rank == 0:
        print('=' * 100)
        print(f'Using RIEMANNION OPTIMIZER ({args.riemannian_optimizer})')
        print(f'  effective_rank={effective_rank}, model_lora_dim={model_lora_dim}')
        if args.riemannian_optimizer == 'adam':
            print(f'  betas=[{args.riemannian_beta1}, {args.riemannian_beta2}]')
        else:
            print(f'  momentum={args.riemannian_sgd_momentum}')
        print('=' * 100)

    optimizer = create_riemannian_optimizer(lm_net, args)

    if args.max_step is None:
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)

    scheduler = create_optimizer_scheduler(optimizer, args)

    if args.fp16:
        lm_net, optimizer = amp.initialize(lm_net, optimizer, opt_level="O1")

    lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)

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
