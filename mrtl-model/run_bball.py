import argparse
import copy
import json
import logging
import os

import torch
import numpy as np

import utils
from config import config
from cp_als import cp_decompose
from data.basketball.dataset import BballRawDataset
from train.basketball.multi import BasketballMulti
from visualization import plot

# Arguments Parse
parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', dest='root_dir')
parser.add_argument('--data-dir', dest='data_dir')
parser.add_argument('--type', dest='type', choices=['multi', 'fixed', 'rand'])
parser.add_argument('--stop-cond',
                    dest='stop_cond',
                    choices=[
                        'val_loss_increase', 'gradient_entropy',
                        'gradient_norm', 'gradient_variance'
                    ])
parser.add_argument('--batch-size', dest='batch_size', type=int)
parser.add_argument('--sigma', dest='sigma', type=float)
parser.add_argument('--K', dest='K', type=int)
parser.add_argument('--step-size', dest='step_size', type=int)
parser.add_argument('--gamma', dest='gamma', type=float)
parser.add_argument('--full-lr', dest='full_lr', type=float)
parser.add_argument('--full-reg', dest='full_reg', type=float)
parser.add_argument('--low-lr', dest='low_lr', type=float)
parser.add_argument('--low-reg', dest='low_reg', type=float)
args = parser.parse_args()

# Parameters
params = dict()
for arg in vars(args):
    if arg not in ['project_dir', 'root_dir', 'data_dir', 'type']:
        params[arg] = getattr(args, arg)

# Save parameters
os.makedirs(args.root_dir, exist_ok=True)
with open(os.path.join(args.root_dir, f"params_{args.type}.json"),
          "w",
          encoding='utf-8') as f:
    json.dump(params, f, ensure_ascii=False, indent=4)

# Create separate params dict
hyper = copy.deepcopy(params)
for k in list(hyper):
    if k.startswith('full') or k.startswith('low'):
        hyper.pop(k)
_ = [hyper.pop(k, None) for k in ['K', 'nonnegative_weights']]

# Set directories
fig_dir = os.path.join(args.root_dir, "fig")
os.makedirs(fig_dir, exist_ok=True)
save_dir = os.path.join(args.root_dir, "saved")
os.makedirs(save_dir, exist_ok=True)

# Set logger
main_logger = logging.getLogger(config.parent_logger_name)
utils.set_logger(main_logger, os.path.join(args.root_dir, f"{args.type}.log"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Results
results = {
    'best_epochs': [],
    'best_lr': [],
    'train_times': [],
    'train_loss': [],
    'grad_norms': [],
    'grad_entropies': [],
    'grad_vars': [],
    'val_times': [],
    'val_loss': [],
    'val_conf_matrix': [],
    'val_acc': [],
    'val_precision': [],
    'val_recall': [],
    'val_F1': [],
    'test_conf_matrix': [],
    'test_acc': [],
    'test_precision': [],
    'test_recall': [],
    'test_F1': [],
    'test_out': [],
    'test_labels': []
}
if args.type == 'multi':
    results['dims'] = [[[4, 5], [6, 6]], [[8, 10], [6, 6]], [[8, 10], [12,
                                                                       12]],
                       [[8, 10], [12, 12]], [[20, 25], [12, 12]],
                       [[40, 50], [12, 12]]]  # Low rank
    results['low_start_idx'] = 3
elif args.type == 'fixed':
    results['dims'] = [[[40, 50], [12, 12]], [[40, 50], [12, 12]]]
    results['low_start_idx'] = 1
elif args.type == 'rand':
    results['dims'] = [[[40, 50], [12, 12]]]
    results['low_start_idx'] = 0

# Create datasets
train_set = BballRawDataset(os.path.join(args.data_dir, config.fn_train))
val_set = BballRawDataset(os.path.join(args.data_dir, config.fn_val))
test_set = BballRawDataset(os.path.join(args.data_dir, config.fn_test))

if args.type == 'multi' or args.type == 'fixed':
    # Full-rank first resolution
    b = results['dims'][0][0]
    c = results['dims'][0][1]
    b_str = utils.size_to_str(b)
    c_str = utils.size_to_str(c)
    train_set.calculate_pos(b, c)
    val_set.calculate_pos(b, c)

    # Train
    multi = BasketballMulti(device)
    multi.init_full_model(train_set)
    hyper['lr'] = params['full_lr']
    hyper['reg_coeff'] = params['full_reg']
    hyper['stop_threshold'] = params.get('full_stop_threshold')
    multi.init_params(**hyper)
    multi.init_loaders(train_set, val_set)
    multi.train_and_evaluate(save_dir)
    
    # Bug fix
    b = [b[0]//4, b[1]]
    c = [c[0]//4, c[1]]

    # Test
    # Create dataset
    test_set.calculate_pos(b, c)
    multi.model.load_state_dict(multi.best_model_dict)
    test_conf_matrix, test_acc, test_precision, test_recall, test_F1, test_out, test_labels = multi.test(
        test_set)

    # Metrics
    results['best_epochs'].append(multi.best_epochs)
    results['best_lr'].append(multi.best_lr)
    results['train_times'].append(multi.train_times[:multi.best_epochs])
    results['train_loss'].append(multi.train_loss[:multi.best_epochs])
    results['grad_norms'].append(multi.grad_norms[:multi.best_epochs])
    results['grad_entropies'].append(multi.grad_entropies[:multi.best_epochs])
    results['grad_vars'].append(multi.grad_vars[:multi.best_epochs])
    results['val_times'].append(multi.val_times[:multi.best_epochs])
    results['val_loss'].append(multi.val_loss[:multi.best_epochs])
    results['val_conf_matrix'].append(
        multi.val_conf_matrix[:multi.best_epochs])
    results['val_acc'].append(multi.val_acc[:multi.best_epochs])
    results['val_precision'].append(multi.val_precision[:multi.best_epochs])
    results['val_recall'].append(multi.val_recall[:multi.best_epochs])
    results['val_F1'].append(multi.val_F1[:multi.best_epochs])
    results['test_conf_matrix'].append(test_conf_matrix)
    results['test_acc'].append(test_acc)
    results['test_precision'].append(test_precision)
    results['test_recall'].append(test_recall)
    results['test_F1'].append(test_F1)
    results['test_out'].append(test_out)
    results['test_labels'].append(test_labels)

    prev_b = b
    prev_c = c

    for b, c in results['dims'][1:results['low_start_idx']]:
        b_str = utils.size_to_str(b)
        c_str = utils.size_to_str(c)

        # Calculate_pos
        train_set.calculate_pos(b, c)
        val_set.calculate_pos(b, c)

        # Finegrain
        prev_model_dict = multi.best_model_dict
        print(prev_model_dict)

        if (4 * b[0]) != prev_model_dict['W'].size(
                1) or b[1] != prev_model_dict['W'].size(2):
            # Separate the tensor into quarters
            W_1, W_2, W_3, W_4 = torch.chunk(prev_model_dict['W'], 4, 1)
            # Finegrain each quarter separately
            W_1 = utils.finegrain(W_1, b, 1)
            W_2 = utils.finegrain(W_2, b, 1)
            W_3 = utils.finegrain(W_3, b, 1)
            W_4 = utils.finegrain(W_4, b, 1)
            # Recombine the quarter chunks into one tensor
            prev_model_dict['W'] = torch.cat((W_1, W_2, W_3, W_4), 1)
        if (4 * c[0]) != prev_model_dict['W'].size(
                3) or c[1] != prev_model_dict['W'].size(4):
            # Separate the tensor into quarters
            W_1, W_2, W_3, W_4 = torch.chunk(prev_model_dict['W'], 4, 3)
            # Finegrain each quarter separately
            W_1 = utils.finegrain(W_1, c, 3)
            W_2 = utils.finegrain(W_2, c, 3)
            W_3 = utils.finegrain(W_3, c, 3)
            W_4 = utils.finegrain(W_4, c, 3)
            # Recombine the quarter chunks into one tensor
            prev_model_dict['W'] = torch.cat((W_1, W_2, W_3, W_4), 3)

        # Train
        # hyper['lr'] = multi.best_lr / ((b[0] / prev_b[0]) * (c[0] / prev_c[0]))
        hyper['lr'] = multi.best_lr
        multi = BasketballMulti(device)
        multi.init_full_model(train_set)
        multi.model.load_state_dict(prev_model_dict)
        multi.init_params(**hyper)
        multi.init_loaders(train_set, val_set)
        multi.train_and_evaluate(save_dir)
        
        # Bug fix
        b = [b[0]//4, b[1]]
        c = [c[0]//4, c[1]]

        # Test
        # Create dataset
        test_set.calculate_pos(b, c)
        multi.model.load_state_dict(multi.best_model_dict)
        test_conf_matrix, test_acc, test_precision, test_recall, test_F1, test_out, test_labels = multi.test(
            test_set)

        # Metrics
        results['best_epochs'].append(multi.best_epochs)
        results['best_lr'].append(multi.best_lr)
        results['train_times'].append(multi.train_times[:multi.best_epochs])
        results['train_loss'].append(multi.train_loss[:multi.best_epochs])
        results['grad_norms'].append(multi.grad_norms[:multi.best_epochs])
        results['grad_entropies'].append(
            multi.grad_entropies[:multi.best_epochs])
        results['grad_vars'].append(multi.grad_vars[:multi.best_epochs])
        results['val_times'].append(multi.val_times[:multi.best_epochs])
        results['val_loss'].append(multi.val_loss[:multi.best_epochs])
        results['val_conf_matrix'].append(
            multi.val_conf_matrix[:multi.best_epochs])
        results['val_acc'].append(multi.val_acc[:multi.best_epochs])
        results['val_precision'].append(
            multi.val_precision[:multi.best_epochs])
        results['val_recall'].append(multi.val_recall[:multi.best_epochs])
        results['val_F1'].append(multi.val_F1[:multi.best_epochs])
        results['test_conf_matrix'].append(test_conf_matrix)
        results['test_acc'].append(test_acc)
        results['test_precision'].append(test_precision)
        results['test_recall'].append(test_recall)
        results['test_F1'].append(test_F1)
        results['test_out'].append(test_out)
        results['test_labels'].append(test_labels)

        prev_b = b
        prev_c = c

    # Draw plots for full rank train
    fp_fig = os.path.join(fig_dir, "full_time_vs_loss.png")
    plot.loss_time(results['train_times'],
                   results['train_loss'],
                   results['val_times'],
                   results['val_loss'],
                   fp_fig=fp_fig)

    # CP_decomposition
    prev_model_dict = multi.best_model_dict
    hyper['K'] = params['K']
    W_size = prev_model_dict['W'].size()
    W = prev_model_dict['W'].view(W_size[0], W_size[1] * W_size[2],
                                         W_size[3] * W_size[4])
    weights, factors = cp_decompose(W, hyper['K'], max_iter=2000)
    factors = [f * torch.pow(weights, 1 / len(factors)) for f in factors]
    
    # Bug fix
    b = [b[0] * 4, b[1]]
    c = [c[0] * 4, c[1]]
    
    prev_model_dict.pop('W')
    prev_model_dict['A'] = factors[0].clone().detach()
    prev_model_dict['B'] = factors[1].clone().detach().view(
        *b, hyper['K'])
    prev_model_dict['C'] = factors[2].clone().detach().view(
        *c, hyper['K'])

    # Draw heatmaps after CP decomposition
    B_1, B_2, B_3, B_4 = torch.chunk(prev_model_dict['B'], 4, 0)
    C_1, C_2, C_3, C_4 = torch.chunk(prev_model_dict['C'], 4, 0)
    
    # Reorder latent factors for visualization
    change = True
    
    while(change):
        change = False
        for i in range(B_1.shape[len(list(B_1.shape)) - 1]):
            first = torch.tensor(B_1[..., i].cpu().numpy().copy())
            baseline = torch.tensor(B_2[..., i].cpu().numpy().copy())
            for j in range(B_2.shape[len(list(B_2.shape)) - 1]):
                second = torch.tensor(B_2[..., j].cpu().numpy().copy())
                difference = np.linalg.norm((first - second).cpu())
                max_difference = np.linalg.norm((first - baseline).cpu())
                if difference < max_difference:
                    change = True
                    max_difference = difference
                    B_2[..., i].copy_(second)
                    B_2[..., j].copy_(baseline)
                    
    change = True
    
    while(change):
        change = False
        for i in range(B_2.shape[len(list(B_2.shape)) - 1]):
            first = torch.tensor(B_2[..., i].cpu().numpy().copy())
            baseline = torch.tensor(B_3[..., i].cpu().numpy().copy())
            for j in range(B_3.shape[len(list(B_3.shape)) - 1]):
                second = torch.tensor(B_3[..., j].cpu().numpy().copy())
                difference = np.linalg.norm((first - second).cpu())
                max_difference = np.linalg.norm((first - baseline).cpu())
                if difference < max_difference:
                    change = True
                    max_difference = difference
                    B_3[..., i].copy_(second)
                    B_3[..., j].copy_(baseline)
                    
    change = True
    
    while(change):
        change = False
        for i in range(B_3.shape[len(list(B_3.shape)) - 1]):
            first = torch.tensor(B_3[..., i].cpu().numpy().copy())
            baseline = torch.tensor(B_4[..., i].cpu().numpy().copy())
            for j in range(B_4.shape[len(list(B_4.shape)) - 1]):
                second = torch.tensor(B_4[..., j].cpu().numpy().copy())
                difference = np.linalg.norm((first - second).cpu())
                max_difference = np.linalg.norm((first - baseline).cpu())
                if difference < max_difference:
                    change = True
                    max_difference = difference
                    B_4[..., i].copy_(second)
                    B_4[..., j].copy_(baseline)
                    
    change = True
    
    while(change):
        change = False
        for i in range(C_1.shape[len(list(C_1.shape)) - 1]):
            first = torch.tensor(C_1[..., i].cpu().numpy().copy())
            baseline = torch.tensor(C_2[..., i].cpu().numpy().copy())
            for j in range(C_2.shape[len(list(C_2.shape)) - 1]):
                second = torch.tensor(C_2[..., j].cpu().numpy().copy())
                difference = np.linalg.norm((first - second).cpu())
                max_difference = np.linalg.norm((first - baseline).cpu())
                if difference < max_difference:
                    change = True
                    max_difference = difference
                    C_2[..., i].copy_(second)
                    C_2[..., j].copy_(baseline)
                    
    change = True
    
    while(change):
        change = False
        for i in range(C_2.shape[len(list(C_2.shape)) - 1]):
            first = torch.tensor(C_2[..., i].cpu().numpy().copy())
            baseline = torch.tensor(C_3[..., i].cpu().numpy().copy())
            for j in range(C_3.shape[len(list(C_3.shape)) - 1]):
                second = torch.tensor(C_3[..., j].cpu().numpy().copy())
                difference = np.linalg.norm((first - second).cpu())
                max_difference = np.linalg.norm((first - baseline).cpu())
                if difference < max_difference:
                    change = True
                    max_difference = difference
                    C_3[..., i].copy_(second)
                    C_3[..., j].copy_(baseline)
                    
    change = True
    
    while(change):
        change = False
        for i in range(C_3.shape[len(list(C_3.shape)) - 1]):
            first = torch.tensor(C_3[..., i].cpu().numpy().copy())
            baseline = torch.tensor(C_4[..., i].cpu().numpy().copy())
            for j in range(C_4.shape[len(list(C_4.shape)) - 1]):
                second = torch.tensor(C_4[..., j].cpu().numpy().copy())
                difference = np.linalg.norm((first - second).cpu())
                max_difference = np.linalg.norm((first - baseline).cpu())
                if difference < max_difference:
                    change = True
                    max_difference = difference
                    C_4[..., i].copy_(second)
                    C_4[..., j].copy_(baseline)
    
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_B_heatmap_1.png".format(b_str, c_str))
    plot.latent_factor_heatmap(B_1, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_B_heatmap_2.png".format(b_str, c_str))
    plot.latent_factor_heatmap(B_2, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_B_heatmap_3.png".format(b_str, c_str))
    plot.latent_factor_heatmap(B_3, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_B_heatmap_4.png".format(b_str, c_str))
    plot.latent_factor_heatmap(B_4, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_C_heatmap_1.png".format(b_str, c_str))
    plot.latent_factor_heatmap(C_1, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_C_heatmap_2.png".format(b_str, c_str))
    plot.latent_factor_heatmap(C_2, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_C_heatmap_3.png".format(b_str, c_str))
    plot.latent_factor_heatmap(C_3, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "full_{0},{1}_C_heatmap_4.png".format(b_str, c_str))
    plot.latent_factor_heatmap(C_4, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)

# Low-rank first resolution
b = results['dims'][results['low_start_idx']][0]
c = results['dims'][results['low_start_idx']][1]
b_str = utils.size_to_str(b)
c_str = utils.size_to_str(c)
train_set.calculate_pos(b, c)
val_set.calculate_pos(b, c)

# Train
multi = BasketballMulti(device)
hyper['K'] = params['K']
hyper['lr'] = params['low_lr']
hyper['reg_coeff'] = params['low_reg']
hyper['stop_threshold'] = params.get('low_stop_threshold')
multi.init_low_model(train_set, hyper['K'])
if args.type == 'multi' or args.type == 'fixed':
    multi.model.load_state_dict(prev_model_dict)
multi.init_params(**hyper)
multi.init_loaders(train_set, val_set)
multi.train_and_evaluate(save_dir)

# Draw heatmaps
print(multi.best_model_dict)
B_1, B_2, B_3, B_4 = torch.chunk(prev_model_dict['B'], 4, 0)
C_1, C_2, C_3, C_4 = torch.chunk(prev_model_dict['C'], 4, 0)
fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap_1.png".format(b_str, c_str))
plot.latent_factor_heatmap(B_1, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap_2.png".format(b_str, c_str))
plot.latent_factor_heatmap(B_2, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap_3.png".format(b_str, c_str))
plot.latent_factor_heatmap(B_3, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap_4.png".format(b_str, c_str))
plot.latent_factor_heatmap(B_4, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap_1.png".format(b_str, c_str))
plot.latent_factor_heatmap(C_1, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap_2.png".format(b_str, c_str))
plot.latent_factor_heatmap(C_2, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap_3.png".format(b_str, c_str))
plot.latent_factor_heatmap(C_3, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap_4.png".format(b_str, c_str))
plot.latent_factor_heatmap(C_4, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)

# Bug fix
b = [b[0]//4, b[1]]
c = [c[0]//4, c[1]]

# Test
# Create dataset
test_set.calculate_pos(b, c)
multi.model.load_state_dict(multi.best_model_dict)
test_conf_matrix, test_acc, test_precision, test_recall, test_F1, test_out, test_labels = multi.test(
    test_set)

# Metrics
results['best_epochs'].append(multi.best_epochs)
results['best_lr'].append(multi.best_lr)
results['train_times'].append(multi.train_times[:multi.best_epochs])
results['train_loss'].append(multi.train_loss[:multi.best_epochs])
results['grad_norms'].append(multi.grad_norms[:multi.best_epochs])
results['grad_entropies'].append(multi.grad_entropies[:multi.best_epochs])
results['grad_vars'].append(multi.grad_vars[:multi.best_epochs])
results['val_times'].append(multi.val_times[:multi.best_epochs])
results['val_loss'].append(multi.val_loss[:multi.best_epochs])
results['val_conf_matrix'].append(multi.val_conf_matrix[:multi.best_epochs])
results['val_acc'].append(multi.val_acc[:multi.best_epochs])
results['val_precision'].append(multi.val_precision[:multi.best_epochs])
results['val_recall'].append(multi.val_recall[:multi.best_epochs])
results['val_F1'].append(multi.val_F1[:multi.best_epochs])
results['test_conf_matrix'].append(test_conf_matrix)
results['test_acc'].append(test_acc)
results['test_precision'].append(test_precision)
results['test_recall'].append(test_recall)
results['test_F1'].append(test_F1)
results['test_out'].append(test_out)
results['test_labels'].append(test_labels)

prev_b = b
prev_c = c
for b, c in results['dims'][results['low_start_idx'] + 1:]:
    b_str = utils.size_to_str(b)
    c_str = utils.size_to_str(c)

    # Calculate_pos
    train_set.calculate_pos(b, c)
    val_set.calculate_pos(b, c)

    # Finegrain
    prev_model_dict = multi.best_model_dict
    if (4 * b[0]) != prev_model_dict['B'].size(
            0) or b[1] != prev_model_dict['B'].size(1):
        # Separate B into quarters
        B_1, B_2, B_3, B_4 = torch.chunk(prev_model_dict['B'], 4, 0)
        # Finegrain each quarter separately
        B_1 = utils.finegrain(B_1, b, 0)
        B_2 = utils.finegrain(B_2, b, 0)
        B_3 = utils.finegrain(B_3, b, 0)
        B_4 = utils.finegrain(B_4, b, 0)
        # Recombine the quarter chunks into one tensor
        prev_model_dict['B'] = torch.cat((B_1, B_2, B_3, B_4), 0)
    if (4 * c[0]) != prev_model_dict['C'].size(
            0) or c[1] != prev_model_dict['C'].size(1):
        # Separate C into quarters
        C_1, C_2, C_3, C_4 = torch.chunk(prev_model_dict['C'], 4, 0)
        # Finegrain each quarter separately
        C_1 = utils.finegrain(C_1, c, 0)
        C_2 = utils.finegrain(C_2, c, 0)
        C_3 = utils.finegrain(C_3, c, 0)
        C_4 = utils.finegrain(C_4, c, 0)
        # Recombine the quarter chunks into one tensor
        prev_model_dict['C'] = torch.cat((C_1, C_2, C_3, C_4), 0)

    # Train
    # hyper['lr'] = multi.best_lr / ((b[0] / prev_b[0]) * (c[0] / prev_c[0]))
    hyper['lr'] = multi.best_lr
    multi = BasketballMulti(device)
    multi.init_low_model(train_set, hyper['K'])
    multi.model.load_state_dict(prev_model_dict)
    multi.init_params(**hyper)
    multi.init_loaders(train_set, val_set)
    multi.train_and_evaluate(save_dir)

    # Draw heatmaps
    B_1, B_2, B_3, B_4 = torch.chunk(prev_model_dict['B'], 4, 0)
    C_1, C_2, C_3, C_4 = torch.chunk(prev_model_dict['C'], 4, 0)
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap_1.png".format(b_str, c_str))
    plot.latent_factor_heatmap(B_1, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap_2.png".format(b_str, c_str))
    plot.latent_factor_heatmap(B_2, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap_3.png".format(b_str, c_str))
    plot.latent_factor_heatmap(B_3, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_B_heatmap_4.png".format(b_str, c_str))
    plot.latent_factor_heatmap(B_4, cmap='RdBu_r', draw_court=True,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap_1.png".format(b_str, c_str))
    plot.latent_factor_heatmap(C_1, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap_2.png".format(b_str, c_str))
    plot.latent_factor_heatmap(C_2, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap_3.png".format(b_str, c_str))
    plot.latent_factor_heatmap(C_3, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
    fp_fig = os.path.join(fig_dir,
                          "low_{0},{1}_C_heatmap_4.png".format(b_str, c_str))
    plot.latent_factor_heatmap(C_4, cmap='RdBu_r', draw_court=False,
                               fp_fig=fp_fig)
    
    # Bug fix
    b = [b[0]//4, b[1]]
    c = [c[0]//4, c[1]]

    # Test
    # Create dataset
    test_set.calculate_pos(b, c)
    multi.model.load_state_dict(multi.best_model_dict)
    test_conf_matrix, test_acc, test_precision, test_recall, test_F1, test_out, test_labels = multi.test(
        test_set)

    # Metrics
    results['best_epochs'].append(multi.best_epochs)
    results['best_lr'].append(multi.best_lr)
    results['train_times'].append(multi.train_times[:multi.best_epochs])
    results['train_loss'].append(multi.train_loss[:multi.best_epochs])
    results['grad_norms'].append(multi.grad_norms[:multi.best_epochs])
    results['grad_entropies'].append(multi.grad_entropies[:multi.best_epochs])
    results['grad_vars'].append(multi.grad_vars[:multi.best_epochs])
    results['val_times'].append(multi.val_times[:multi.best_epochs])
    results['val_loss'].append(multi.val_loss[:multi.best_epochs])
    results['val_conf_matrix'].append(
        multi.val_conf_matrix[:multi.best_epochs])
    results['val_acc'].append(multi.val_acc[:multi.best_epochs])
    results['val_precision'].append(multi.val_precision[:multi.best_epochs])
    results['val_recall'].append(multi.val_recall[:multi.best_epochs])
    results['val_F1'].append(multi.val_F1[:multi.best_epochs])
    results['test_conf_matrix'].append(test_conf_matrix)
    results['test_acc'].append(test_acc)
    results['test_precision'].append(test_precision)
    results['test_recall'].append(test_recall)
    results['test_F1'].append(test_F1)
    results['test_out'].append(test_out)
    results['test_labels'].append(test_labels)

    prev_b = b
    prev_c = c

if args.type == 'multi' or args.type == 'fixed':
    # Draw loss curve for low rank
    fp_fig = os.path.join(fig_dir, "low_time_vs_loss.png")
    plot.loss_time(results['train_times'][results['low_start_idx']:],
                   results['train_loss'][results['low_start_idx']:],
                   results['val_times'][results['low_start_idx']:],
                   results['val_loss'][results['low_start_idx']:],
                   fp_fig=fp_fig)

# Draw loss curve of all
fp_fig = os.path.join(fig_dir, "all_time_vs_loss.png")
plot.loss_time(results['train_times'],
               results['train_loss'],
               results['val_times'],
               results['val_loss'],
               low_index=results['low_start_idx'],
               fp_fig=fp_fig)

# Draw F1 scores of all
fp_fig = os.path.join(fig_dir, "all_time_vs_F1.png")
plot.F1_time(results['val_times'],
             results['val_F1'],
             low_index=results['low_start_idx'],
             fp_fig=fp_fig)

# Save results
torch.save(results, os.path.join(save_dir, "results.pt"))

main_logger.info('FINISH')
