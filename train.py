#!/usr/bin/env python
"""
For Evaluation
Extended from ADNet code by Hansen et al.
"""
import os
import random
import logging
import shutil

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.fewshot_cnn_kl import FewShotSeg
from models.fewshot_sam_kl import FewShotSegSAM, visualize_five_panels
# from dataloaders.datasets import TrainDataset as TrainDataset
from dataloaders.datasets_ct import TrainDataset as TrainDataset
from torch.nn import functional as F
import matplotlib.pyplot as plt
from utils import *
from config import ex
import torch.optim as optim
import numpy as np
from utils import get_criterion
from utils import CrossArchitectureKDLoss
from torchvision.utils import make_grid
import torchvision
from tensorboardX import SummaryWriter
from torchvision.utils import draw_bounding_boxes, make_grid


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    snapshot_path = _run.observers[0].dir
    image_path = os.path.join(snapshot_path, 'image')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    writer = SummaryWriter(snapshot_path + '/log')

    _log.info(f'Create model...')
    model = FewShotSeg()
    model = model.cuda()
    model.train()

    model_sam = FewShotSegSAM(_config)
    model_sam = model_sam.cuda()
    model_sam.train()

    _log.info(f'Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * _config['max_iters_per_load'] for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    if _config['warmup']:  # true
        b_lr = _config['base_lr'] / _config['warmup_period']
    else:
        b_lr = _config['base_lr']

    if _config['AdamW']:  # true
        optimizer_sam = optim.AdamW(filter(lambda p: p.requires_grad, model_sam.parameters()), lr=b_lr, betas=(0.9, 0.999),
                                weight_decay=0.1)
    else:
        optimizer_sam = optim.SGD(filter(lambda p: p.requires_grad, model_sam.parameters()), lr=b_lr, momentum=0.9,
                              weight_decay=0.0001)


    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)
    criterion_sam = get_criterion()  # SAMUS
    kd_criterion = CrossArchitectureKDLoss(T=10.0)

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=_config['batch_size'],
                              shuffle=True,
                              num_workers=_config['num_workers'],
                              pin_memory=True,
                              drop_last=True)

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']  # number of times for reloading
    log_loss = {'total_loss': 0, 'query_loss': 0, 'align_loss': 0, 'sam_loss': 0, 'loss_kd_sam2cnn': 0, 'loss_kd_cnn2sam': 0}

    i_iter = 0
    _log.info(f'Start training...')
    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')

        for _, sample in enumerate(train_loader):

            # Prepare episode data.
            support_images = [[shot.float().cuda() for shot in way]
                              for way in sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way]
                               for way in sample['support_fg_labels']]

            query_images = [query_image.float().cuda() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)


            # Compute outputs and losses.
            query_pred, align_loss, gamma_1, gamma_2 = model(support_images, support_fg_mask, query_images, query_labels, train=True)
            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                        1 - torch.finfo(torch.float32).eps)), query_labels)

            query_pred_sam, region_ids, pos_points, neg_points, K, maps, box_np, uncertainty_area, R_gradient, fusion_map =\
                model_sam(support_images, support_fg_mask, query_images, query_labels, query_pred,  isval= False)

            sam_query_loss = criterion_sam(query_pred_sam, sample['query_labels'].to(query_pred_sam.device)) #torch.Size([1, 1, 256, 256]) torch.Size([1, 1, 256, 256])

            loss_kd_sam2cnn = kd_criterion(query_pred, query_pred_sam, direction="sam_to_cnn")
            loss_kd_cnn2sam = kd_criterion(query_pred, query_pred_sam, direction="cnn_to_sam")

            loss = query_loss + align_loss + 0.10 * loss_kd_sam2cnn
            sam_loss = sam_query_loss + 0.10 * loss_kd_cnn2sam

            for param in model_sam.parameters():
                  param.grad = None

            sam_loss.backward()
            optimizer_sam.step()

            # warmup=True、warmup_period=250
            if _config['warmup'] and i_iter < _config['warmup_period']:
                lr_ = _config['base_lr'] * ((i_iter + 1) / _config['warmup_period'])
                for param_group in optimizer_sam.param_groups:
                    param_group['lr'] = lr_
            else:
                if _config['warmup']:
                    shift_iter = i_iter - _config['warmup_period']
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = i_iter
                lr_ = _config['base_lr'] * (
                            1.0 - shift_iter / _config['n_steps']) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer_sam.param_groups:
                    param_group['lr'] = lr_

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy()
            sam_loss = sam_loss.detach().data.cpu().numpy()
            loss_kd_cnn2sam = loss_kd_cnn2sam.detach().data.cpu().numpy()
            loss_kd_sam2cnn = loss_kd_sam2cnn.detach().data.cpu().numpy()

            _run.log_scalar('total_loss', loss.item())
            _run.log_scalar('query_loss', query_loss)
            _run.log_scalar('align_loss', align_loss)
            _run.log_scalar('sam_loss', sam_loss)
            _run.log_scalar('loss_kd_cnn2sam', loss_kd_cnn2sam)
            _run.log_scalar('loss_kd_sam2cnn', loss_kd_sam2cnn)

            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss
            log_loss['align_loss'] += align_loss
            log_loss['sam_loss'] += sam_loss
            log_loss['loss_kd_cnn2sam'] += loss_kd_cnn2sam
            log_loss['loss_kd_sam2cnn'] += loss_kd_sam2cnn

            # Print loss and take snapshots.
            if (i_iter + 1) % _config['visual_print_interval'] == 0 and region_ids > 0:

                grid_image = make_grid(support_images[0][0], normalize=True)
                writer.add_image('train/Image', grid_image, i_iter)
                torchvision.utils.save_image(grid_image, os.path.join(image_path, '{}_image_support.jpg'.format(
                    i_iter)))

                grid_image = make_grid(support_fg_mask[0][0].float().unsqueeze(1), normalize=False)
                writer.add_image('train/Image', grid_image, i_iter)
                torchvision.utils.save_image(grid_image, os.path.join(image_path, '{}_image_support_gt.jpg'.format(
                    i_iter)), normalize=True)

                grid_image = make_grid(query_images[0], normalize=True)
                writer.add_image('train/Image', grid_image, i_iter)
                torchvision.utils.save_image(grid_image, os.path.join(image_path, '{}_image_query.jpg'.format(
                    i_iter)))

                grid_image = make_grid(query_labels.float().unsqueeze(1), normalize=False)
                writer.add_image('train/Image', grid_image, i_iter)
                torchvision.utils.save_image(grid_image, os.path.join(image_path, '{}_image_query_gt.jpg'.format(
                    i_iter)), normalize=True)

                outputs = torch.argmax(query_pred, dim=1, keepdim=True)
                grid_image = make_grid(outputs.float(), normalize=False)
                writer.add_image('train/Image', grid_image, i_iter)
                torchvision.utils.save_image(grid_image,
                                             os.path.join(image_path, '{}_image_query_pred_cnn.jpg'.format(i_iter)),
                                             normalize=True)

                outputs = torch.sigmoid(query_pred_sam)
                outputs = (outputs > 0.5).float()
                grid_image = make_grid(outputs * 50., 6, padding=10, normalize=False, pad_value=255)
                writer.add_image('train/Image', grid_image, i_iter)
                torchvision.utils.save_image(grid_image,
                                             os.path.join(image_path,
                                                          '{}_image_query_pred_sam.jpg'.format(i_iter)),
                                             normalize=True)

                grid_image = make_grid(uncertainty_area.float(), normalize=False)
                writer.add_image('train/Image', grid_image, i_iter)
                torchvision.utils.save_image(grid_image, os.path.join(image_path, '{}_image_query_uncertainty_area.jpg'.format(
                    i_iter)), normalize=True)

                grid_image = make_grid(R_gradient.float(), normalize=False)
                writer.add_image('train/Image', grid_image, i_iter)
                torchvision.utils.save_image(grid_image,
                                             os.path.join(image_path, '{}_image_query_morphological_gradient.jpg'.format(
                                                 i_iter)), normalize=True)

                outputs = torch.softmax(fusion_map, dim=1)
                outputs = torch.argmax(outputs, dim=1, keepdim=True)
                grid_image = make_grid(outputs.float(), normalize=False)
                writer.add_image('train/Image', grid_image, i_iter)
                torchvision.utils.save_image(grid_image,
                                             os.path.join(image_path, '{}_image_query_fusion_map.jpg'.format(i_iter)),
                                             normalize=True)

                visualize_five_panels(query_images[0][0][0].detach().cpu().numpy(), maps, pos_points, neg_points, K, box_np,image_path,i_iter)


            # Print loss and take snapshots.
            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']
                sam_loss = log_loss['sam_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0
                log_loss['align_loss'] = 0
                log_loss['sam_loss'] = 0
                log_loss['loss_kd_sam2cnn'] = 0
                log_loss['loss_kd_cnn2sam'] = 0

                _log.info(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss},'
                          f' align_loss: {align_loss}, sam_loss: {sam_loss}, loss_kd_sam2cnn: {loss_kd_sam2cnn}, loss_kd_cnn2sam: {loss_kd_cnn2sam}, gamma_1: {gamma_1}, gamma_2: {gamma_2}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}_cnn.pth'))
                torch.save(model_sam.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}_sam.pth'))

            i_iter += 1
    _log.info('End of training.')
    return 1
