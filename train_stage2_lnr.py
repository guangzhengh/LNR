import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pprint
import math
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
from datasets.places import Places_LT
from datasets.imagenet import ImageNet_LT
from datasets.ina2018 import iNa2018

from models import resnet
from models import resnet_places
from models import resnet_cifar

from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration

from methods import mixup_data, mixup_criterion
from methods import LabelAwareSmoothing, LearnableWeightScaling
from utils.metrics_selmix import *
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-2)')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


best_acc1 = 0
its_ece = 100


def main():

    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    if config.deterministic:
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, logger, model_dir)


def main_worker(gpu, ngpus_per_node, config, logger, model_dir):
    global best_acc1, its_ece
    config.gpu = gpu
#     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    if config.dataset == 'cifar10' or config.dataset == 'cifar100':
        model = getattr(resnet_cifar, config.backbone)()
        classifier = getattr(resnet_cifar, 'Classifier')(feat_in=64, num_classes=config.num_classes)

    elif config.dataset == 'imagenet' or config.dataset == 'ina2018':
        model = getattr(resnet, config.backbone)()
        classifier = getattr(resnet, 'Classifier')(feat_in=2048, num_classes=config.num_classes)

    elif config.dataset == 'places':
        model = getattr(resnet_places, config.backbone)(pretrained=True)
        classifier = getattr(resnet_places, 'Classifier')(feat_in=2048, num_classes=config.num_classes)
        block = getattr(resnet_places, 'Bottleneck')(2048, 512, groups=1, base_width=64,
                                                     dilation=1, norm_layer=nn.BatchNorm2d)

    lws_model = LearnableWeightScaling(num_classes=config.num_classes)

    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            classifier.cuda(config.gpu)
            lws_model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[config.gpu])
            lws_model = torch.nn.parallel.DistributedDataParallel(lws_model, device_ids=[config.gpu])

            if config.dataset == 'places':
                block.cuda(config.gpu)
                block = torch.nn.parallel.DistributedDataParallel(block, device_ids=[config.gpu])
        else:
            model.cuda()
            classifier.cuda()
            lws_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier)
            lws_model = torch.nn.parallel.DistributedDataParallel(lws_model)
            if config.dataset == 'places':
                block.cuda()
                block = torch.nn.parallel.DistributedDataParallel(block)

    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        classifier = classifier.cuda(config.gpu)
        lws_model = lws_model.cuda(config.gpu)
        if config.dataset == 'places':
            block.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()
        lws_model = torch.nn.DataParallel(lws_model).cuda()
        if config.dataset == 'places':
            block = torch.nn.DataParallel(block).cuda()

    # optionally resume from a checkpoint
    
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.resume, map_location=loc)
            # config.start_epoch = checkpoint['epoch']
            print(checkpoint.keys())
            best_acc1 = checkpoint['best_acc1']
            its_ece = checkpoint['its_ece']
            if config.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(config.gpu)
            model.load_state_dict(checkpoint['state_dict_model'])
            classifier.load_state_dict(checkpoint['state_dict_classifier'])
            lws_model.load_state_dict(checkpoint['state_dict_lws_model'])
            if config.dataset == 'places':
                block.load_state_dict(checkpoint['state_dict_block'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

        if os.path.isfile(config.noisemodel):
            logger.info("=> loading checkpoint '{}'".format(config.noisemodel))
            if config.gpu is None:
                checkpoint = torch.load(config.noisemodel)
            else:
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.noisemodel, map_location=loc)
            model_n = copy.deepcopy(model)
            classifier_n = copy.deepcopy(classifier)
            lws_model_n = copy.deepcopy(lws_model)
            model_n.load_state_dict(checkpoint['state_dict_model'])
            classifier_n.load_state_dict(checkpoint['state_dict_classifier'])
            lws_model_n.load_state_dict(checkpoint['state_dict_lws_model'])

    # Data loading code
    if config.dataset == 'cifar10':
        dataset = CIFAR10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                             batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'cifar100':
        dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'places':
        dataset = Places_LT(config.distributed, root=config.data_path,
                            batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'imagenet':
        dataset = ImageNet_LT(config.distributed, root=config.data_path,
                              batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'ina2018':
        dataset = iNa2018(config.distributed, root=config.data_path,
                          batch_size=config.batch_size, num_works=config.workers)

    train_loader = dataset.train_balance
    train_loader_all = dataset.train_balance
    val_loader = dataset.eval
    cls_num_list = dataset.cls_num_list
    train_dataset = dataset.train_dataset
    if config.distributed:
        train_sampler = dataset.dist_sampler

    # define loss function (criterion) and optimizer

    criterion = LabelAwareSmoothing(cls_num_list=cls_num_list, smooth_head=config.smooth_head,
                                    smooth_tail=config.smooth_tail).cuda(config.gpu)

    optimizer = torch.optim.SGD([{"params": classifier.parameters()},
                                {'params': lws_model.parameters()}], config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    is_best = 1
    best_acc1 = 0
    bepoch = 0
    for epoch in range(config.num_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)

        if config.dataset != 'places':
            block = None
        # train for one epoch
        train_lnr(train_loader,train_loader_all, model, classifier, lws_model, criterion, optimizer, epoch, config, logger,model_n,classifier_n,lws_model_n, block, is_best, cls_num_list)

        # evaluate on validation set
        acc1, ece = validate(val_loader, model, classifier, lws_model, criterion, config, logger, block)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            its_ece = ece
            bepoch = epoch
        logger.info('Best Prec@1: %.3f%% ECE: %.3f%% at round %.3f%%\n' % (best_acc1, its_ece, bepoch))
        if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                      and config.rank % ngpus_per_node == 0):
            if config.dataset == 'places':
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_classifier': classifier.state_dict(),
                    'state_dict_block': block.state_dict(),
                    'state_dict_lws_model': lws_model.state_dict(),
                    'best_acc1': best_acc1,
                    'its_ece': its_ece,
                }, is_best, model_dir)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_classifier': classifier.state_dict(),
                    'state_dict_lws_model': lws_model.state_dict(),
                    'best_acc1': best_acc1,
                    'its_ece': its_ece,
                }, is_best, model_dir)



import pickle
def label_noise_rebalance(train_dataloader,net,classifier, unique_id, args, thre = 3, class_num = 100, read = False, store = False, dataset_name = 0):
    if not read and not store:
        return None
    if read:
        if os.path.exists('uid_'+str(unique_id)+'_noise_info_'+dataset_name+'.pkl'):
            print('read')
            with open('uid_'+str(unique_id)+'_noise_info_'+dataset_name+'.pkl','rb') as j:
                noise_info = pickle.load(j)
            return noise_info
        else:
            return None

    pre_dict = {}
    for e in range(2):
        print(e)
        for batch_idx, (index, x, target) in enumerate(train_dataloader):
            index, x, target = index, x.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
            feat = net(x)
            out = classifier(feat.detach())
            pospre = F.softmax(out, dim=1).cpu().detach().numpy()
            target = target.long()
            for i, value in enumerate(x.cpu()):
                if str(index[i]) in pre_dict:
                    pre_dict[str(index[i])].append((pospre[i],target.cpu().detach().numpy()[i]))
                else:
                    pre_dict[str(index[i])] = [(pospre[i],target.cpu().detach().numpy()[i])]

    targets = np.array([t[0][1] for t in pre_dict.values()])
    for k,v in pre_dict.items():
        pre_dict[k] = (np.mean(np.array([item[0] for item in v]),axis = 0), v[0][1])
    preds = np.array([t[0] for t in pre_dict.values()])
    classes = np.array(list(range(class_num)))
    cls,cls_cnt = np.unique(targets, return_counts=True)
    priors = []
    for c in classes:
        if c in cls:
            priors.append(cls_cnt[np.where(cls == c)[0]][0])
        else:
            priors.append(0)
    
    priors = np.array(priors)
    priors = 1-  (priors - np.min(priors))/(np.max(priors)-np.min(priors))
    preds_mean, preds_std = [],[]
    comp = {}
    for k in classes:
        comp[k] = {}

    for k in classes:
        if len(preds[np.where(targets != k)[0],k]) > 1:
            preds_mean.append(np.mean(preds[np.where(targets != k)[0],k]))
            preds_std.append(np.std(preds[np.where(targets != k)[0],k]))
        else:
            preds_mean.append(0)
            preds_std.append(-1)

    zscores = {}
    fliprate = {}
    noisecnt = 0
    noise_flag = {}    
    label_cnt = {key: 0 for key in range(len(classes))}   

    for key,value in pre_dict.items():
        (pred_prob, c) = value
        zscores[key] = (pred_prob - np.array(preds_mean)) / np.array(preds_std)
        prior_weight = np.array([max(p-priors[c],0) for p in priors])
        fliprate[key] = np.tanh((zscores[key]-thre))*prior_weight
        uniform_rand = np.random.rand(len(classes))
        noise_class = np.where(fliprate[key] > uniform_rand)[0]
        if len(noise_class):
            compare_flip = fliprate[key][noise_class]
            highest_flip = np.where(compare_flip == max(compare_flip))[0]
            noise_idx = noise_class[highest_flip]
            noisecnt += len(noise_idx)
            noise_flag[key] = classes[noise_idx]
            label_cnt[classes[noise_idx][0]] += 1
            if classes[noise_idx][0] not in comp[c]:
                comp[c][classes[noise_idx][0]] = 1
            else:
                comp[c][classes[noise_idx][0]] += 1
        else:
            label_cnt[c] += 1

    
    noise_info={}
    noise_info['noise_flag'] = noise_flag

    if store:
        with open('uid_'+str(unique_id)+'_noise_info_'+dataset_name+'.pkl', 'wb') as file:
            pickle.dump(noise_info, file)
    print('label flipping dict:', comp)
    print('total label noise:', noisecnt)
    print('after rebalancing:', label_cnt)
    return noise_info

def train_lnr(train_loader,train_loader_all, model, classifier, lws_model, criterion, optimizer, epoch, config, logger,model_n,classifier_n,lws_model_n, block=None, is_best = 0, cls_num_list = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    training_data_num = len(train_loader.dataset)
    end_steps = int(np.ceil(float(training_data_num) / float(train_loader.batch_size)))
    progress = ProgressMeter(
        end_steps,
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode

    if config.dataset == 'places':
        model.eval()
        if config.shift_bn:
            block.train()
        else:
            block.eval()
    else:
        if config.shift_bn:
            model.train()
        else:
            model.eval()
    classifier.train()

    end = time.time()
    if config.num_classes == 10:
        thre = 7.5
    else:
        thre = 14.5
    if epoch == 0:
        noise_info = label_noise_rebalance(train_loader_all,model,classifier,config.uid,
                                              config, thre = thre, class_num = config.num_classes, read = 0, store= 1, dataset_name=config.dataset)
    noise_info = label_noise_rebalance(train_loader_all,model,classifier,config.uid,
                                              config, thre = thre, class_num = config.num_classes, read = 1, store= 0, dataset_name=config.dataset)
    fflag = 0        
    for i, (index, images, target) in enumerate(train_loader):
        if i > end_steps:
            break
        if epoch >= 0:
            for j, value in enumerate(index):
                if str(index[j]) in noise_info['noise_flag'].keys():
                    fflag = fflag + 1
                    target[j] = noise_info['noise_flag'][str(index[j])][0]
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)

        if config.mixup is True:
            images, targets_a, targets_b, lam = mixup_data(images, target, alpha=config.alpha)
            with torch.no_grad():
                if config.dataset == 'places':
                    feat = block(model(images))
                else:
                    feat = model(images)
            output = classifier(feat.detach())
            output = lws_model(output)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam, n = cls_num_list)
        else:
            # compute output
            with torch.no_grad():
                if config.dataset == 'places':
                    feat = block(model(images))
                else:
                    feat = model(images)
            output = classifier(feat.detach())
            output = lws_model(output)
            loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)
    print('noise:', fflag)



def validate(val_loader, model, classifier, lws_model, criterion, config, logger, block=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()
    if config.dataset == 'places':
        block.eval()
    classifier.eval()
    class_num = torch.zeros(config.num_classes).cuda()
    correct = torch.zeros(config.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            if config.dataset == 'places':
                feat = block(model(images))
            else:
                feat = model(images)
            output = classifier(feat)
            #if not config.mixup:
            output = lws_model(output)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, config.num_classes)
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)
        cm = confusion_matrix(true_class, pred_class)
        acc_classes = correct / class_num
        head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100
        med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100

        logger.info('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

        cal = calibration(true_class, pred_class, confidence, num_bins=15)
        logger.info('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))
        print('* confusion matrix   : ', cm.diagonal())
        
    return top1.avg, cal['expected_calibration_error'] * 100


def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate"""
    lr_min = 0
    lr_max = config.lr
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))
    for idx, param_group in enumerate(optimizer.param_groups):
        if idx == 0:
            param_group['lr'] = config.lr_factor * lr
        else:
            param_group['lr'] = 1.00 * lr


if __name__ == '__main__':
    main()
