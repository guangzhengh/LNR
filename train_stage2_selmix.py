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
from sklearn.metrics import confusion_matrix
#selmix#
from datasets.sampler import FastJointSampler
from utils_selmix import get_metrics
from datasets.data_utils_selmix import get_data_loader
from utils.metrics_selmix import *
from torch.distributions.uniform import Uniform

class selmix:
    def __init__(self):
        self.num_classes = 10
        self.lambdas = [1 / self.num_classes] * self.num_classes      
        self.temperature = 0.1
        self.T = 1.0
        self.alpha = 0.95
        self.beta = 0.0
        self.val_lr = 50 # 25 for cifar100
        self.tau =  0.01
        self.lambda_max = 100
        self.MaxGain = [1000]
        self.cit = 0
        self.num_train_iter = 100000000
        self.MixupSampler = None
        self.P = None

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
                             batch_size=config.batch_size, num_works=config.workers, config = config)

    elif config.dataset == 'cifar100':
        dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers, config = config)

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
    sel_val = dataset.val
    print(sel_val.get_cls_num_list())
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
    selm = selmix()
    selm.num_classes = config.num_classes
    selm.num_train_iter = config.num_epochs * 10000
    bepoch = 0
    for epoch in range(config.num_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)

        if config.dataset != 'places':
            block = None
        # train for one epoch
        selm = train(train_loader,train_loader_all, train_dataset, model, classifier, lws_model, criterion, optimizer, epoch, config, logger,model_n,classifier_n,lws_model_n, block, is_best, cls_num_list, selm = selm, sel_val = sel_val)

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


def feedforward(model, classifier, lws_model, dataloader, gpu, return_feats=True):
    model.eval()
    classifier.eval()
    preds, labels, features = [], [], []
    with torch.no_grad():
        for index, x, y in dataloader:
            x, y = x.cuda(gpu), y.cuda(gpu)
            feats = model(x)
            out = classifier(feats)
            #out = lws_model(out)
            feats = feats.cpu().detach().tolist()
            features.extend(feats)
            preds.append(torch.argmax(out, dim=1).cpu().detach().numpy())
            labels.append(y.cpu().detach().numpy())
        preds, labels = (np.concatenate(preds, axis=0), np.concatenate(labels, axis=0))
    return labels, preds, features


def has_converged(MaxGain, window_size=10, tolerance_percentage=0.05):
    if len(MaxGain) < window_size:
        return False
    window = MaxGain[-window_size:]
    mean = sum(window) / window_size
    return abs(mean - MaxGain[-1]) < tolerance_percentage * abs(mean)
@torch.no_grad()
def selval(model, classifier, lws_model, val_data, objective_name, gpu, lb_dataset, batch_size, para):
    model.eval()
    classifier.eval()

    val_loader = get_data_loader(val_data, batch_size= 256)
    
    labels, preds, features = feedforward(model, classifier, lws_model, val_loader, gpu, return_feats=True)
    prototypes = np.zeros((para.num_classes, classifier.module.fc.in_features))
    features = np.array(features)
    for class_label in range(para.num_classes):
        feats_for_class = features[labels == class_label]
        prototypes[class_label] = np.mean(feats_for_class, axis=0)
    classes =  [str(i) for i in range(para.num_classes)]
    val_metrics, CM = get_metrics(preds, labels, classes, tag="val/")
    if objective_name == "mean_recall":
        objective = MeanRecall(CM, prototypes, model, classifier, lws_model, para.temperature)
    elif objective_name == "min_recall":
        objective = MinRecall(CM, prototypes, model, classifier, lws_model, para.temperature,para.lambdas, para.beta, para.val_lr)
        para.lambdas = objective.lambdas
    elif objective_name == "min_HT_recall":
        objective = MinHTRecall(CM, prototypes, model, classifier, lws_model, para.temperature,\
                       para.lambdas, para.beta, para.val_lr)
        para.lambdas = objective.lambdas
    elif objective_name == "mean_recall_min_coverage":
        objective = MeanRecallWithCoverage(CM, prototypes, model,classifier, lws_model,\
                                    para.temperature, para.lambdas,\
                                    alpha=para.alpha, tau=para.tau,
                                    lambda_max=para.lambda_max)
        para.lambdas = objective.lambdas
    elif objective_name == "mean_recall_min_HT_coverage":
        objective = MeanRecallWithHTCoverage(CM, prototypes, model,classifier, lws_model,\
                                      para.temperature, para.lambdas,
                                      alpha=para.alpha, tau=para.tau,
                                      lambda_max=para.lambda_max)
        para.lambdas = objective.lambdas

    elif objective_name == "g_mean":
        objective = Gmean(CM, prototypes, model, classifier, lws_model,para.temperature)
    elif objective_name == "h_mean":
        objective = Hmean(CM, prototypes, model, classifier, lws_model,para.temperature)
    elif objective_name == "HM_min_coverage":
        objective = HmeanWithCoverage(CM, prototypes, model,classifier, lws_model,\
                            para.temperature,\
                            alpha=para.alpha, tau=para.tau,
                            lambda_max=para.lambda_max)
        para.lambdas = objective.lambdas
    para.P = objective.P
    para.MaxGain.append(np.max(objective.G))
    #if has_converged(para.MaxGain):
    #    para.cit = para.num_train_iter -1
    JS = FastJointSampler(lb_dataset, lb_dataset, model, classifier, lws_model,
                          samp_dist=para.P, batch_size=batch_size, 
                          semi_supervised=False)
    objective = objective
    para.MixupSampler= JS
    print(" val Acc:", para.cit, "is : ", val_metrics["val/mean_recall"])
    return val_metrics, para

def train(train_loader,train_loader_all, train_dataset, model, classifier, lws_model, criterion, optimizer, epoch, config, logger,model_n,classifier_n,lws_model_n, block=None, is_best = 0, cls_num_list = None, selm = None, sel_val = None):
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
    if epoch == 0:
        val_metrics, selm = selval(model, classifier, lws_model, sel_val, 'min_recall', config.gpu, train_dataset, config.batch_size, selm)     

    for i, (index, images, target) in enumerate(train_loader):
        if i > end_steps:
            break
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)

        if config.mixup is True:
            model.train()
            classifier.train()
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    #print("changing")
                    module.momentum = 0.00
                    module.track_running_stats = False
                    module.requires_grad_ = False
            
            if selm.cit % 50 == 0:
                val_metrics, selm = selval(model, classifier, lws_model, sel_val, 'min_recall', config.gpu, train_dataset, config.batch_size, selm)
                model.train()
                classifier.train()

            x_lb_MO, y_lb_MO, u_w_MO, y_pl_MO = selm.MixupSampler.get_batch()
            x_lb_MO, y_lb_MO, u_w_MO, y_pl_MO = x_lb_MO.cuda(config.gpu), \
                                               y_lb_MO.cuda(config.gpu), \
                                               u_w_MO.cuda(config.gpu), \
                                               y_pl_MO.cuda(config.gpu)

            num_lb = x_lb_MO.shape[0]
            num_ulb = u_w_MO.shape[0]
            assert num_ulb == num_lb
            features1 = model(x_lb_MO)
            features2 = model(u_w_MO)
            thisbatch_size = features2.shape[0] # type: ignore
            mixup_coeff = Uniform(0.6,1).sample([thisbatch_size]).cuda() # type: ignore
            feat = (features1.T * mixup_coeff).T + (features2.T * (1-mixup_coeff)).T 
            output = classifier(feat.detach())
            #output = lws_model(output)
  
            loss = F.cross_entropy(output / selm.T, y_lb_MO, reduction="mean")

            selm.cit += 1
        
        else:
            # compute output
            with torch.no_grad():
                if config.dataset == 'places':
                    feat = block(model(images))
                else:
                    feat = model(images)
            output = classifier(feat.detach())
            #output = lws_model(output)
            loss = criterion(output, target)
        if config.mixup:
            feat = model(images)
            output = classifier(feat.detach())
            #output = lws_model(output)

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
    return selm


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
            #output = lws_model(output)
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
