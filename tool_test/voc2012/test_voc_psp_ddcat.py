import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn

from model.pspnet import PSPNet_DDCAT, DeepLabV3_DDCAT
from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize


from torch import optim
from torch import autograd
import random
from torch.autograd import grad
import torchvision

from cospgd import functions as attack_funcs

cv2.ocl.setUseOpenCL(False)

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('--attack', action='store_true', help='evaluate the model with attack or not')
    parser.add_argument('--attack_name', type=str, default='BIM', help='name of the attack, default BIM, options: cospgd, segpgd, pgd')
    parser.add_argument('--attack_iterations', type=int, default=4, help='iterations for the attack')
    parser.add_argument('--attack_epsilon', type=float, default=0.03, help='epsilon for the attack')
    parser.add_argument('--attack_alpha', type=float, default=0.01, help='alpha for the attack')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id to be used')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None

    global attack_flag
    global attack_flag
    global attack_name
    global attack_iterations
    global attack_epsilon
    global attack_alpha
    global gpu_id
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpu_id)
    
    if args.attack:
        attack_flag = True
    else:
        attack_flag = False

    attack_name = args.attack_name
    attack_iterations = args.attack_iterations
    attack_epsilon = args.attack_epsilon
    attack_alpha = args.attack_alpha*255
    
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


@torch.autocast(device_type="cuda")
def BIM(input, target, model, normalize_layer=None):

    global attack_name, attack_iterations, attack_epsilon, attack_alpha
    
    orig_images = input.detach().clone()
    
    if 'pgd' in attack_name:
        input = attack_funcs.init_linf(images=input, epsilon=attack_epsilon, clamp_min=0, clamp_max=255)
    
    ignore_label = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none').cuda()

    adversarial_example = input.detach().clone()
    
    for itr in range(attack_iterations):
        model.zero_grad()
        adversarial_example.requires_grad = True
        result = model(normalize_layer(adversarial_example)) #NORMALIZE THE INPUT BEFORE PASSING IT TO THE MODEL
        loss = criterion(result, target.detach())
        if attack_name.lower()=='cospgd':
            loss = attack_funcs.cospgd_scale(predictions=result, labels=target, loss=loss, num_classes=21, targeted=False, one_hot=True)
        elif attack_name.lower()=='segpgd':
            loss = attack_funcs.segpgd_scale(predictions=result, labels=target, loss=loss, iteration=itr, iterations=attack_iterations, targeted=False)
        
        loss.mean().backward()
        adversarial_example = attack_funcs.step_inf(perturbed_image=adversarial_example, epsilon=attack_epsilon, data_grad=adversarial_example.grad, orig_image=orig_images, alpha=attack_alpha, targeted=False, clamp_min=0, clamp_max=255)

    return adversarial_example



def main():
    global args, logger, gpu_id
    args = get_parser()
    logger = get_logger()
    
    logger.info(args)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.test_h - 1) % 8 == 0 and (args.test_w - 1) % 8 == 0
    assert args.split in ['train', 'val', 'test']
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    global mean_origin
    global std_origin
    mean_origin = [0.485, 0.456, 0.406]
    std_origin = [0.229, 0.224, 0.225]
    
    if attack_flag:
        args.save_folder = os.path.join(args.save_folder, '{}_attack'.format(attack_name), 'itrs_{}'.format(attack_iterations), 'eps_{}'.format(attack_epsilon), 'alpha_{}'.format(attack_alpha))
    else:
        args.save_folder = os.path.join(args.save_folder, 'no_attack')


    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')

    test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)
    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(test_data.data_list)
    else:
        index_end = min(index_start + args.index_step, len(test_data.data_list))
    test_data.data_list = test_data.data_list[index_start:index_end]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]

    if not args.has_prediction:
        model = PSPNet_DDCAT(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
        logger.info(model)
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if os.path.isfile(args.model_path):
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.model_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        normalize_layer =  torchvision.transforms.Normalize(mean=mean, std=std)
        test(test_loader, test_data.data_list, model, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, gray_folder, color_folder, colors, normalize_layer=normalize_layer)
    if args.split != 'test':
        cal_acc(test_data.data_list, gray_folder, args.classes, names)


def net_process(model, image, target, mean, std=None, normalize_layer=None):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float() # PLEASE NOTE: IMAGE HERE ARE BETWEEN [0, 255]
    target = torch.from_numpy(target).long()

    input = input.unsqueeze(0).cuda()
    target = target.unsqueeze(0).cuda()

    if attack_flag:
        flip = False
    else:
        flip = True

    if flip:
        input = torch.cat([input, input.flip(3)], 0)
        target = torch.cat([target, target.flip(2)], 0)

    if attack_flag:
        adver_input = BIM(input, target, model, normalize_layer=normalize_layer)
        with torch.no_grad():
            output = model(normalize_layer(adver_input)) #NORMALIZE THE INPUT BEFORE PASSING IT TO THE MODEL
    else:
        with torch.no_grad():
            output = model(normalize_layer(input)) #NORMALIZE THE INPUT BEFORE PASSING IT TO THE MODEL

    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)

    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, target, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3, normalize_layer=None):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
        target = cv2.copyMakeBorder(target, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=255)

    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()

            target_crop = target[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, target_crop, mean, std, normalize_layer=normalize_layer)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors, normalize_layer=None):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        color_path = os.path.join(color_folder, image_name + '.png')

        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        target = np.squeeze(target.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))

        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)

            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            target_scale = cv2.resize(target.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            prediction += scale_process(model, image_scale, target_scale, classes, crop_h, crop_w, h, w, mean, std, normalize_layer=normalize_layer)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))


if __name__ == '__main__':
    main()
