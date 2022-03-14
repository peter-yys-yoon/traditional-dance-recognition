
import numpy as np
import os
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm

from models import build_model
from utils.utils import build_dataflow, AverageMeter, accuracy
from utils.video_transforms import *
from utils.video_dataset import VideoDataSet
from utils.dataset_config import get_dataset_config
from opts import arg_parser
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def main():
 
    data_list_name = 'val.txt'

    DATA_ROOT= '/home/peter/dataset/kor_dance/kordance600_39'

    # Data loading code
    data_list = os.path.join(DATA_ROOT, data_list_name)


    augments = []
    if args.num_crops == 1:
        augments += [
            GroupScale(scale_size),
            GroupCenterCrop(args.input_size)
        ]
    else:
        flip = True if args.num_crops == 10 else False
        augments += [
            GroupOverSample(args.input_size, scale_size, num_crops=args.num_crops, flip=flip),
        ]
    augments += [
        Stack(threed_data=args.threed_data),
        ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
        GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
    ]

    augmentor = transforms.Compose(augments)
    val_dataset = VideoDataSet(DATA_ROOT, data_list)
    # val_dataset = VideoDataSet(args.datadir, data_list, args.groups, args.frames_per_group,
    #                              num_clips=args.num_clips, modality=args.modality,
    #                              image_tmpl=image_tmpl, dense_sampling=args.dense_sampling,
    #                              fixed_offset=not args.random_sampling,
    #                              transform=augmentor, is_train=False, test_mode=not args.evaluate,
    #                              seperator=filename_seperator, filter_video=filter_video)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=4,
                                 workers=0)

    total_batches = len(data_loader)
    label_list = []
    with torch.no_grad(), tqdm(total=total_batches) as t_bar:
        end = time.time()
        for i, (video, label) in enumerate(data_loader):
            label_list += label

    
    print(label_list)


RES_ROOT ='/home/peter/workspace/projects/tradance/traditional-dance-recognition/logs/kordance600_39-rgb-i3d-resnet-18-ts-f32'
class_ind_file ='/home/peter/dataset/kor_dance/kordance600_39/class_ind.txt'

# RES_ROOT ='/home/peter/workspace/projects/tradance/traditional-dance-recognition/logs/kordance600_13-rgb-i3d-resnet-18-ts-f32'
# class_ind_file ='/home/peter/dataset/kor_dance/kordance600_13/class_ind.txt'
label_file=f'{RES_ROOT}/labels.npy'
npy_file =f'{RES_ROOT}/val_1crops_3clips_224_details.npy'


def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation= 70)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def bb():
    pred_list = np.load(npy_file) # 600,39
    label_list = np.load(label_file)
    pred_list_top1 = np.argmax(pred_list, axis=1)

    with open(class_ind_file, 'r') as f:
        target_names = [ x.rstrip().split()[1] for x in f.readlines()]

    cm = confusion_matrix(label_list, pred_list_top1)
    plot_confusion_matrix(cm, target_names = target_names, labels=False)



def acc_per_class():
    pred_list = np.load(npy_file) # 600,39
    label_list = np.load(label_file)
    
    with open(class_ind_file, 'r') as f:
        target_names = [ x.rstrip().split()[1] for x in f.readlines()]

    N = len(target_names)

    pred_list_top1 = np.argmax(pred_list, axis=1)

    for n in range(N):
        
        n_idxs =np.where(label_list==n)

        n_pred_list = pred_list_top1[n_idxs]
        n_label_list = label_list[n_idxs]

        res = np.mean((n_pred_list==n_label_list).astype(np.int8))
        print(n, target_names[n] ,len(n_idxs[0]), f'{res:.2f}')





def aa():


    pred_list = np.load(npy_file) # 600,39
    label_list = np.load(label_file)
    pred_list_top1 = np.argmax(pred_list, axis=1)

    
    print(pred_list_top1.shape)

    print(pred_list_top1)
    corret =  (pred_list_top1 == label_list).astype(np.int32)

    print(corret)
    print(np.mean(corret))


if __name__ == '__main__':
    # bb()
    acc_per_class()










score = 0
for e in range(N): # einnings

    out_count = 0
    attackers = []
    while out_count< 3:

        for c in cands:
            state = board[e][c]

            if state == 0:
                out_count +=1
            elif state == 1:
                attackers = [a+1 for a in attackers] + [1]
            elif state == 2:
                attackers = [a+2 for a in attackers] + [2]
            elif state == 3:
                attackers = [a+3 for a in attackers] + [3]
            else:
                attackers = [a+4 for a in attackers] + [4]

            for at in attackers:
                if at > 3:
                    score +=1
                attackers.remove(at)










