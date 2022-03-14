
import os
from collections import Counter
import random
import numpy as np
import argparse


n_action ='A'
n_camid='C'
n_bgcolor ='B'
n_dress='D'
n_person='P'
n_sample='S'
nkeys =[n_action, n_camid, n_bgcolor, n_dress, n_person, n_sample]

cond_ac= 'ac' #  action+cam
cond_a= 'a' # ation
cond_c = 'c'


parser = argparse.ArgumentParser(description='[korea dance] Video conversion')
parser.add_argument('-i', '--input_root', help='absolute path to input video ',
         type=str, default='/home/peter/workspace/dataset/kor_dance')

parser.add_argument('-o', '--output_root', help='location of output list files', type=str
        , default= '/home/peter/workspace/dataset/kor_dance')

parser.add_argument('-c', '--condition', help='location of output list files', type=str,
                        choices=['ac', 'a'], default= 'ac')

parser.add_argument('-r', '--ratio', help='ratio of training samples of dataset', type=float, default= 0.66)

args = parser.parse_args()


def get_tag(video):
        tmp = []
        for k in nkeys:
            tmp.append( video[video.index(k) : video.index(k)+4])
        return tmp


def get_key(video, cond):
    a,c,b,d,p,s = get_tag(video)

    if cond == cond_ac:
        return a+c
    elif cond == cond_a:
        return a
    else:
        return c

#-------------------------------
def get_unique_label_dict(video_list, key):
     
    video_dict = {}

    for _video in video_list:
        video = os.path.basename(_video)
        a,c,_,_,_,_ = get_tag(video)

        if key== cond_ac:
            k = a+c
        elif key == cond_a:
            k =a
        else:
            k= c

        if k in video_dict.keys():
            tmp = video_dict[k]
            video_dict[k] = tmp+[_video]
        else:
            video_dict[k] = [_video]

    return video_dict

def split_acdict_to_train_test(ac_dict, ratio):
    ac_keys = list(ac_dict.keys())

    train_ac_list, valid_ac_list = [] ,[]
    for ac_key in ac_keys:

        ac_video_list = ac_dict[ac_key]
        n_vidoes = len(ac_video_list)
        n_train = int(n_vidoes*ratio)
        selected_idx = np.random.choice(n_vidoes,n_train, replace=False)

        for idx, v in enumerate(ac_video_list):
            if idx in selected_idx:
                train_ac_list.append(v)
            else:
                valid_ac_list.append(v)

    return train_ac_list, valid_ac_list


def _merge_labeling(train_ac_list, valid_ac_list, cond):

    train_a_dict = get_unique_label_dict(train_ac_list, cond)
    valid_a_dict = get_unique_label_dict(valid_ac_list, cond)
    label_list = list(set(list(train_a_dict.keys()) + list(valid_a_dict.keys())))
    return label_list
  
def relabeling(label_list, ac_list,  cond):

    nlist = []
    for v in ac_list:
        a,c,_,_,_,_ = get_tag(v)
        if cond == cond_ac:
            label_ind = label_list.index(a+c)
        elif cond == cond_a:
            label_ind = label_list.index(a)
        else:
            label_ind = label_list.index(c)
        # nlist.append(str(label_ind)+' '+os.path.join(videopath,v))
        nlist.append(str(label_ind)+' '+v)
    return nlist

def write_txt(split, vlist):        
    txt_path = os.path.join(args.output_root, f'{split}_list.txt')
    with open(txt_path,'w') as f:
        for x in vlist:
            f.write(x+'\n')
    print(split, '# of samples: ', len(vlist))      

def count_samples(labels, train, valid):

    clist1 = []
    for line in train:
        ll = line.rstrip().split(' ')[0]
        clist1.append(ll)

    clist2 = []
    for line in valid:
        ll = line.rstrip().split(' ')[0]
        clist2.append(ll)

    cnt1 = Counter(clist1)
    cnt2 = Counter(clist2)
    print(' ------ | ----- | -----')
    print(' Labels | Train | Valid')
    print(' ------ | ----- | -----')

    for l in labels:
        k = str(labels.index(l))
        v1 = str(cnt1[k]).zfill(2) if k in cnt1.keys() else '00'
        v2 = str(cnt2[k]).zfill(2) if k in cnt2.keys() else '00'
    
        print(f'  {l}  |  #{v1}  |  #{v2}  ')
    # print(Counter(clist))


def doing():
    
    
    raw_list = [os.path.join(args.input_root, v) for v in os.listdir(args.input_root)]

    ac_dict = get_unique_label_dict(raw_list, cond_ac)
    train_ac_list, valid_ac_list = split_acdict_to_train_test(ac_dict, args.ratio)

    cond = args.condition
    a_label_list = _merge_labeling(train_ac_list, valid_ac_list,cond)
    # train_a_list = relabeling(a_label_list, train_ac_list, videopath, cond)
    # valid_a_list = relabeling(a_label_list, valid_ac_list, videopath, cond)
    train_a_list = relabeling(a_label_list, train_ac_list,  cond)
    valid_a_list = relabeling(a_label_list, valid_ac_list,  cond)


    print('Action # labels: ' , len(a_label_list))
    write_txt('train', train_a_list)
    write_txt('valid', valid_a_list)
    count_samples(a_label_list, train_a_list, valid_a_list)

    with open(os.path.join(args.output_root, f'class_ind.txt'),'w') as f:
        for idx,kx in enumerate(a_label_list):
            f.write(f'{idx} {kx}\n')




if __name__ =='__main__':
    print('===> Creating split list of ')
    doing()

    
