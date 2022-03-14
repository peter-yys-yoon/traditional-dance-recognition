#!/usr/bin/env python3

import os
import concurrent.futures
from shutil import copyfile
import subprocess
import argparse

parser = argparse.ArgumentParser(description='[Something-Something-V2] Video conversion')
parser.add_argument('-i', '--data_root', help='location of root direcotry.', type=str)
parser.add_argument('-d', '--dataset', help='name of dataset, not a path', type=str)
args = parser.parse_args()

# input
train_file = os.path.join(args.data_root, args.dataset, 'train_list.txt')
val_file = os.path.join(args.data_root, args.dataset, 'valid_list.txt')
video_folder =os.path.join(args.data_root, 'videos')

# output
train_img_folder = os.path.join(args.data_root,args.dataset,'train')
val_img_folder = os.path.join(args.data_root,args.dataset,'val')
train_file_list = os.path.join(args.data_root,args.dataset,'train.txt')
val_file_list = os.path.join(args.data_root,args.dataset,'val.txt')


def load_video_list(file_path):
    videos = []

    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            label_name, video_id= line.split(" ")
            videos.append([video_id, label_name])
    return videos


train_videos = load_video_list(train_file)
val_videos = load_video_list(val_file)


def video_to_images(video, basedir, targetdir):

    cls_id= video[1]
    filename = video[0]
    video_basename = os.path.basename(filename).split('.')[0]
    output_foldername = os.path.join(targetdir, video_basename)

    if not os.path.exists(filename):
        print("{} is not existed.".format(filename))
        return video[0], cls_id, 0
    else:
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        command = ['ffmpeg',
                   '-i', '"%s"' % filename,
                   '-threads', '1',
                   '-vf','scale=360:-1',
                   '-loglevel', 'panic',
                   '-q:v', '0',
                   '{}/'.format(output_foldername) + '"%05d.jpg"']
        command = ' '.join(command)

        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except:
            print("fail to convert {}".format(filename))
            return video[0], cls_id, 0

        # get frame num
        i = 0
        while True:
            img_name = os.path.join(output_foldername + "/{:05d}.jpg".format(i + 1))
            if os.path.exists(img_name):
                i += 1
            else:
                break

        frame_num = i
        print("Finish {}, id: {} frames: {}".format(filename, cls_id, frame_num))
        return video_basename, cls_id, frame_num



def create_train_video():

    with open(train_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        
        futures = [executor.submit(video_to_images, video, os.path.join(video_folder, 'train'), train_img_folder)for video in train_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                # print("{} 1 {} {}".format(os.path.join(train_img_folder, video_id), frame_num, label_id), file=f, flush=True)
                print("{} 1 {} {}".format(os.path.join('train',video_id), frame_num, label_id), file=f, flush=True)
            
            print("{}/{}".format(curr_idx, total_videos), flush=True)
            curr_idx += 1
    print("Completed")


def create_val_video():
    with open(val_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, os.path.join(video_folder, 'val'), val_img_folder)
                   for video in val_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                # print("{} 1 {} {}".format(os.path.join(val_img_folder, video_id), frame_num, label_id), file=f, flush=True)
                print("{} 1 {} {}".format(os.path.join('val',video_id), frame_num, label_id), file=f, flush=True)

            print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")


create_train_video()
create_val_video()
