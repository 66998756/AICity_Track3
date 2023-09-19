import os
import csv
import glob
import random
import itertools
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms as T
from torchvision.datasets.folder import make_dataset


def _times2sec(times: str, fps=30):
    times = times.split(':')
    target_time = (int(times[0]) * 60 * 60 + int(times[1]) * 60 + int(times[2]))
    return target_time


# 最後一筆csv讀入有問題: user_id_30932
class doubleDataset(Dataset):
    def __init__(self, root, epoch_size=None, mode=None, video_transform=None, clip_len=32, stride=0):

        self.clip_len = clip_len
        self.stride = stride
        self.video_transform = video_transform

        self.data_dict = {
            'dashbord': {'path': [], 'start_time': [], 'end_time': [], 'label': []}, 
            'right_window': {'path': [], 'start_time': [], 'end_time': [], 'label': []}
        }

        datapath = os.path.join(root, mode+'.csv')
        with open(datapath, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                # id 86952 例外處理
                if row[0] == 'path' or '86952' in row[0] or '86356' in row[0] or '30932' in row[0] or '28557' in row[0] or '47457' in row[0] or '60167' in row[0]: continue
                if '61962' in row[0]: continue
                if '63764' in row[0]: continue
                if 'Rear' in row[0]:
                    if row[0] == 'Rear_view_user_id_60768_NoAudio_5.MP4' and row[1] == '75' and row[2] == '94': continue
                    if row[0] == 'Rear_view_user_id_63513_NoAudio_5.MP4' and row[1] == '66' and row[2] == '86': continue # 1302行
                    if row[0] == 'Rear_view_user_id_63513_NoAudio_5.MP4' and row[1] == '109' and row[2] == '119': continue # 1303行
                    if row[0] == 'Rear_view_user_id_63513_NoAudio_7.MP4' and row[1] == '168' and row[2] == '174': continue # 1321行
                    if row[0] == 'Rear_view_user_id_63513_NoAudio_7.MP4' and row[1] == '189' and row[2] == '194': continue # 1322行
                    if row[0] == 'Rear_view_user_id_63764_NoAudio_5.MP4' and row[1] == '24' and row[2] == '28': continue # 1603行
                    if row[0] == 'Rear_view_user_id_63764_NoAudio_5.MP4' and row[1] == '47' and row[2] == '61': continue # 1608行
                    if row[0] == 'Rear_view_user_id_63764_NoAudio_5.MP4' and row[1] == '75' and row[2] == '91': continue # 1609行
                    self.data_dict['dashbord']['path'].append(os.path.join(root, row[0][row[0].find('user'):row[0].find('_No')], row[0]))
                    self.data_dict['dashbord']['start_time'].append(int(row[1]))
                    self.data_dict['dashbord']['end_time'].append(int(row[2]))
                    self.data_dict['dashbord']['label'].append(int(row[3].split(' ')[1]))
                if 'Right' in row[0]:
                    if row[0] == 'Right_side_window_user_id_60768_NoAudio_5.MP4' and row[1] == '75' and row[2] == '93': continue
                    if row[0] == 'Right_side_window_user_id_63513_NoAudio_5.MP4' and row[1] == '65' and row[2] == '86': continue # 1334行
                    if row[0] == 'Right_side_window_user_id_63513_NoAudio_5.MP4' and row[1] == '110' and row[2] == '120': continue # 1335行
                    if row[0] == 'Right_side_window_user_id_63513_NoAudio_7.MP4' and row[1] == '168' and row[2] == '172': continue # 1353行
                    if row[0] == 'Right_side_window_user_id_63513_NoAudio_7.MP4' and row[1] == '188' and row[2] == '193': continue # 1354行
                    if row[0] == 'Right_side_window_user_id_63764_NoAudio_5.MP4' and row[1] == '24' and row[2] == '26': continue # 1637行
                    if row[0] == 'Right_side_window_user_id_63764_NoAudio_5.MP4' and row[1] == '46' and row[2] == '61': continue # 1638行
                    if row[0] == 'Right_side_window_user_id_63764_NoAudio_5.MP4' and row[1] == '76' and row[2] == '92': continue # 1639行
                    self.data_dict['right_window']['path'].append(os.path.join(root, row[0][row[0].find('user'):row[0].find('_No')], row[0]))
                    self.data_dict['right_window']['start_time'].append(int(row[1]))
                    self.data_dict['right_window']['end_time'].append(int(row[2]))
                    self.data_dict['right_window']['label'].append(int(row[3].split(' ')[1]))

        self.datas = {
            'dashbord': {'path': [], 'start_time': [], 'label': []},
            'right_window': {'path': [], 'start_time': [], 'label': []},
        }
        for k in self.datas.keys():
            for i in range(len(self.data_dict[k]['path'])):
                for j in range(self.data_dict[k]['start_time'][i] * 30, (self.data_dict[k]['end_time'][i]-self.stride) * 30, self.clip_len*(self.stride+1)):
                    if self.clip_len*(self.stride+1) + self.data_dict[k]['start_time'][i] * 30 > (self.data_dict[k]['end_time'][i]-self.stride) * 30: continue
                    self.datas[k]['path'].append(self.data_dict[k]['path'][i])
                    self.datas[k]['start_time'].append(int(j/30))
                    self.datas[k]['label'].append(self.data_dict[k]['label'][i])
        
        ### """ 直接讀檔 """
        self.output_list = []
        print(range(len(self.datas['dashbord']['label'])))
        print(range(len(self.datas['right_window']['label'])))
        input_iter = tqdm(range(len(self.datas['dashbord']['label'])), desc="Loading video", leave=True)
        for idx in input_iter:

            k = 'dashbord'
            d_path = self.datas[k]['path'][idx]
            
            d_label = [0. for i in range(16)]
            d_label[int(self.datas[k]['label'][idx])] = 1.0
            d_label = torch.FloatTensor(d_label)

            d_start_time = self.datas[k]['start_time'][idx]

            vid = torchvision.io.VideoReader(d_path, "video")
            metadata = vid.get_metadata()
            
            # print(d_start_time)
            video_clips = []
            for frame in itertools.islice(vid.seek(d_start_time), self.clip_len):
                video_clips.append(frame['data'])
                current_pts = frame['pts']
                for i in range(self.stride):
                    next(vid)
            # print(current_pts)

            d_video = torch.stack(video_clips, 0)
            if self.video_transform:
                d_video = self.video_transform(d_video)

            # right window
            k = 'right_window'
            r_path = self.datas[k]['path'][idx]
            
            r_label = [0. for i in range(16)]
            r_label[int(self.datas[k]['label'][idx])] = 1.0
            r_label = torch.FloatTensor(r_label)

            r_start_time = self.datas[k]['start_time'][idx]

            vid = torchvision.io.VideoReader(r_path, "video")
            metadata = vid.get_metadata()
            
            video_clips = []
            for frame in itertools.islice(vid.seek(r_start_time), self.clip_len):
                video_clips.append(frame['data'])
                current_pts = frame['pts']
                next(vid)

            r_video = torch.stack(video_clips, 0)
            if self.video_transform:
                r_video = self.video_transform(r_video)
            
            output = {
                'dashbord': {
                    'path': d_path,
                    'video': d_video,
                    'target': d_label,
                    'start': d_start_time,
                },
                'right_window': {
                    'path': r_path,
                    'video': r_video,
                    'target': r_label,
                    'start': r_start_time,
                }
            }
            self.output_list.append(output)


    def __len__(self):
        return len(self.output_list)


    def __getitem__(self, idx):
        # 在init的時候讀影片
        output = self.output_list[idx]

        return output


class dataset(Dataset):
    def __init__(self, root, mode, mask, epoch_size=None, video_transform=None, clip_len=32):
        # super(RandomDataset).__init__()

        self.clip_len = clip_len
        self.video_transform = video_transform

        self.data_dict = {'path': [], 'start_time': [], 'end_time': [], 'label': []}

        datapath = os.path.join(root, mode+'.csv')
        with open(datapath, 'r') as file:
            csvreader = csv.reader(file) 
            for row in csvreader:
                if row[0] == 'path': continue
                for i in range(len(mask)):
                    if mask[i] in row[0]:
                        self.data_dict['path'].append(os.path.join(root, row[0][row[0].find('user'):row[0].find('_No')], row[0]))
                        self.data_dict['start_time'].append(int(row[1]))
                        self.data_dict['end_time'].append(int(row[2]))
                        self.data_dict['label'].append(int(row[3].split(' ')[1]))

        self.datas = {'path': [], 'start_time': [], 'label': []}
        for i in range(len(self.data_dict['path'])):
            for j in range(self.data_dict['start_time'][i] * 30, (self.data_dict['end_time'][i]-self.stride) * 30, self.clip_len * (self.stride+1)):
                self.datas['path'].append(self.data_dict['path'][i])
                self.datas['start_time'].append(int(j/30))
                self.datas['label'].append(self.data_dict['label'][i])
        
    
    def __len__(self):
        return len(self.datas['label'])


    def __getitem__(self, idx):
        path = self.datas['path'][idx]
        
        label = [0. for i in range(16)]
        label[int(self.datas['label'][idx])] = 1.0
        label = torch.FloatTensor(label)

        start_time = self.datas['start_time'][idx]

        vid = torchvision.io.VideoReader(path, "video")
        metadata = vid.get_metadata()
        
        video_clips = []
        print(start_time)
        for frame in itertools.islice(vid.seek(start_time), self.clip_len):
            video_clips.append(frame['data'])
            current_pts = frame['pts']
            next(vid)
        print(current_pts)

        video = torch.stack(video_clips, 0)
        if self.video_transform:
            video = self.video_transform(video)
        print(video.shape)
        # input()

        output = {
            'path': path,
            'video': video,
            'target': label,
            'start': start_time,
            'end': current_pts
        }
        
        # print(output['video'][0].shape)
        # print(output['target'].shape)
        # print(output['start'])
        # print(output['video'])

        return output

class dtypeChange:
    """Rotate by one of the given angles."""
    def __call__(self, x):
        return x.to(torch.float).div(255)

if __name__ == "__main__":
    from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo, ToTensorVideo, \
        NormalizeVideo
    from pytorchvideo.transforms import Permute, RandAugment, Normalize
    from opt import arg_parse

    args = arg_parse()

    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    imagenet_mean = [0.5, 0.5, 0.5]
    imagenet_std = [0.5, 0.5, 0.5]
    
    # org = THWC
    # totensor = CTHW

    # clear = TCHW
    # totensor = WTCH
    # 
    img_size = [144, 256]

    train_transform = T.Compose([
        # Permute(dims=[0, 2, 3, 1]),
        Permute(dims=[1, 0, 2, 3]),
        # dtypeChange(),
        # T.ToTensor()
        # ToTensorVideo(),  # T, C, H, W
        # Permute(dims=[1, 0, 2, 3]),
        # RandAugment(magnitude=10, num_layers=2),
        # Permute(dims=[1, 0, 2, 3]),
        T.Resize(size=img_size),
        # T.ToPILImage()
        # Normalize(mean=imagenet_mean, std=imagenet_std),
        # Permute(dims=[1, 0, 2, 3]),
    ])

    # d = Dataset('/mnt/Nami/2023_AI_City_challenge_datasets/Track_3/A1', video_transform=train_transform, clip_len = args.frames_per_clip)
    train_set = doubleDataset(
        root = args.dataset_root,
        mode = 'test',
        epoch_size = None,
        video_transform = train_transform,
        clip_len = args.frames_per_clip,
        stride=1
    )

    print(len(train_set))
    # invTrans = T.Compose([ Normalize(mean = [ 0., 0., 0. ],
    #                                                  std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    #                             Normalize(mean = [ -0.485, -0.456, -0.406 ],
    #                                                  std = [ 1., 1., 1. ]),
    #                            ])

    # print(train_set[0]['dashbord']['video'].shape)
    print(type(train_set[0]['dashbord']['video']))
    vid_f_1 = []
    vid_f_2 = []
    for i in tqdm(range(len(train_set))):
        if train_set[i]['dashbord']: pass
        if int(train_set[i]['dashbord']['target'].argmax(dim=0)) != int(train_set[i]['right_window']['target'].argmax(dim=0)):
            print(i, 'wrong!!!')
            print(train_set[i]['dashbord']['path'])
            print(train_set[i]['right_window']['path'])
            print(int(train_set[i]['dashbord']['target'].argmax(dim=0)))
            print(int(train_set[i]['right_window']['target'].argmax(dim=0)))
            input()
        vid_f_1.append(train_set[i]['dashbord']['video'])
        vid_f_2.append(train_set[i]['right_window']['video'])
    vid_1 = torch.concat((vid_f_1[0], vid_f_1[1]), dim=1)
    vid_2 = torch.concat((vid_f_2[0], vid_f_2[1]), dim=1)
    #print(vid)
    for i in range(2, len(vid_f_1)):
        vid_1 = torch.concat((vid_1, vid_f_1[i]), dim=1)
        vid_2 = torch.concat((vid_2, vid_f_2[i]), dim=1)
    # vid = invTrans(vid)
    # print(vid.shape)
    torchvision.io.write_video('./test1.mp4', vid_1.permute(dims=[1, 2, 3, 0]), fps=30./(train_set.stride+1))
    torchvision.io.write_video('./test2.mp4', vid_2.permute(dims=[1, 2, 3, 0]), fps=30./(train_set.stride+1))
    # root = '/mnt/Nami/2023_AI_City_challenge_datasets/Track_3/A1'
    # class_cont = [0 for i in range(16)]

    # for root, _, files in os.walk(root):
    #     for file in files:
    #         if file.endswith('.csv'):
    #             # print(file)
    #             with open(os.path.join(root, file), 'r') as label_csv:
    #                 labels = label_csv.readlines()
    #                 for label in labels:
    #                     if label[0] == "F": continue
    #                     # print(label)
    #                     label = label.split(',')
    #                     class_cont[int(label[5].split(' ')[1])] += _times2sec(label[4]) - _times2sec(label[3])
    
    # import matplotlib.pyplot as plt

    # class_ = [str(i) for i in range(16)]
    # plt.bar(class_, class_cont)
    # for i in range(16):
    #     plt.text(class_[i], class_cont[i]+0.005, '{}'.format(class_cont[i]), fontsize=11, horizontalalignment='center', color='black')
    # plt.title("class analyze")
    # plt.xlabel("class id")
    # plt.ylabel('count')
    # plt.savefig("./class_anz_sec.jpg")

    