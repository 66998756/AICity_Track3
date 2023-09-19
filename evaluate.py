import os
import csv
import argparse

import torch
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms as T
from torchvision.datasets.folder import make_dataset


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', '-d', help='path to data.')
parser.add_argument('--weight_path', '-w', help='path to weight.')

args = parser.parse_args()


class evaluatedDataset(Dataset):
    def __init__(self, root, mode, epoch_size=None, video_transform=None, clip_len=32):
        # super(RandomDataset).__init__()
        self.clip_len = clip_len
        self.video_transform = video_transform
        print('------')

        video_path = []
        datapath = os.path.join(root, mode+'.csv')
        print(datapath)
        with open(datapath, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                path = os.path.join(root, row[0][row[0].find('user'):row[0].find('_No')], row[0])
                if row[0] == 'path': continue
                if path not in video_path: video_path.append(path)

        self.video = {'video': [], 'times': []}
        for path in video_path:
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()

            self.video['video'].append(vid)
            self.video['times'].append(int(metadata["video"]["duration"][0]))

    
    def __getitem__(self, idx):
        pass

    
    def __len__(self):
        return sum(self.video['times'])


def main():
    # model = torch.load(args.weight_path)

    # model.cuda()
    # model.eval()
    root = '/mnt/Nami/2023_AI_City_challenge_datasets/Track_3/A1'
    dataset = evaluatedDataset(root, 'test')

    datapath = os.path.join(root, 'test.csv')
    with open(datapath, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            path = os.path.join(root, row[0][row[0].find('user'):row[0].find('_No')], row[0])
            if row[0] == 'path': continue
            if path not in video_path: video_path.append(path)

    self.video = {'video': [], 'times': []}
    for path in video_path:
        vid = torchvision.io.VideoReader(path, "video")
        metadata = vid.get_metadata()

        self.video['video'].append(vid)
        self.video['times'].append(int(metadata["video"]["duration"][0]))

    for frame in itertools.islice(vid.seek(0), self.clip_len):
            video_frames.append(frame['data'])
            current_pts = frame['pts']


if __name__ == "__main__":
    main()
    