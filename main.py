import os
import time
import pickle
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
# from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo, ToTensorVideo, \
#     NormalizeVideo
from pytorchvideo.transforms import Permute, RandAugment, Normalize

from opt import arg_parse
from dataset import doubleDataset, dataset
from model.TubeViT import TubeViT
from model.ViViT.ViViT_model2 import ViViT
from model.ViViT.ViViT_model3 import ViViTBackbone
from model.ViViT.ViViT import ViViTBackbone as two_stage_ViViT
from trainer import train, validate
# from TubeViT.video_transforms import ResizedVideo

args = arg_parse()
train_id = "debug" if args.debug else int(time.time()) % 100000
# os.environ['CUDA_VISIBLE_DEVICES']=str(args.device)

torch.cuda.set_device(args.device)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def plot_img(epoch, y1, y2, act):
    epochs = [str(i) for i in range(1, epoch+2)]
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, y1, label="training {}".format(act))
    plt.plot(epochs, y2, label="testing {}".format(act))
    # plt.text(best_loss[0], best_loss[1]+0.005, 'epoch {}: {:.6f}'.format(best_loss[0]+1, best_loss[1]), fontsize=11, horizontalalignment='right', color='black')
    # plt.plot(best_loss[0], best_loss[1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")

    plt.xticks(rotation=60)
    axis = plt.gca()
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=20))
    plt.title("{} {} History".format(train_id, act))
    plt.xlabel("epochs")
    plt.ylabel(act)
    plt.legend()
    plt.savefig("./figure/{}/{}_{}_history.jpg".format(train_id, train_id, act))
    plt.close()


class dtypeChange:
    """Rotate by one of the given angles."""
    def __call__(self, x):
        return x.to(torch.float).div(255)


def main():
    try:
        os.mkdir("./figure/{}".format(train_id))
        os.mkdir("./weight/{}".format(train_id))
    except FileExistsError:
        if train_id == "debug":
            pass

    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    imagenet_mean = [0.5, 0.5, 0.5]
    imagenet_std = [0.5, 0.5, 0.5]

    img_size = [144, 256] # 25 times 9:16
    # training transform
    train_transform = T.Compose([
        # Permute(dims=[0, 2, 3, 1]),
        # T.ToTensor()
        # ToTensorVideo(),  # T, C, H, W
        dtypeChange(),
        Permute(dims=[1, 0, 2, 3]),
        # RandAugment(magnitude=10, num_layers=2),
        # Permute(dims=[1, 0, 2, 3]),
        T.Resize(size=img_size),
        # Normalize(mean=imagenet_mean, std=imagenet_std),
        # Permute(dims=[1, 0, 2, 3]),
    ])

    # validate transform
    val_transform = T.Compose([
        dtypeChange(),
        Permute(dims=[1, 0, 2, 3]),
        # Permute(dims=[0, 2, 3, 1]),
        # ToTensorVideo(),  # T, C, H, W
        T.Resize(size=img_size),
        # Normalize(mean=imagenet_mean, std=imagenet_std),
        # Permute(dims=[1, 0, 2, 3]),
    ])

    # dataset
    train_set = doubleDataset(
        root = args.dataset_root,
        mode = 'train',
        epoch_size = None,
        video_transform = train_transform,
        clip_len = args.frames_per_clip,
        stride=1
    )

    val_set = doubleDataset(
        root = args.dataset_root,
        mode = 'test',
        epoch_size = None,
        video_transform = val_transform,
        clip_len = args.frames_per_clip,
        stride=1
    )
    # indices = torch.arange(len(train_set))
    # train_set = torch.utils.data.Subset(train_set, indices)

    # dataloader
    train_dataloader = DataLoader(
        train_set,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = False,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # model init
    '''
    
    model = TubeViT(
        num_classes=args.num_classes,
        video_shape=item['video'].shape[1:],
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
    )
    '''
    # patch default: 8, 4, 4
    # model = two_stage_ViViT(
    #     t=args.frames_per_clip,
    #     h=img_size[0],
    #     w=img_size[1],
    #     patch_t=8,
    #     patch_h=4,
    #     patch_w=4,
    #     num_classes=16,
    #     dim=512,
    #     depth=6,
    #     heads=10,
    #     mlp_dim=8,
    #     model=3
    # )
    model = torch.load('./weight/debug/Dash_checkpoint_41.pt')
    '''model = ViViT(
        image_size = 224,
        patch_size = 16,
        num_classes = 16,
        num_frames = args.frames_per_clip, 
    )'''
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

    # optimizer and loss function
    if args.optimizer == 'Adam':
        opti = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        opti = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # opti = torch.nn.DataParallel(opti, device_ids=[0, 1])
    loss_func = nn.CrossEntropyLoss()

    # train and validate
    train_loss_his, val_loss_his = [], []
    train_acc_his, val_acc_his = [], []
    for epoch in range(41, args.epochs):
        train_loss, train_acc = train(args, epoch+1, model, opti, loss_func, train_dataloader)
        train_loss_his.append(train_loss)
        train_acc_his.append(train_acc)

        val_loss, val_acc = validate(args, epoch+1, model, loss_func, val_dataloader)
        val_loss_his.append(val_loss)
        val_acc_his.append(val_acc)
        
        plot_img(epoch, train_loss_his, val_loss_his, 'loss')
        plot_img(epoch, train_acc_his, val_acc_his, 'acc')

        if epoch % 10 == 0:
            torch.save(model, './weight/{}/Dash_checkpoint_{}.pt'.format(train_id, epoch+1))


if __name__ == '__main__':
    main()
    
    # y = torch.tensor([[0., 1., 2., 1., 0.], [0., 1., 1., 2., 0.]])
    # m = nn.Softmax(dim=1)
    # print(m(y))