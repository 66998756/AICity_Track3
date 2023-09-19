from tqdm import tqdm

import torch
from torch import nn, Tensor, optim


# class Trainer:
#     def __init__(
#         self,
#         model,
#         dataloader,
#         lr: float = 3e-4,
#         weight_decay: float = 0,
#         weight_path: str = None,
#         max_epochs: int = None,
#         label_smoothing: float = 0.0,
#         dropout: float = 0.0,
#         attention_dropout: float = 0.0,
#         **kwargs
#     ):


def train(args, epoch, model, opt, loss_func, dataloader, weight_path: str = None):
    model.train()
    train_loss, train_acc = [], []

    # in epoch
    train_iter = tqdm(dataloader, desc="Epoch: {}/{} ({}%), loss: NaN, acc: 0/{} |Training.|".format(
        epoch, args.epochs, int(epoch/args.epochs), args.batch_size), leave=False)
    for sample in train_iter:
        data_1, target_1 = sample['dashbord']['video'], sample['dashbord']['target']
        data_2, target_2 = sample['right_window']['video'], sample['right_window']['target']
        data_1, target_1 = data_1.cuda(), target_1.cuda()
        data_2, target_2 = data_2.cuda(), target_2.cuda()
        # print(target_1 == target_2)
        # print(data_1.shape)
        # print(data_2.shape)

        opt.zero_grad()
        output = model(data_1, data_2)
        loss = loss_func(output, target_1)
        loss.backward()
        opt.step()

        train_loss.append(loss.cpu().item())
        round_acc = int((output.argmax(dim=1) == target_1.argmax(dim=1)).float().sum().item())
        train_acc.append(round_acc)

        train_iter.set_description("Epoch: {}/{} ({}%), loss: {:.6f}, acc: {}/{} |Training|".format(
            epoch, args.epochs, int(epoch/args.epochs), round(loss.item(), 6), round_acc, args.batch_size))
    
    return round(sum(train_loss) / len(train_loss), 6), round(sum(train_acc) / len(train_acc), 6)


def validate(args, epoch, model, loss_func, dataloader, weight_path: str = None):
    model.eval()
    val_loss, val_acc = [], []

    with torch.no_grad():

        val_iter = tqdm(dataloader, desc="Epoch: {}/{} ({}%), loss: NaN, acc: 0/{} |validate|".format(
        epoch, args.epochs, int(epoch/args.epochs), args.batch_size), leave=False)
        for sample in val_iter:
            data_1, target_1 = sample['dashbord']['video'], sample['dashbord']['target']
            data_2, target_2 = sample['right_window']['video'], sample['right_window']['target']
            data_1, target_1 = data_1.cuda(), target_1.cuda()
            data_2, target_2 = data_2.cuda(), target_2.cuda()
            
            output = model(data_1, data_2)
            loss = loss_func(output, target_1)

            val_loss.append(loss.cpu().item())
            round_acc = int((output.argmax(dim=1) == target_1.argmax(dim=1)).float().sum().item())
            val_acc.append(round_acc)

            val_iter.set_description("Epoch: {}/{} ({}%), loss: {:.6f}, acc: {}/{} |validate|".format(
                epoch, args.epochs, int(epoch/args.epochs), round(loss.item(), 6), round_acc, args.batch_size))
    return round(sum(val_loss) / len(val_loss), 6), round(sum(val_acc) / len(val_acc), 6)


def test(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    import torch
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', '--learning_rate', type=int, default=1e-4, help='learning rate.')
    opt = parser.parse_args()

    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)

    opt = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=0.001)
    print(test(opt))
    