import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root', '-r', default='/mnt/Nami/2023_AI_City_challenge_datasets/Track_3/A1', help='path to dataset.')
    parser.add_argument('--num_classes', '-nc', type=int, default=16, help='num of classes of dataset.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size.')
    parser.add_argument('--frames_per_clip', '-f', type=int, default=32, help='frame per clip.')
    parser.add_argument('--video_size', '-v', default=[224, 224], help='video size.')
    parser.add_argument('--device', '-d', type=int, default=0, help='debug option.')
    parser.add_argument('--debug', action="store_true", help='debug option.')

    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='total epochs.')
    parser.add_argument('-opt', '--optimizer', default="SGD", choices=["SGD", "Adam"], help='optimizer.')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    print(int(1e-4))