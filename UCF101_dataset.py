from typing import Optional, Callable, Tuple

from torch import Tensor
from torchvision.datasets import UCF101


class MyUCF101(UCF101):
    def __init__(self, transform: Optional[Callable] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label


if __name__ == "__main__":
    from torchvision.transforms import transforms as T
    from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo, ToTensorVideo, \
        NormalizeVideo
    from torch.utils.data import DataLoader, RandomSampler

    dataset_root = '/home/Bill0041/dataset/UCF101/UCF-101'
    annotation_path = '/home/Bill0041/dataset/UCF101/ucfTrainTestlist'
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        ToTensorVideo(),
        RandomHorizontalFlipVideo(),
        RandomResizedCropVideo(size=(224, 224)),
        NormalizeVideo(mean=imagenet_mean, std=imagenet_std, inplace=True)
    ])

    train_set = MyUCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        # _precomputed_metadata=train_precomputed_metadata,
        frames_per_clip=32,
        train=True,
        output_format='THWC',
        transform=train_transform,
    )

    print(len(train_set))
    print(len(train_set[0])) 
    # tuple(data, class)
    # data.shape = [3, 32, 224, 224]
    print(train_set[0][0].shape)

    train_sampler = RandomSampler(train_set, num_samples=len(train_set) // 10)
    train_dataloader = DataLoader(
        train_set,
        batch_size=2,
        num_workers=1,
        shuffle=False,
        drop_last=True,
        sampler=train_sampler,
    )

    sample = next(iter(train_dataloader))
    print(len(sample))
    print(sample[0].shape)
    print(sample[1])