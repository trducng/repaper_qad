import torchvision.datasets as datasets
import torchvision.transforms as T
train_size = [32, 32]
validation_size = [32, 32]

dataset_root = "/home/john/repaper_qad/vitca.pytorch/reproduce/downloads/datasets"
train_dataset = datasets.CelebA(
    root=dataset_root,
    split="train",
    transform=T.Compose([T.ToTensor(), T.Resize(train_size, antialias=True)]),
)
validation_dataset = datasets.CelebA(
    root=dataset_root,
    split="valid",
    transform=T.Compose(
        [T.ToTensor(), T.Resize(validation_size, antialias=True)]
    ),
)

