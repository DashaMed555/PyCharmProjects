import os
import datetime

import cv2
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet34


imgsz = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# class DownSample(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.swish = nn.SELU()
#         self.maxpool = nn.MaxPool2d(2, 2)
#
#         self.conv1 = nn.Conv2d(in_channels=channels, out_channels=(2 * channels), kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(2 * channels)
#         self.conv2 = nn.Conv2d(in_channels=(2 * channels), out_channels=(2 * channels), kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(2 * channels)
#
#     def forward(self, x):
#         x = self.bn1(self.swish(self.conv1(x)))
#         x_skip = self.bn2(self.swish(self.conv2(x)))
#         x = self.maxpool(x)
#         return x, x_skip


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.swish = nn.SELU()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=2,
                                                 stride=2)

        self.conv1 = nn.Conv2d(in_channels=(2 * channels), out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=(channels // 2), kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels // 2)

    def forward(self, x, x_skip):
        x = self.conv_transpose(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.bn1(self.swish(self.conv1(x)))
        x = self.bn2(self.swish(self.conv2(x)))
        return x


class BottleNeck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.swish = nn.SELU()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.bn1(self.swish(self.conv1(x)))
        x = self.bn2(self.swish(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = nn.SELU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        children = list(resnet34(pretrained=True).children())
        # self.preparing = nn.Sequential(*children[:4])

        self.layer1 = children[4]
        self.layer2 = children[5]
        self.layer3 = children[6]
        self.layer4 = children[7]

        # self.down1 = DownSample(32)
        # self.down2 = DownSample(64)
        # self.down3 = DownSample(128)
        # self.down4 = DownSample(256)

        self.bottleneck = BottleNeck(512)

        self.up4 = UpSample(512)
        self.up3 = UpSample(256)
        self.up2 = UpSample(128)
        self.up1 = UpSample(64)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.preparing(x)

        x = self.bn1(self.swish(self.conv1(x)))

        x_skip1 = self.layer1(x)
        x_skip2 = self.layer2(x_skip1)
        x_skip3 = self.layer3(x_skip2)
        x_skip4 = self.layer4(x_skip3)

        x = self.bottleneck(x_skip4)

        x = self.up4(x, x_skip4)
        x = self.up3(x, x_skip3)
        x = self.up2(x, x_skip2)
        x = self.up1(x, x_skip1)

        x = self.sigmoid(self.conv2(x))

        return x


class RoadsDataset(Dataset):
    def __init__(self, images_path, masks_path, image_transform=None, mask_transform=None, transform=None):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.transform = transform
        self.filenames = list(filter(lambda filename: filename.endswith('tiff'), os.listdir(images_path)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        image = cv2.imread(f'{self.images_path}/{filename}')
        mask = cv2.imread(f'{self.masks_path}/{filename[:-1]}', cv2.IMREAD_GRAYSCALE)[..., None]
        mask[(image > 250).all(axis=2)] = 0

        if self.transform:
            image, mask = self.transform(image=image, mask=mask).values()

        if self.image_transform is not None:
            image = self.image_transform(image=image)['image']

        if self.mask_transform is not None:
            mask = self.mask_transform(image=mask)['image']
            mask = torch.where(mask > 250, 1, 0).type(torch.float32)

        return image.to(device), mask.to(device)


def compute_IoU(mask, pred_mask):
    intersection = torch.logical_and(mask, pred_mask).sum()
    union = torch.logical_or(mask, pred_mask).sum()
    return intersection / union


def train_loop(model, optimizer, dataloader, criterion):
    print('--------------------------- TRAINING START ---------------------------')
    model.train()

    images_num = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_loss = 0
    running_IoU = 0

    start = datetime.datetime.now()
    for i, (images, masks) in enumerate(dataloader):
        pred_masks = model(images)
        loss = criterion(pred_masks, masks)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        IoU = compute_IoU(masks, torch.where(pred_masks > 0.5, 1, 0))
        running_IoU += IoU

        loss, current = loss.item(), i * dataloader.batch_size + len(images)
        finish = datetime.datetime.now()
        difference = finish - start
        print(f'loss: {loss:>7f} IoU: {(IoU * 100):.1f}% [{current:>5d}/{images_num:>5d} images] [{i + 1}/{num_batches} batches] {difference}')

    print('--------------------------- TRAINING FINISH ---------------------------\n')
    return running_loss / num_batches, running_IoU / num_batches


def val_loop(model, dataloader, criterion, mode='VALIDATION'):
    print(f'--------------------------- {mode} START ---------------------------')
    model.eval()

    images_num = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_loss = 0
    running_IoU = 0

    start = datetime.datetime.now()
    for i, (images, masks) in enumerate(dataloader):
        pred_masks = model(images)
        loss = criterion(masks, pred_masks)

        running_loss += loss.item()
        IoU = compute_IoU(masks, pred_masks)
        running_IoU += IoU

        loss, current = loss.item(), i * dataloader.batch_size + len(images)
        finish = datetime.datetime.now()
        difference = finish - start
        print(f'loss: {loss:>7f} IoU: {(IoU * 100):.1f}% [{current:>5d}/{images_num:>5d} images] [{i + 1}/{num_batches} batches] {difference}')

    print(f'--------------------------- {mode} FINISH ---------------------------\n')
    return running_loss / num_batches, running_IoU / num_batches


def fit(model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, criterion, epochs):
    writer = SummaryWriter()
    _, best_IoU = val_loop(model, val_dataloader, criterion)
    for epoch in range(epochs):
        print(f'------------------------------------- EPOCH {epoch + 1}-------------------------------------')

        lr = optimizer.param_groups[0]["lr"]
        loss, IoU = train_loop(model, optimizer, train_dataloader, criterion)
        print(f'\n~~~ train_loss = {loss}\ttrain_IoU = {IoU} ~~~\n')
        writer.add_scalar('Learning rate', lr, epoch + 1)
        writer.add_scalar('Training loss', loss, epoch + 1)
        writer.add_scalar('Training IoU', IoU, epoch + 1)

        loss, IoU = val_loop(model, val_dataloader, criterion)
        print(f'\n~~~ val_loss = {loss}\ttval_IoU = {IoU} ~~~\n')
        writer.add_scalar('Validation loss', loss, epoch + 1)
        writer.add_scalar('Validation IoU', IoU, epoch + 1)

        scheduler.step(IoU)

        if IoU > best_IoU:
            best_IoU = IoU
            torch.save(model, 'best_model.pt')

        writer.flush()
    writer.close()

    loss, IoU = val_loop(model, test_dataloader, criterion, 'TESTING')
    print(f'\n~~~ test_loss = {loss}\ttest_IoU = {IoU} ~~~\n')


def make_transformations(p=0.5, smallest_p=0.01):
    affine = A.Compose([A.Flip(p=p),
                        A.Rotate(p=p),
                        A.ShiftScaleRotate(p=p, rotate_limit=(-90, 90))])

    blur = A.SomeOf([A.AdvancedBlur(p=p),
                     A.Blur(p=p),
                     A.GaussianBlur(p=p),
                     A.MedianBlur(p=p)], 1)

    noise = A.SomeOf([A.GaussNoise(p=p),
                      A.ISONoise(p=p),
                      A.MultiplicativeNoise(p=p)], 1)

    sharpen = A.SomeOf([A.Emboss(p=p),
                        A.Sharpen(p=p)], 1)

    colors = A.SomeOf([A.CLAHE(p=p),
                       A.ColorJitter(p=p, hue=0.08),
                       A.FancyPCA(p=p),
                       A.RandomBrightnessContrast(p=p),
                       A.RandomGamma(p=p),
                       A.RandomToneCurve(p=p),
                       A.ToGray(p=smallest_p),
                       A.ToSepia(p=smallest_p)], 1)
    augmentations = [affine, blur, noise, sharpen, colors]
    return augmentations


def main():
    image_transform = A.Compose([A.Resize(imgsz, imgsz),
                                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ToTensorV2()])

    mask_transform = A.Compose([A.Resize(imgsz, imgsz),
                                ToTensorV2()])

    augment_transform = A.Compose(make_transformations())

    dataset_path = 'datasets/MassachusettsRoads/tiff'

    train_dataset = RoadsDataset(f'{dataset_path}/images/train',
                                 f'{dataset_path}/binary_masks/train',
                                 image_transform, mask_transform, augment_transform)

    val_dataset = RoadsDataset(f'{dataset_path}/images/val',
                               f'{dataset_path}/binary_masks/val',
                               image_transform, mask_transform)

    test_dataset = RoadsDataset(f'{dataset_path}/images/test',
                                f'{dataset_path}/binary_masks/test',
                                image_transform, mask_transform)

    # for image, mask in iter(train_dataset):
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    #     plt.axis('off')
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    #     plt.axis('off')
    #
    #     plt.show()

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    criterion = torch.nn.BCELoss()
    epochs = 5
    fit(model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, criterion, epochs)


if __name__ == '__main__':
    main()
