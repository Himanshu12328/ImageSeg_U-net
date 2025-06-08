import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import SegmentationDataset
from models.unet import UNet
from utils.metrics import dice_coeff, iou_score, pixel_accuracy
from torchvision import transforms
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_img = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset("dataset/images", "dataset/masks", 
                                  transform_img=transform_img, transform_mask=transform_mask)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/10")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device).float()
            preds = model(images)
            loss = criterion(preds, masks)
            dice = dice_coeff(preds, masks).item()
            iou = iou_score(preds, masks).item()
            acc = pixel_accuracy(preds, masks).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item(), dice=dice, iou=iou, acc=acc)

        torch.save(model.state_dict(), f"unet_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()
