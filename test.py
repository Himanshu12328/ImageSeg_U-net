import torch
from torch.utils.data import DataLoader
from data.dataset import SegmentationDataset
from models.unet import UNet
from utils.visualization import visualize
from utils.metrics import dice_coeff, iou_score, pixel_accuracy
from torchvision import transforms

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
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_epoch10.pth"))
    model.eval()

    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device).float()
            preds = model(images)
            preds = torch.sigmoid(preds)

            dice = dice_coeff(preds, masks).item()
            iou = iou_score(preds, masks).item()
            acc = pixel_accuracy(preds, masks).item()
            # Print metrics
            print(f"Sample {idx+1}: Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {acc:.4f}")
            # Visualize the first image, mask, and prediction
            visualize(images[0].cpu(), masks[0].cpu(), preds[0].cpu())

if __name__ == "__main__":
    main()
