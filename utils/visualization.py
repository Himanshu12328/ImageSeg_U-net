import matplotlib.pyplot as plt

def visualize(image, mask, pred=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.axis("off")
    if pred is not None:
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred.squeeze(), cmap='gray')
        plt.axis("off")
    plt.show()