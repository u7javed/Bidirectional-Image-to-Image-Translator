import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

from models import *

def main():
    #take in arguments
    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')

    # parameters needed to enhance image
    parser.add_argument('--image_file', type=str, default='test_image.png', help='location of image file')
    parser.add_argument('--file_width', type=int, default=256, help='width of the processed file image (NOTE: Should match what generator was trained on)')
    parser.add_argument('--file_height', type=int, default=256, help='height of the processed file_image (NOTE: Should match what generator was trained on)')
    parser.add_argument('--channels', type=int, default=3, help='Number of feature channels for input image (NOTE: Should match what generator was trained on)')
    parser.add_argument('--dir_to_generator', type=str, default='seg2image_generator.pt', help='directory to generator (choose depending on which direction you wish to translate.)')
    parser.add_argument('--save_directory', type=str, default='', help='directory where translated image will be saved')

    args = parser.parse_args()

    image_file = args.image_file
    file_width = args.file_width
    file_height = args.file_height
    channels = args.channels
    dir_to_generator = args.dir_to_generator
    save_directory = args.save_directory


    #transformation applied to image
    transformation = transforms.Compose([
        transforms.Resize((file_width, file_height), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])


    pil_image = Image.open(image_file)

    #conver to tensor and conver to batch of size 1
    image_tensor = transformation(pil_image)
    image_tensor = image_tensor.view(1, *(image_tensor.size()))

    #load enhancer (generator)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Generator(channels, file_width, file_height).to(device)
    model.load_state_dict(torch.load(dir_to_generator))

    translated_image = model(image_tensor.to(device))
    image_grid = torchvision.utils.make_grid(translated_image.cpu().detach(), nrow=1, normalize=True)
    _, plot = plt.subplots(figsize=(8, 8))
    plt.axis('off')
    plot.imshow(image_grid.permute(1, 2, 0))
    plt.savefig(save_directory + '/translated_image.png', bbox_inches='tight')

if __name__ == "__main__":
    main()