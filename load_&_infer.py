from matplotlib import pyplot as plt
from Architecture.Generator import Generator
import config
import torch
from PIL import Image



# Messy Code, here we go,

if __name__ == "__main__":
    horse2zebra = Generator(img_channels=3).to(config.DEVICE)
    checkpoint_h2z = torch.load("pretrained_models/genh.pth.tar", map_location=config.DEVICE)
    horse2zebra.load_state_dict(checkpoint_h2z["state_dict"])
    # zebra2horse = Generator(img_channels=3).to(config.DEVICE)
    # checkpoint_z2h = torch.load("pretrained_models/genz.pth.tar", map_location=config.DEVICE)
    # zebra2horse.load_state_dict(checkpoint_z2h)
    img = Image.open('pretrained_models/1.jpg').convert('RGB')
    img_transformed = config.transforms(img).to(config.DEVICE)
    output = horse2zebra(img_transformed)
    plt.imshow(output.cpu().detach().numpy().transpose(1, 2, 0))
    plt.show()
