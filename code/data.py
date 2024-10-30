import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import v2

def load(path_image, path_mask):
    image, mask = np.load(path_image), np.load(path_mask)
    image, mask = torch.tensor(image), torch.tensor(mask)
    return image, mask

def resize(input_image, input_mask,h,w):
    transform = v2.Resize(size=(h, w))
    input_image = transform(input_image)
    input_mask = transform(input_mask)
    return input_image, input_mask

def random_crop(input_image, input_mask,h,w):
    stacked_image = torch.stack((input_image, input_mask), dim=0)
    t = v2.RandomCrop(size=[h, w])
    cropped_image = t(stacked_image)
    return cropped_image[0], cropped_image[1]

def random_jitter(input_image, input_mask, up, down):
    # Resizing to 286x286
    input_image, input_mask = resize(input_image, input_mask, up, up) 
    # Random cropping back to 256x256
    input_image, input_mask = random_crop(input_image, input_mask, down, down)

    return input_image, input_mask

def horizontal_flip(input_image, input_mask):
    if np.random.random()>0.5:
        input_image = transforms.functional.hflip(input_image)
        input_mask = transforms.functional.hflip(input_mask)
    return input_image, input_mask

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_sat, file_mask):
        self.file_sat =file_sat
        self.file_mask=file_mask
        
    def __len__(self):
        return len(self.file_sat)

    def __getitem__(self, ix):
        image_sat,image_mask = load(self.file_sat[ix], self.file_mask[ix])
        image_sat,image_mask = horizontal_flip(image_sat,image_mask)
        return (image_sat/255).float(),(image_mask[1:2]).float()