import random
from matplotlib import image
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset, Dataset
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image 


#No longer in use, this loader would preload all the data before training but it would 
#saturate the RAM faster than you can possibly imagine since all samples occupied around 200Gb
#when in a decompressed numpy array
# def load_data(sample_dir,mask_dir, pool_size =100000):

#     #dataloading
#     samples = []
#     masks = []
#     _filenames = os.listdir(sample_dir)
#     if len(_filenames) > pool_size:
#         _filenames = random.sample(_filenames,pool_size)

#     for _filename in _filenames:
#         _filename = os.path.splitext(_filename)[0]
#         _mask = image.imread(f'{mask_dir}/{_filename}.jpeg').astype("float32")
#         _img = image.imread(f'{sample_dir}/{_filename}.jpeg').astype("float32")
#         samples.append(_img/255)
#         masks.append(_mask/255)
#     samples = torch.tensor(np.array(samples,dtype='float32'))
#     masks = torch.tensor(np.array(masks,dtype='float32')).swapaxes(-1,-3)
    
#     return(TensorDataset(samples,masks))

"""
This is the dataloader used for lazy loading. This means that the data is not pre loaded but streamed to the GPU when required.
with the original dataloader system the RAM would fill up faster than you could possibly imagine.
"""
class Transformer_dataset(Dataset):
    def __init__(self,path,num_samples= 100000000):
        self.path = path
        self.filenames = os.listdir(f'{self.path}/sample')

        #this is convenient if you want to train on a subset of the data for some reason
        if num_samples < len(self.filenames):
            self.filenames = random.sample(self.filenames,num_samples)

        #Normalizing transformation applied to the data, it helps with model performance only the sample is normalized since
        #the mask can easily be mapped to a range of 0 to 1 by dividing by 255 
        self.transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5,.5,.5], std =[.5,.5,.5])])

    def __len__(self):
        return(len(self.filenames))
    
    #__getitem__ is a default pytorch dataset function that is overwritten in this case. It is called by the dataloader
    def __getitem__(self, index):

        filename = os.path.splitext(self.filenames[index])[0]
        
        #the two files are opened using different functions because the read_image function was causing some problems with the
        #data normalization, this is not necesarry for the mask so i guess i could have changed that to the Image method aswell
        #but i didn't see a reason for doing this.
        mask = read_image(f'{self.path}/mask/{filename}.jpeg')/255
        sample = Image.open(f'{self.path}/sample/{filename}.jpeg')
        
        #applying the normalization to the sample        
        sample = self.transform_norm(sample)

        #swapping the axis to be compatible with the model
        return(sample.permute(1,2,0),mask)
    