import os
import torch
import torch.nn as nn
from nn_modules import Conv_ATM,AutoEncoder,Encoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import Transformer_dataset

#this is necesarry for multiple num_workers > 1 otherwise the multiprocessing will cause an endless loop
if __name__ == '__main__':
    
    """These are the model parameters that i ended up using, 
    you could theoretically change these but be cautious that some of
    the settings might increase the power requirements quite significantly"""
    
    #image input size
    crop_size = 128
    #the number blocks for each layer of the convolutional encoder
    depths=[3, 3, 3, 9, 3]
    #the number of dims for each layer of the convolutional encoder
    dims=[3, 6, 12, 24, 48]
    #resolution of the patch
    patch_size=16
    #Convolutional encoder kernel size
    dConv_kernel_size = 5
    #Name used for model saving
    name = 'Transformer classifier'
    #Device either 'cuda' for GPU or 'cpu' for eternal waiting
    device = 'cuda'

    """
    tuneable hyperparameters
    The influence of the mask in calculating the loss
    """
    beta = 3

    #Learning rate
    lr = 5e-5

    """
    Weights controls the loss fraction of the different channels, collapsed, non collapsed and background.
    It compensates for an unbalance in the dataset, 
    there is more background than anything else so i'm being less harsh with missing the background.
    """
    weight =torch.tensor([.8,1,.3]).to(device)

    #Weight decay is a regularization technique it helps with overfitting the influence was quite subtle in the results.
    weight_decay=0.01

    #Create an instance of the convolutional encoder#
    encoder = Encoder(depths=depths,dims = dims, dConv_kernel_size = dConv_kernel_size)

    """
    Alternatively you can use an encoder pretrained with the autoencoder.
    I found this doesn't really have a significant effect on training times. 
    ckpt is the path to the checkpoint
    """
    #autoencoder = AutoEncoder(depths,dims,dConv_kernel_size)
    #autoencoder.load_state_dict(torch.load(ckpt))
    #encoder = autoencoder.encoder
    
    #If you want to run this on a singular GPU you have to set device_ids to [0]
    model = nn.DataParallel(Conv_ATM(encoder=encoder,dim=dims[-1]),device_ids=[0,1]).to(device)

    #Training
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    logger = SummaryWriter(os.path.join("runs", name))

    #You can set these individually since it is actually viable to use L1 or MSE for the labels, i tried but prefer this.
    class_loss = nn.CrossEntropyLoss(weight=weight)
    label_loss = nn.CrossEntropyLoss(weight=weight)

    """
    Create the dataloader, Pin memory can be turned of if num_workers = 1 otherwise it will crash, 
    for debugging it can be usefull to  set num_workers = 1 since it reduces startup times at the cost of speed
    """
    dataloader = DataLoader(Transformer_dataset('./Transformer_Training'), batch_size = 256, 
                                                shuffle=True, pin_memory= True, num_workers=8)

    """
    Adding plus one because otherwise epoch%5 would not save at the end because it starts at zero. 
    I originally had epoch 0 as a warmup (very low learning rate), for increased stability but there was not really an effect.
    """
    for epoch in range(10+1):
        
        pbar = tqdm(dataloader)
        l = len(pbar)
        for i,(samples,masks) in enumerate(pbar):
            
            num_patches = masks.shape[-1]//patch_size
            samples = samples.to(device)
            
            #Used for loss
            masks = masks.to(device)
            class_distribution = masks.mean(axis=(-1,-2))
            
            """
            Making a background of ones which the activations have to outperform in order to get marked in the softmax
            this is a significant deviation from the original paper. essentially i replaced the sigmoid in the ATM with a 
            Softmax in the semantic segmentation to implement the CE_Loss which i found to work better. Essentially i'm 
            assuming that if there is no class sample the attention will be low and will therefore be drowned out by the background
            """
            background = torch.ones((samples.shape[0],1,crop_size,crop_size)).to(device)

            #Model inference on sample returning the final q, attentions and class predictions
            q, attns, class_prediction = model(samples.to(device))

            #reshaping the attentions back into a two dimensional "image like" structure instead of a one dimensional embedding.
            fold = nn.Fold((num_patches,num_patches),kernel_size=1)
            
            out_masks = []
            """
            Upscaling the attentions using bilinear interpolation and adding them together. I dont do this inplace in order
            to access the individual layers for debugging but it is otherwise inplace addition would be more efficient i think.
            """
            for i, attn in enumerate(attns):
                if i == 0:
                    out_masks.append(nn.functional.interpolate(fold(attn),size=(128,128), mode = 'bilinear',align_corners=False))
                else:
                    out_masks.append(out_masks[i-1]+
                                    nn.functional.interpolate(fold(attn),size=(128,128), mode = 'bilinear',align_corners=False)) 
            
            #get the final out mask
            out_masks = out_masks[-1]
            #Add the background to the output masks, again this is a CE_loss compatibility hack
            mask_with_background = torch.concat([out_masks,background],dim=1)

            #Einstein summation in order to compute the final semantic segmentation
            semseg = torch.einsum('bcq,bqhw->bqhw', class_prediction,mask_with_background).softmax(dim=1)
            


            cl = class_loss(class_prediction.mean(dim=1),class_distribution)
            ll = label_loss(semseg,masks)

            """"
            This is a necesarry tweak since the gradients collapsed sometimes if the weight of classes became zero
            leading to it to have gradients of 0. This happened because the background is actually the most common class
            (cl + beta*ll)*(1/(distinctness*10000)+1) im using this formula for the loss the sharply increase the loss when
            the whole batch of predictions only contain background forcing it to make a prediction instead of guessing the most
            common class. A standard deviation of 0 over dimensions Batch W and H would mean that there is only one class predicted
            """
            distinctness = semseg.std(dim=(0,-1,-2)).mean()
            loss = (cl + beta*ll)*(1/(distinctness*10000)+1)
            
            #optimization step allowing it to not compute gradients for non altered weight, i think?
            optimizer.zero_grad()
            #backprob, the mean is necesarry for multiple GPUs
            loss.mean().backward()
            optimizer.step()

            #logging 
            pbar.set_postfix(class_loss = cl.item() , label_loss=ll.item(), distinctness=distinctness.item())
            logger.add_scalar("Total_loss", loss.item(), global_step=epoch * l + i)
        
        #save checkpoints after epoch 0 5 and 10
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join("models", name, f"{epoch}ATM_ViT_ckpt.pt"))




