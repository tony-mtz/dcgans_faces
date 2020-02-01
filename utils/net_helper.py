
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def conv_layer(chanIn, chanOut, kernel_size = 3,strd=1, padding=1, drop=.04):
    return nn.Sequential(
        nn.Conv2d(chanIn, chanOut, kernel_size,stride=strd, padding=padding),        
        nn.BatchNorm2d(chanOut),
        nn.LeakyReLU(),
        nn.Dropout(drop)
        )

def discr_dense(chanIn):
    return nn.Sequential(
        Flatten(),
        nn.Linear(chanIn, 1),
        nn.Sigmoid() #...switch to bcewlogits if you remove sigmoid
        )

def gen_dense(chanIn, chanOut):
    return nn.Sequential(
        nn.Linear(chanIn, chanOut),
        # nn.ReLU()
        )

def gen_fake_images(model, 
                    device,
                    noise=None, 
                    amount=16, 
                    z_size=100, 
                    display_size=15):

    noise = torch.randn(amount, 1, z_size, device=device)
    model.eval()
    fake = model(noise).detach().cpu()
    plt.figure()
    plt.subplots(figsize=(display_size,display_size))
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1,2,0)))
    plt.show()



   