# Library imports
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml
from sklearn import datasets
import tensorflow as tf
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import time
import datetime
from alive_progress import alive_bar



n_qubits = 5


@qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="torch", diff_method="parameter-shift")
def quantum_generator_circuit(noise, gen_weights, gen_n_layers, n_qubits):

    gen_weights = gen_weights.reshape(gen_n_layers, n_qubits)

    # Encoding layer
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # PQC layers
    for i in range(gen_n_layers):

        # Rotation gates
        for y in range(n_qubits):
            qml.RY(gen_weights[i][y], wires=y)

        # Entangling gates
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    # Returning probability of each computational basis state
    return qml.probs(wires=list(range(n_qubits)))



class QuantumGenerator(nn.Module):

    def __init__(self, n_qubits, ancillary_qubits, gen_n_layers, n_generators, device, q_delta=1):

        self.n_qubits = n_qubits
        self.ancillary_qubits = ancillary_qubits
        self.gen_n_layers = gen_n_layers
        self.n_generators = n_generators
        self.device = device

        super().__init__()

        self.q_params = nn.ParameterList([nn.Parameter(q_delta * torch.rand(self.gen_n_layers * self.n_qubits), 
                                          requires_grad=True)for _ in range(self.n_generators)])

    def forward(self, x):
        
        patch_size = 2 ** (n_qubits - self.ancillary_qubits)

        images = torch.Tensor(x.size(0), 0).to(self.device)

        # Iterate over all sub-generators
        for params in self.q_params:

            
            patches = torch.Tensor(0, patch_size).to(self.device)
            for elem in x:

                probs = quantum_generator_circuit(elem, params, self.gen_n_layers, self.n_qubits)
                partial_meas = probs[: (2 ** (n_qubits - self.ancillary_qubits))]
                partial_meas /= torch.sum(probs)

            
                out = partial_meas / torch.max(partial_meas)
                out = out.float().unsqueeze(0)
                patches = torch.cat((patches, out))

            # define the image
            images = torch.cat((images, patches), 1)

        return images



class Discriminator(nn.Module):

    def __init__(self, image_size):
        super(Discriminator, self).__init__()

        self.image_size = image_size

        self.model = nn.Sequential(
                                    # Inputs to first hidden layer (num_input_features -> 64)
                                    nn.Linear(self.image_size * self.image_size, 64),
                                    nn.ReLU(),
                                    # First hidden layer (64 -> 16)
                                    nn.Linear(64, 16),
                                    nn.ReLU(),
                                    # Second hidden layer (16 -> output)
                                    nn.Linear(16, 1),
                                    nn.Sigmoid(),
                                    )

    def forward(self, x):
        return self.model(x)




class QGAN():
    def __init__(self, dataloader, gen_net, disc_net, z_dim, image_size, batch_size, lrG, lrD, gen_loss, disc_loss, save_path, device):

        self.dataloader = dataloader
        self.gen_net = gen_net
        self.disc_net = disc_net
        self.z_dim = z_dim
        self.image_size = image_size
        self.batch_size = batch_size
        self.lrG = lrG
        self.lrD = lrD
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.save_path = save_path
        self.device = device

        # Optimisers
        self.optD = optim.SGD(self.disc_net.parameters(), lr=self.lrD)
        self.optG = optim.SGD(self.gen_net.parameters(), lr=self.lrG)

        self.real_labels = torch.full((self.batch_size,), 1.0, dtype=torch.float, device=self.device)
        self.fake_labels = torch.full((self.batch_size,), 0.0, dtype=torch.float, device=self.device)  

        self.loss_g, self.loss_d = [], []  
        #self.total_fid = []                       

    # def generated_and_save_images(self, results):

    #     fig = plt.figure(figsize=(20, 10))
    #     outer = gridspec.GridSpec(5, 2, wspace=0.1)

    #     for i, images in enumerate(results):
    #         inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0), subplot_spec=outer[i])
            
    #         images = torch.squeeze(images, dim=1)
    #         for j, im in enumerate(images):

    #             ax = plt.Subplot(fig, inner[j])
    #             ax.imshow(im.numpy(), cmap="gray")
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             if j==0:
    #                 ax.set_title(f'Iteration {50+i*50}', loc='left', color = 'White')
    #             fig.add_subplot(ax)

    #     plt.show()

    # def calculate_fid(self, dist1, dist2):

    #     # calculate mean and covariance statistics
    #     mu1, sigma1 = dist1.mean(axis=0), cov(dist1, rowvar=False)
    #     mu2, sigma2 = dist2.mean(axis=0), cov(dist2, rowvar=False)
        
    #     # calculate sum squared difference between means
    #     ssdiff = torch.sum((mu1 - mu2)**2.0)
    #     # calculate sqrt of product between cov
    #     covmean = sqrtm(sigma1.dot(sigma2))

    #     # check and correct imaginary numbers from sqrt
    #     if iscomplexobj(covmean):
    #         covmean = covmean.real
            
    #     # calculate score
    #     fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    #     return fid

    def train_step(self, data):

        data = data.reshape(-1, self.image_size * self.image_size)
        real_data = data.to(self.device)

        noise = torch.rand(self.batch_size, self.z_dim, device=self.device) 
        fake_data = self.gen_net(noise)

        # Discriminator training
        self.disc_net.zero_grad()
        outD_real = self.disc_net(real_data).view(-1)
        outD_fake = self.disc_net(fake_data.detach()).view(-1)

        errD_real = self.disc_loss(outD_real, self.real_labels)
        errD_fake = self.disc_loss(outD_fake, self.fake_labels)
     
        errD_real.backward()
        errD_fake.backward()

        errD = errD_real + errD_fake
        self.optD.step()

        # Generator training
        self.gen_net.zero_grad()
        outD_fake = self.disc_net(fake_data).view(-1)
        errG = self.gen_loss(outD_fake, self.real_labels)

        errG.backward()
        self.optG.step()

        return errG, errD


    def learn(self, epochs):

        
        self.fixed_noise = torch.rand(8, self.z_dim, device=self.device) 

        
        epoch = 0      

        results = []

        with alive_bar(epochs, force_tty = True) as bar:

            while True:
                
                for _, (data, _) in enumerate(self.dataloader):

                    lg, ld = self.train_step(data)                
                    
                    epoch += 1

                    #time.sleep(0.05) 
                    bar()                   

                    # Show loss values         
                    if epoch % 10 == 0:
                        #print(f'Iteration: {epoch}, Generator Loss: {lg:0.3f}, Discriminator Loss: {ld:0.3f}')
                        test_images = self.gen_net(self.fixed_noise).view(8,1,self.image_size,self.image_size).cpu().detach()
                        torch.save(self.gen_net, self.save_path + f'q_gen_epoch_{epoch}')
                        #torch.save(self.gen_net.state_dict(), save_path + f'q_gen_epoch_{epoch}')
                        
                        # Save images every 50 iterations
                        if epoch % 50 == 0:
                            results.append(test_images)  
                            torch.save(results, self.save_path + 'synthetic.pt')     
                            #fid = self.calculate_fid(data[0], results[0][0][0])  
                            #self.total_fid.append(fid.item())                               
                    
                    self.loss_g.append(lg.detach().numpy())
                    self.loss_d.append(ld.detach().numpy())      

                    if epoch == epochs:
                        break
                if epoch == epochs:
                    break       



def generated_images(results):

    fig = plt.figure(figsize=(20, 10))
    outer = gridspec.GridSpec(len(results)//2, 2, wspace=0.1)

    for i, images in enumerate(results):
        inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0), subplot_spec=outer[i])
        
        images = torch.squeeze(images, dim=1)
        for j, im in enumerate(images):

            ax = plt.Subplot(fig, inner[j])
            ax.imshow(im.numpy(), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if j==0:
                ax.set_title(f'Run {i+1}', loc='left', color = 'White')
            fig.add_subplot(ax)

    plt.show()


def resize_data(x, y, label, image_size):

    arr = []
    arr_input = []

    for t, l in zip(x, y):
        if l in label:
            t = torch.tensor(t, dtype = torch.float32).reshape(image_size, image_size)
            t = t/16
            arr.append((t, l))
            arr_input.append(t)
    return arr, arr_input
  