
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml
from sklearn import datasets
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchsummary import summary

import time
import datetime
from alive_progress import alive_bar

# from numpy import cov
# from numpy import trace
# from numpy import iscomplexobj
# from scipy.linalg import sqrtm

class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense_layer = nn.Linear(self.z_dim, 64)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))



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
        super(QuantumGenerator, self).__init__()

        self.n_qubits = n_qubits
        self.ancillary_qubits = ancillary_qubits
        self.gen_n_layers = gen_n_layers
        self.n_generators = n_generators
        self.device = device
        self.vqc_params = nn.ParameterList([nn.Parameter(q_delta * torch.rand(self.gen_n_layers * self.n_qubits), 
                                          requires_grad=True)for _ in range(self.n_generators)])

    def forward(self, x):
        
        patch_size = 2 ** (n_qubits - self.ancillary_qubits)

        images = torch.Tensor(x.size(0), 0).to(self.device)

        # Iterate over all sub-generators
        for params in self.vqc_params:
            
            patches = torch.Tensor(0, patch_size).to(self.device)
            for elem in x:

                probs = quantum_generator_circuit(elem, params, self.gen_n_layers, self.n_qubits)
                partial_measure = probs[: (2 ** (n_qubits - self.ancillary_qubits))]
                partial_measure /= torch.sum(probs)
            
                out = partial_measure / torch.max(partial_measure)
                out = out.float().unsqueeze(0)
                patches = torch.cat((patches, out))

            # Building the image
            images = torch.cat((images, patches), 1)

        return images


class Discriminator(nn.Module):

    def __init__(self, image_size):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.linear1 = nn.Linear(self.image_size * self.image_size, 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 16)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        return x


class GAN():
    def __init__(self, model, dataloader, gen_net, disc_net, z_dim, image_size, batch_size, lr_gen, lr_disc, gen_loss, disc_loss, save_path, device):

        self.model = model
        self.dataloader = dataloader
        self.gen_net = gen_net
        self.disc_net = disc_net
        self.z_dim = z_dim
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.save_path = save_path
        self.device = device

        self.opt_gen = optim.SGD(self.gen_net.parameters(), lr=self.lr_gen)
        self.opt_disc = optim.SGD(self.disc_net.parameters(), lr=self.lr_disc)        

        self.real_labels = torch.full((self.batch_size,), 1.0, dtype=torch.float, device=self.device)
        self.fake_labels = torch.full((self.batch_size,), 0.0, dtype=torch.float, device=self.device)        

        self.loss_g, self.loss_d = [], []
        #self.total_fid = []

    # def calculate_fid(self, act1, act2):

    #     # calculate mean and covariance statistics
    #     mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    #     mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        
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

        # Defining training data
        data = data.reshape(-1, self.image_size * self.image_size)
        real_data = data.to(self.device)

        # Generating random noise
        noise = torch.rand(self.batch_size, self.z_dim, device=self.device) #* math.pi / 2
        fake_data = self.gen_net(noise)

        # Training the discriminator
        self.disc_net.zero_grad()        
        disc_real = self.disc_net(real_data).view(-1)

        if self.model == 'Classical':
            disc_fake = self.disc_net(fake_data.view(fake_data.size(0), -1).detach()).view(-1)
        elif self.model == 'Quantum':
            disc_fake = self.disc_net(fake_data.detach()).view(-1)
        else:
            print('Typology not admitted.')

        ld_real = self.disc_loss(disc_real, self.real_labels)
        ld_fake = self.disc_loss(disc_fake, self.fake_labels)
        ld_real.backward()
        ld_fake.backward()
        ld = ld_real + ld_fake
        self.opt_disc.step()

        # Training the generator
        self.gen_net.zero_grad()
        disc_fake = self.disc_net(fake_data).view(-1)
        lg = self.gen_loss(disc_fake, self.real_labels)
        lg.backward()
        self.opt_gen.step()

        return lg, ld


    def learn(self, epochs):

        # Defining a fixed noise for tracking the training progress 
        self.fixed_noise = torch.rand(8, self.z_dim, device=self.device) #* math.pi / 2

        # Initializing epochs
        epoch = 0        

        results = []

        with alive_bar(epochs, force_tty = True) as bar:

            while True:            
                    
                for _, (data, _) in enumerate(self.dataloader):

                    lg, ld = self.train_step(data)                
                    
                    epoch += 1

                    time.sleep(0.05)
                    bar()

                    # Saving models each 10 epochs
                    if epoch % 10 == 0:

                        test_images = self.gen_net(self.fixed_noise).view(8,1,self.image_size,self.image_size).cpu().detach()
                        
                        if self.model == 'Classical':
                            torch.save(self.gen_net, self.save_path + f'gen_epoch_{epoch}')
                        elif self.model == 'Quantum':
                            torch.save(self.gen_net, self.save_path + f'q_gen_epoch_{epoch}')
                            #torch.save(self.gen_net.state_dict(), self.save_path + f'gen_epoch_{epoch}')
                        else:
                            print('Typology not admitted.')
                        
                        # Save results every 50 iterations
                        if epoch % 50 == 0:
                            results.append(test_images) 
                            torch.save(results, self.save_path + 'synthetic.pt')
                            #print(results[0][0][0].shape)
                            #print(data[0].shape)                                 
                            #fid = self.calculate_fid(data[0], results[0][0][0])  
                            #self.total_fid.append(fid.item())
                            #print('fid: ', fid)      
                            #torch.save({'Gen_state_dict': self.gen_net.state_dict(),
                            #            'Disc_state_dict': self.disc_net.state_dict()}, save_path + f'GAN_epoch_{epoch}')                                
                    
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



