import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml
from pennylane.templates import RandomLayers
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
from tqdm import tqdm

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


class ConvGenerator(nn.Module):

    def __init__(self, z_dim):
        super(ConvGenerator, self).__init__()
        self.z_dim = z_dim

        self.convt_1 = nn.ConvTranspose2d(self.z_dim, 32, 2, 2, 0)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU(32)
        #self.convt_2 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        #self.batch_norm_2 = nn.BatchNorm2d(64)
        #self.relu_2 = nn.ReLU(64)
        self.convt_3 = nn.ConvTranspose2d(32, 1, 4, 4, 0)

    def forward(self, x):

        x = x.view(x.shape + (1, 1))
        #print(x.shape)
        x = self.convt_1(x)
        x = self.batch_norm_1(x)
        #print(x.shape)
        x = self.relu_1(x)
        #x = self.convt_2(x)
        #x = self.batch_norm_2(x)
        #x = self.relu_2(x)
        #print(x.shape)
        x = self.convt_3(x)
        #rint(x.shape)
        
        return x



n_qubits = 5

@qml.qnode(qml.device("lightning.qubit", wires=n_qubits), interface="torch", diff_method="parameter-shift")
def quantum_generator_circuit(noise, gen_weights, gen_n_layers, n_qubits):

    gen_weights = gen_weights.reshape(gen_n_layers, n_qubits)

    # Encoding layer
    #for i in range(n_qubits):
    #    qml.RY(noise[i], wires=i)

    qml.AngleEmbedding(noise, wires=range(n_qubits))

    # PQC layers
    for i in range(gen_n_layers):

        # Rotation gates
        for y in range(n_qubits):
            #for w in range(n_gate_per_layer):
            qml.RX(gen_weights[i][y], wires=y)  
            qml.RY(gen_weights[i][y], wires=y)   
            qml.RZ(gen_weights[i][y], wires=y) 

    #RandomLayers(gen_weights, wires = list(range(n_qubits)), ratio_imprim = 0.1)
            
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
        #self.n_gate_per_layer = n_gate_per_layer
        self.n_generators = n_generators
        self.device = device
        self.vqc_params = nn.ParameterList([nn.Parameter(q_delta * torch.rand(self.gen_n_layers * self.n_qubits), 
                                          requires_grad=True)for _ in range(self.n_generators)])

    def forward(self, x):
        
        patch_size = 2 ** (self.n_qubits - self.ancillary_qubits)

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
            #print(images.shape)

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



class ConvDiscriminator(nn.Module):

    def __init__(self, disc_input_shape):
        super(ConvDiscriminator, self).__init__()
        self.disc_input_shape = disc_input_shape

        self.convt_1 = nn.ConvTranspose2d(self.disc_input_shape, 64, 2, 2, 0)
        self.relu_1 = nn.ReLU(64)
        self.convt_2 = nn.ConvTranspose2d(64, 128, 2, 2, 0)
        self.relu_2 = nn.ReLU(128)
        self.flat = nn.Flatten()
        self.lin = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.convt_1(x)
        #print(x.shape)
        x = self.relu_1(x)
        x = self.convt_2(x)
        x = self.relu_2(x)
        #print(x.shape)
        x = self.flat(x)
        #print(x.shape)
        x = self.lin(x)
        x = self.sigmoid(x)
        #print(x.shape)
        
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
        

    def train_step(self, data):

            # d_losses = []
            # g_losses = []
            # #===============================
            # # Discriminator Network Training
            # #===============================

            # data = data.reshape(-1, self.image_size * self.image_size)

            # self.opt_disc.zero_grad()
            # real_preds = self.disc_net(data).reshape(self.batch_size, 1)
            # real_targets = torch.ones(data.size(0),1)
            # real_loss = torch.nn.functional.binary_cross_entropy(real_preds, real_targets)
            
            # # Generate fake images
            # latent = torch.randn(self.batch_size, self.z_dim)
            # fake_images = self.gen_net(latent)

            # # Pass fake images through discriminator
            # fake_targets = torch.zeros(fake_images.size(0), 1)
            # fake_preds = self.disc_net(fake_images)
            # fake_loss = torch.nn.functional.binary_cross_entropy(fake_preds, fake_targets)

            # # Update discriminator weights
            # d_loss = real_loss + fake_loss
            # d_loss.backward()
            # self.opt_disc.step()
            # #===============================
            # # Generator Network Training
            # #===============================
            # self.opt_gen.zero_grad()
        
            # # Generate fake images
            # latent = torch.randn(self.batch_size, self.z_dim)
            # fake_images = self.gen_net(latent)
            
            # # Try to fool the discriminator
            # preds = self.disc_net(fake_images.view(fake_images.size(0), -1).detach()).reshape(self.batch_size,1)
            # targets = torch.ones(self.batch_size,1)
            # g_loss = torch.nn.functional.binary_cross_entropy(preds, targets)
            
            # # Update generator weights
            # g_loss.backward()
            # self.opt_gen.step()

            # return g_loss, d_loss

        # Defining training data
            data = data.reshape(-1, self.image_size * self.image_size)
            real_data = data.to(self.device)

            # Generating random noise
            noise = torch.rand(self.batch_size, self.z_dim, device=self.device) #* math.pi / 2
            fake_data = self.gen_net(noise)

            # Training discriminator
            self.opt_disc.zero_grad()    

            if self.model == 'Classical_linear' or self.model == 'Quantum_linear':    
                disc_real = self.disc_net(real_data).view(-1)
            elif self.model == 'Classical_convolutional' or self.model == 'Quantum_convolutional':
                disc_real = self.disc_net(real_data.view(1, self.image_size*self.image_size, 1, 1)).view(-1)
            else:
                print('Typology not admitted.')  


            if self.model == 'Classical_linear':
                disc_fake = self.disc_net(fake_data.view(fake_data.size(0), -1).detach()).view(-1)
            elif self.model == 'Quantum_linear':
                disc_fake = self.disc_net(fake_data.detach()).view(-1)
            elif self.model == 'Classical_convolutional'or self.model == 'Quantum_convolutional':
                disc_fake = self.disc_net(fake_data.view(1, self.image_size*self.image_size, 1, 1).detach()).view(-1)            
            else:
                print('Typology not admitted.')

            ld_real = self.disc_loss(disc_real, self.real_labels)
            ld_fake = self.disc_loss(disc_fake, self.fake_labels)
            ld_real.backward()
            ld_fake.backward()
            ld = ld_real + ld_fake
            self.opt_disc.step()

            # Training generator
            self.opt_gen.zero_grad()
            if self.model == 'Classical_linear' or self.model == 'Quantum_linear':
                disc_fake = self.disc_net(fake_data).view(-1)
            elif self.model == 'Classical_convolutional'or self.model == 'Quantum_convolutional':
                disc_fake = self.disc_net(fake_data.view(1, self.image_size*self.image_size, 1, 1)).view(-1)
            else:
                print('Typology not admitted.')

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

        #with tqdm(range(epochs)) as t:

        #    for epoch in t:

        while True:
            for data, _ in self.dataloader:

                lg, ld = self.train_step(data) 

                #t.set_postfix({"Discriminator loss" : ld, "Generator loss" : lg})  

                epoch += 1
                print(f'Epoch_{epoch}')               

                # Saving models each 10 epochs
                if epoch % 10 == 0:

                    test_images = self.gen_net(self.fixed_noise).view(8,1,self.image_size,self.image_size).cpu().detach()
                    
                    if self.model == 'Classical_linear':
                        torch.save(self.gen_net, self.save_path + f'lin_gen_epoch_{epoch}')
                    elif self.model == 'Quantum_linear':
                        torch.save(self.gen_net, self.save_path + f'lin_q_gen_epoch_{epoch}')
                    elif self.model == 'Classical_convolutional':
                        torch.save(self.gen_net, self.save_path + f'conv_gen_epoch_{epoch}')
                    elif self.model == 'Quantum_convolutional':
                        torch.save(self.gen_net, self.save_path + f'conv_q_gen_epoch_{epoch}')
                    else:
                        print('Typology not admitted.')
                    
                    # Save results every 50 iterations
                    if epoch % 50 == 0:
                        results.append(test_images) 
                        torch.save(results, self.save_path + 'synthetic.pt')                                           
                
                self.loss_g.append(lg.item())#.detach().numpy())
                self.loss_d.append(ld.item())#.detach().numpy())     

                #print('Gen params: ', self.gen_net.state_dict())

                if epoch==epochs:
                    break

            if epoch==epochs:
                break  

            torch.save(self.loss_g, self.save_path + 'gen_loss.pt') 
            torch.save(self.loss_d, self.save_path + 'disc_loss.pt')  


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



