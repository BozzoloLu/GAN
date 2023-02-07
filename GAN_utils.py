import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from tqdm import tqdm


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cpu")



def resize_data(x, y, label, image_size):

    ''' Function scaling real data.
        
        Input:  image array, label array, image size, number of classes.    
        Output : list of tuple (image torch tensor, label torch tensor), 
                 list of image tensor. '''

    images_tuple = []
    images = []

    for t, l in zip(x, y):
        if l in label:
            t = torch.tensor(t, dtype = torch.float32).reshape(image_size, image_size)
            t = t/16
            images_tuple.append((t, l))
            images.append(t)
    return images_tuple, images



#------------------------------- Classical Linear Generator -------------------------------#

class Generator(nn.Module):

    ''' Model generating images starting from random noise.
        
        Input: noise dimention. 
        Output: torch tensor of size (1, image size x image size). '''

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.dense_layer = nn.Linear(self.z_dim, 1) #32 has 2304 params
        self.relu = nn.LeakyReLU()
        self.lin = nn.Linear(1, 64)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.lin(self.relu(self.dense_layer(x))))



#------------------------------- Quantum Generator -------------------------------#


#------------- Making quantum circuit -------------#
n_qubits = 5

@qml.qnode(qml.device("lightning.qubit", wires=n_qubits), interface="torch", diff_method="parameter-shift")
def quantum_generator_circuit(noise, gen_weights, gen_n_layers, n_qubits):

    ''' Function creating quantum circuit.
    
        Input: noise torch tensor, weights torch tensor, number of layers, number of qubits.
        Ouput: list of probabilities. '''

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



#------------------------------- Patching images with quantum circuit -------------------------------#

class QuantumGenerator(nn.Module):

    ''' Model patching image with variational quantum circuit. 
    
        Input: number of qubits, number of ancillary qubits, number of VQC layers, number of circuits (generators) for patching, device for running. 
        Output: torch tensor of size (1, image size x image size). '''

    def __init__(self, n_qubits, ancillary_qubits, gen_n_layers, n_generators, device):
        super(QuantumGenerator, self).__init__()

        self.n_qubits = n_qubits
        self.ancillary_qubits = ancillary_qubits
        self.gen_n_layers = gen_n_layers
        self.n_generators = n_generators
        self.device = device
        self.vqc_params = nn.ParameterList([nn.Parameter(torch.rand(self.gen_n_layers * self.n_qubits), 
                                            requires_grad=True)for _ in range(self.n_generators)])

    def forward(self, x):
        
        patch_size = 2 ** (self.n_qubits - self.ancillary_qubits)
        images = torch.Tensor(x.size(0), 0).to(self.device)

        # Iterating over all generators
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


#------------------------------- Classical Linear Discriminator -------------------------------#


class Discriminator(nn.Module):

    ''' Model for binary classification task.
    
        Input: image size.        
        Output: tensor of dimention 2. '''

    def __init__(self, image_size):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.linear1 = nn.Linear(self.image_size * self.image_size, 64)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear2(self.relu1(self.linear1(x))))


class GAN():

    ''' Class trining generator and discriminator models. 
        
        Input: model typology, real dataloader, generator model, discriminator model, noise dimention, image size, batch size, 
               loss, generator learning rate, discriminator learning rate, device for running, path for saving. 
        Output: loss values and model parameters saved. '''
        

    def __init__(self, model_type, dataloader, gen_net, disc_net, noise_dim, image_size, batch_size, loss, lr_gen, lr_disc, 
                 device, save_path):

        "For 0 and 1 features ---> opt = torch.Adam, random = torch.randn, epochs = 1000, lr_disc = lr_gen = 0.0001"

        self.model_type = model_type
        self.dataloader = dataloader
        self.gen_net = gen_net
        self.disc_net = disc_net
        self.noise_dim = noise_dim 
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.loss = loss
        self.device = device
        self.save_path = save_path

        self.gen_opt = optim.SGD(self.gen_net.parameters(), lr=self.lr_gen)
        self.disc_opt = optim.SGD(self.disc_net.parameters(), lr=self.lr_disc)        

        self.real_labels = torch.full((self.batch_size,), 1.0, dtype=torch.float, device=self.device)
        self.fake_labels = torch.full((self.batch_size,), 0.0, dtype=torch.float, device=self.device)

        self.loss_g = []
        self.loss_d = []

    
    def train_step(self, data):

        # Data for training the discriminator
        data = data.reshape(-1, self.image_size * self.image_size)
        real_data = data.to(self.device)

        # Generating noise
        noise = torch.rand(self.batch_size, self.noise_dim, device=self.device) * np.pi / 2
        fake_data = self.gen_net(noise)

        #---------------- Discriminator training ----------------# 
        self.disc_opt.zero_grad()
        disc_out_real = self.disc_net(real_data).view(-1)
        disc_out_fake = self.disc_net(fake_data.detach()).view(-1)

        disc_error_real = self.loss(disc_out_real, self.real_labels)
        disc_error_fake = self.loss(disc_out_fake, self.fake_labels)
        
        disc_error_real.backward()
        disc_error_fake.backward()

        disc_error = disc_error_real + disc_error_fake
        self.disc_opt.step()

        #---------------- Generator training ----------------#
        self.gen_opt.zero_grad()
        # Generating fake images
        #noise = torch.rand(self.batch_size, self.noise_dim)* np.pi / 2
        #fake_data = self.gen_net(noise)
        disc_out_fake = self.disc_net(fake_data).view(-1)
        gen_error = self.loss(disc_out_fake, self.real_labels)
        gen_error.backward()
        self.gen_opt.step()        

        self.loss_g.append(gen_error.item())
        self.loss_d.append(disc_error.item())

        return gen_error, disc_error


    def learn(self, epochs):

        # Fixed noise to track the generated images throughout training
        fixed_noise = torch.rand(8, self.noise_dim, device=self.device) * np.pi / 2

        self.results = []        
        self.ep_d_loss = []
        self.ep_g_loss = []                          
        
        with tqdm(range(epochs)) as tepochs:

            for _ in tepochs:

                self.loss_g = []
                self.loss_d = []     
        
                for data, _ in self.dataloader:

                    lg, ld = self.train_step(data)

                    self.loss_g.append(lg.item())
                    self.loss_d.append(ld.item())                    

                test_images = self.gen_net(fixed_noise).view(8,1,self.image_size,self.image_size).cpu().detach()                
                
                # Collecting and saving results
                self.results.append(test_images)
                torch.save(self.results, self.save_path + 'synthetic.pt')  

                # Collecting and saving losses
                self.ep_g_loss.append(np.mean(self.loss_g))
                self.ep_d_loss.append(np.mean(self.loss_d))
                torch.save(self.ep_g_loss, self.save_path + 'gen_loss.pt') 
                torch.save(self.ep_d_loss, self.save_path + 'disc_loss.pt')  
                
                tepochs.set_postfix({'Generator loss' : np.mean(self.loss_g), 'Discriminator loss': np.mean(self.loss_d)})

            # Saving generator model
            if self.model_type == 'Classical_linear':
                torch.save(self.gen_net, self.save_path + f'lin_gen')
            elif self.model_type == 'Quantum_linear':
                torch.save(self.gen_net, self.save_path + f'lin_q_gen')
            else:
                print('Typology not admitted.')



def run_model(dataloader, generator, discriminator, noise_dim, image_size, batch_size, loss, lr_gen, lr_disc, device, epochs, 
              model_type, reset_parameters=True):
    
    ''' Function managing single run.
        
        Input: dataloader, generator trained model, discriminator trained model, noise dimention, image size, batch size, loss, 
               generator learning rate, discriminator learning rate, device for running, epochs, model typology, reset parameters 
               for choosing if resetting model parameters at the end of the run. '''


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if model_type == 'Classical_linear':
        
        save_path = 'torch_results/GAN/GAN_linear/' + current_time + '/'

    elif model_type == 'Quantum_linear':

        save_path = 'torch_results/QGAN/QGAN_linear/' + current_time + '/'

    else:
        print('Typology not admitted.')
    summary_writer = tf.summary.create_file_writer(save_path)

    gan = GAN(model_type = model_type, dataloader = dataloader, gen_net = generator, disc_net = discriminator, noise_dim = noise_dim, 
                image_size = image_size, batch_size = batch_size, loss = loss, lr_gen = lr_gen, lr_disc = lr_disc, 
                device = device, save_path = save_path)

    gan.learn(epochs)

    if reset_parameters:

        for layer in gan.gen_net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for layer in gan.disc_net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        #return gan.ep_g_loss, gan.ep_d_loss
    #return gan.results#, gan.ep_g_loss, gan.ep_d_loss


def generated_images(results):

    ''' Function visualizing images generation during training.
        
        Input: list of images torch tensors. 
        output: batch of images for choosen epochs'''


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
                ax.set_title(f'Epoch {i+1}', loc='left', color = 'White')
            fig.add_subplot(ax)

    plt.show()

