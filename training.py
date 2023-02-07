import torch
import torch.nn as nn
import numpy as np
import random
from GAN_utils import *
from sklearn import datasets

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cpu")


#----------------------------------- Setting up parameters -----------------------------------#

labels = [0]    # Choosing a label among [0,1,2,3,4]
epochs = 6
batch_size = 1
noise_dim = 6
image_size = 8
loss = nn.BCELoss()
lr_gen = 0.3#0.03# 0.001
lr_disc = 0.01#0.001
n_qubits = 5 
ancillary_qubits = 1  
gen_n_layers = 6  
n_generators = 4 
model_type = ['Classical_linear', 'Quantum_linear']


#----------------------------------- Loading data -----------------------------------#

digits, targets = datasets.load_digits(return_X_y=True)
rd, _ = resize_data(digits, targets, label = labels, image_size = image_size)
dataloader = torch.utils.data.DataLoader(rd, batch_size=batch_size, shuffle=True, drop_last=True)


#----------------------------------- Training Classical GAN and Quantum GAN -----------------------------------#

discriminator = Discriminator(image_size).to(device)
qgenerator = QuantumGenerator(n_qubits, ancillary_qubits, gen_n_layers, n_generators, device).to(device)
generator = Generator(noise_dim).to(device)

results = run_model(dataloader, generator, discriminator, noise_dim, image_size, batch_size, loss, lr_gen, lr_disc, 
                    device, epochs, model_type[0])
results