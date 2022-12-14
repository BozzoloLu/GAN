from utils import *
import torch
import numpy as np

# Choosing the device
device = torch.device("cpu")


# Setting the random seed
seed = 40
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#----------------- Setting path for saving models -----------------#

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = 'torch_results/QGAN/QGAN_linear/' + current_time + '/'
summary_writer = tf.summary.create_file_writer(save_path)


#----------------- Defining model parameters -----------------#

z_dim = n_qubits
ancillary_qubits = 1
gen_n_layers = 6
#n_gate_per_layer = 3
n_generators = 4
image_size = 8
batch_size = 1
loss = nn.BCELoss()
lr_gen = 0.3
lr_disc = 0.01
epochs = 1500


#----------------- Loading real data -----------------#

digits = datasets.load_digits()

x_train = digits.data
y_train = digits.target

x_train = x_train.reshape(len(x_train), 8, 8)
x_train.shape

rd, inp = resize_data(x_train, y_train, label = (0,1), image_size = 8)
torch.save(inp, save_path + 'real.pt')
dataloader = torch.utils.data.DataLoader(rd, batch_size=batch_size, shuffle=True, drop_last=True)


#----------------- Defining models -----------------#

gen_net = QuantumGenerator(n_qubits = n_qubits, ancillary_qubits = ancillary_qubits, 
                           gen_n_layers = gen_n_layers, n_generators = n_generators, device = device).to(device)
disc_net = Discriminator(image_size).to(device)

runs = 1 # Setting how many run must be executed

loss_g_mean = []
loss_d_mean = []


for run in range(runs): 

    qgan = GAN(model = 'Quantum_linear', dataloader = dataloader, gen_net = gen_net, disc_net = disc_net, z_dim = z_dim, image_size = image_size, 
            batch_size = batch_size, lr_gen = lr_gen, lr_disc = lr_disc, gen_loss = loss, disc_loss = loss, save_path = save_path, device = device)

    qgan.learn(epochs)

    loss_g_mean.append(qgan.loss_g)
    loss_d_mean.append(qgan.loss_d)

    # Saving loss results    
    torch.save(loss_g_mean, save_path + 'gen_loss.pt') 
    torch.save(loss_d_mean, save_path + 'disc_loss.pt') 