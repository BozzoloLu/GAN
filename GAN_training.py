from utils.Linear_GAN_utils import *
import torch
import numpy as np
#from numpy import cov
#from numpy import trace
#from numpy import iscomplexobj
#from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

device = torch.device("cpu")

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = 'torch_results/GAN/GAN_linear/' + current_time + '/'
summary_writer = tf.summary.create_file_writer(save_path)


z_dim = 5
image_size = 8
batch_size = 1
loss = nn.BCELoss()
lr_gen = 0.3
lr_disc = 0.01
epochs = 500


digits = datasets.load_digits()

x_train = digits.data
y_train = digits.target

x_train = x_train.reshape(len(x_train), 8, 8)
x_train.shape

rd, inp = resize_data(x_train, y_train, label = (0,), image_size = 8)
torch.save(inp, save_path + 'real.pt')
dataloader = torch.utils.data.DataLoader(rd, batch_size=batch_size, shuffle=True, drop_last=True)


gen_net = Generator(z_dim = z_dim).to(device)
disc_net = Discriminator(image_size).to(device)

runs = 1

loss_g_tot = []
loss_d_tot = []
#fid_tot = []

for run in range(runs): 

    gan = GAN(model = 'Classical', dataloader = dataloader, gen_net = gen_net, disc_net = disc_net, z_dim = z_dim, image_size = image_size, 
                batch_size = batch_size, lr_gen = lr_gen, lr_disc = lr_disc, gen_loss = loss, disc_loss = loss, save_path = save_path, device = device)

    gan.learn(epochs)

    loss_g_tot.append(gan.loss_g)
    loss_d_tot.append(gan.loss_d)
    #fid_tot.append(gan.total_fid)
    
    torch.save(loss_g_tot, save_path + 'gen_loss.pt') 
    torch.save(loss_d_tot, save_path + 'disc_loss.pt') 
    #torch.save(fid_tot, save_path + 'fid.pt') 