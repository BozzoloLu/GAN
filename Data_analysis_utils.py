import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.stats import f
import matplotlib.pyplot as plt
import os
import datetime
from tqdm import tqdm


def resize_data_one_hot(x, y, label, image_size, num_classes):

    arr, arr_input, labels = [], [], []

    for t, l in zip(x, y):
        if l in label:
            t = torch.tensor(t, dtype = torch.float32).reshape(image_size, image_size)
            t = t/16
            # One-hot encoding
            lab = torch.nn.functional.one_hot(torch.tensor(l), num_classes=num_classes)
            arr.append((t, lab))
            arr_input.append(t)
            labels.append(l)
    return arr, arr_input, labels


def split_train_test(real_data, idx, num_classes):

    train_data, test_data = [], []

    for i, (img, label) in enumerate(real_data):

        if i < idx:

            train_data.append((img, label))
        
        else:
            
            test_data.append((img, label))

    return train_data, test_data



def build_dataset(model_type, noise_dim, path, labels, num_classes):

    if model_type == 'Classical_linear':

        GAN_imgs, dataset, idx = [], [], 0      

        dirs = os.listdir(path)

        for elem in sorted(dirs):
            #print(elem)
            #file = os.listdir(path+elem)
            #print(file)

        # r=root, d=directories, f=files
        # for r, d, f in os.walk(path): 
        #     #print('directory: ', d)
        #     #print(f)                 
        #     for file in f:                 
        #         if file.endswith("lin_gen_epoch_*"):    

            #print(os.path.join(r, file))                     
            model = torch.load(path+elem+'/lin_gen')
            model.eval()

            for _ in range(500):      
        
                image = model(torch.rand(1, noise_dim)).view(8,8).cpu().detach()
                #print('img shape: ', image.shape)
                GAN_imgs.append(image)
                label = torch.nn.functional.one_hot(torch.tensor(labels[idx]), num_classes=num_classes)
                dataset.append((image, label))                    
                
            idx += 1
            

    elif model_type == 'Quantum_linear':

        GAN_imgs, dataset, idx = [], [], 0  
    
        dirs = os.listdir(path)

        for elem in sorted(dirs):

            model = torch.load(path+elem+'/lin_q_gen')    
            model.eval()              

            for _ in range(500):   

                image = model(torch.rand(1, noise_dim)).view(8,8).cpu().detach()
                
                GAN_imgs.append(image.shape)

                label = torch.nn.functional.one_hot(torch.tensor(labels[idx]), num_classes=num_classes)
                dataset.append((image, label))
                
            idx += 1                
    
    else:

        print('Network typology not admitted.')

    return GAN_imgs, dataset




class Classificator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding = 'same')
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        #self.fc1 = nn.Linear(576, 120)
        #self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(576, num_classes)

    def forward(self, x):
        #print(x.shape)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        #print(x.shape)
        x = torch.flatten(x,1) # flatten all dimensions except batch
        #print(x.shape)
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
        



def training_classification(classificator, train_dataloader, epochs, learning_rate, batch_size, path):

    optimizer = torch.optim.SGD(classificator.parameters(),lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    episode_loss = []
    tot_loss_mean = []

    with tqdm(range(epochs)) as tepochs:

        for epoch in tepochs:  

            running_loss = 0.0
            for i, data in enumerate(train_dataloader):

                #(train, label) = data
                #print(data[0].shape)
                #print(data[1].shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimization
                outputs = classificator(data[0].view(batch_size,1,8,8))#(data[0].size(0), -1))
                #print('Outputs training: ', outputs.shape)
                
                label = data[1].float()

                loss = loss_fn(outputs, label)
                loss.backward()
                optimizer.step()

                episode_loss.append(loss.item())                
                tepochs.set_postfix({'Classificator loss' : loss.item()})
            
            tot_loss_mean.append(np.mean(episode_loss))
            
            torch.save(classificator.state_dict(), path + 'classificator.pt')

    return tot_loss_mean          


        
def evaluate_accuracy(classificator, test_dataloader, path):

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            
            labels = data[1]
            #print('labels: ', labels)
            #print('Data:', data[0].shape)

            classificator.load_state_dict(torch.load(path + 'classificator.pt'))
            
            outputs = classificator(data[0].view(1,1,8,8))#.view(data.size(0), -1)

            act_label = np.argmax(labels) # act_label = 1 (index)
            #print('act labels: ', act_label)
            pred_label = np.argmax(outputs) # pred_label = 1 (index)
            #print('pred labels: ', pred_label)
            if(act_label == pred_label):
                correct += 1
            
            total += 1
    
    # print('correct: ', correct)
    # print('total: ', total)
    print('Accuracy score: ', correct / total)

    return correct / total 



def training_run(model_type, classificator, dataloader, real_dataloader, learning_rate, epochs, batch_size, reset_parameters=True):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if model_type == 'Classical_linear':
        
        save_path = 'torch_results/Classificator/GAN_data/' + current_time + '/'

    elif model_type == 'Quantum_linear':

        save_path = 'torch_results/Classificator/QGAN_data/' + current_time + '/'
    
    else:
        print('Model typology not admitted.')
    summary_writer = tf.summary.create_file_writer(save_path)

    loss_synthetic = training_classification(classificator, dataloader, epochs, learning_rate, batch_size, save_path)
    acc_synthetic = evaluate_accuracy(classificator, real_dataloader, save_path)


    if reset_parameters:
        for layer in classificator.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    
    return acc_synthetic, loss_synthetic
    


def multiple_runs(runs, model_type, c_classificator, q_classificator, synthetic_dataloader, q_synthetic_dataloader, real_dataloader, num_labels, learning_rate, epochs, batch_size, reset_parameters=True):

    acc_tot = []
    q_acc_tot = []
    loss_tot = []
    q_loss_tot = []

    for run in range(runs):

        print(f'---------------------------------------- RUN {run+1} ----------------------------------------')
        
        print('Classical GAN training: ')
        classical_accuracy, classical_loss = training_run(model_type[0], c_classificator, synthetic_dataloader, real_dataloader, learning_rate, epochs, batch_size, reset_parameters=True)
        #print(classical_accuracy)

        print('Quantum GAN training: ')
        quantum_accuracy, quantum_loss = training_run(model_type[1], q_classificator, q_synthetic_dataloader, real_dataloader, learning_rate, epochs, batch_size, reset_parameters=True)

        acc_tot.append(classical_accuracy)
        q_acc_tot.append(quantum_accuracy)
        loss_tot.append(classical_loss)
        q_loss_tot.append(quantum_loss)

    c_acc_mean = np.mean(acc_tot)
    c_acc_std = np.std(acc_tot)
    q_acc_mean = np.mean(q_acc_tot)    
    q_acc_std = np.std(q_acc_tot)

    c_loss_mean = np.mean(loss_tot)
    c_loss_std = np.std(loss_tot)
    q_loss_mean = np.mean(q_loss_tot)    
    q_loss_std = np.std(q_loss_tot)

    delta_acc = [np.abs(acc_tot[i]-q_acc_tot[i]) for i in range(len(acc_tot))]

    dict = {'GAN_data_accuracy': [acc_tot], 'QGAN_data_accuracy': [q_acc_tot]}  
       
    df = pd.DataFrame(dict)         
    # saving the dataframe 
    df.to_csv(f'torch_results/Metrics/Classification_accuracies_{num_labels}_labels.csv')

    return acc_tot, q_acc_tot, delta_acc, classical_loss, quantum_loss#, c_acc_mean, c_acc_std, q_acc_mean, q_acc_std, c_loss_std, , q_loss_std


def show_images(data, n_samples):

    plt.figure(figsize=(8,2))

    for i in range(n_samples):
        image = data[i][0]#.reshape(8, 8)
        plt.subplot(1,n_samples,i+1)
        plt.axis('off')
        plt.imshow(image.numpy(), cmap='gray')
        
    plt.show()


def accuracy_boxplot(c_acc_final, q_acc_final, labels):

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='oldlace')

    #for i in range(num_tests):
    
    boxes = ax.boxplot(c_acc_final, patch_artist=True, showmeans = False, showfliers=False, widths = 0.12, labels = labels)
    for box in boxes["boxes"]:
        box.set(facecolor = "green")

    boxes = ax.boxplot(q_acc_final, patch_artist=True, showmeans = False, showfliers=False, widths = 0.12, labels = labels)
    for box in boxes["boxes"]:
        box.set(facecolor = "palegreen")

    fig.text(0.92, 0.73, f'GAN linear\nGen: 134 params\nDisc: 4225 params', backgroundcolor='green', color='black', weight='roman')
    fig.text(0.92, 0.63, f'QGAN linear\nGen: 120 params\nDisc: 4225 params', backgroundcolor='palegreen', color='black', weight='roman')
    # fig.text(0.92, 0.53, f'QGAN linear\nGen: 120 params\nDisc: 5127 params', backgroundcolor='darkkhaki', color='black', weight='roman')
    # fig.text(0.92, 0.43, f'QGAN linear\nGen: 120 params\nDisc: 5127 params', backgroundcolor='palegoldenrod', color='black', weight='roman')

    ax.set_xlabel('Number of labels', fontsize=16)
    ax.set_ylabel('Accuracy score', fontsize=16)
    plt.grid()
    plt.title(f"Accuracy classification score - Handwritten 8x8 dataset", fontsize=18)
    plt.savefig(f'torch_results/Metrics/Classification_accuracies.jpeg')
    plt.show()