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



def show_images(data, n_samples):

    ''' Function showing image.
        
        Input: tensor images, number of samples to show. '''

    plt.figure(figsize=(8,2))

    for i in range(n_samples):
        image = data[i][0]
        plt.subplot(1,n_samples,i+1)
        plt.axis('off')
        plt.imshow(image.numpy(), cmap='gray')
        
    plt.show()



def resize_data_one_hot(x, y, label, image_size, num_classes):

    ''' Function scaling end encoding real data.
        
        Input:  image array, label array, image size, number of classes.
    
        Output : list of tuple (image torch tensor, label torch tensor), 
                 list of image tensor, list of label tensor. '''

    tuple_imgs, imgs_input, labels_input = [], [], []

    for t, l in zip(x, y):
        if l in label:
            t = torch.tensor(t, dtype = torch.float32).reshape(image_size, image_size)
            t = t/16
            # One-hot encoding
            lab = torch.nn.functional.one_hot(torch.tensor(l), num_classes=num_classes)
            tuple_imgs.append((t, lab))
            imgs_input.append(t)
            labels_input.append(l)

    return tuple_imgs, imgs_input, labels_input



def split_train_test(real_data, idx):

    ''' Function splitting real data into train and test set.

        Input:  list of tuple (image torch tensor, label torch tensor), index for splitting data.
    
        Output : list of training data, list of test data. '''

    train_data, test_data = [], []

    for i, (img, label) in enumerate(real_data):

        if i < idx:

            train_data.append((img, label))
        
        else:
            
            test_data.append((img, label))

    return train_data, test_data



def build_dataset(model_type, noise_dim, path, labels, num_classes, n_samples):

    ''' Function building dataset by making inference from trained qclassical and quantum GAN.

        Input:  model typology, noise dimention, path for loading trained models, list of labels used, 
                number of classes and number of sample to create with inference.
    
        Output : list of image torch tensor, list of tuple (image torch tensor, label torch tensor). '''


    if model_type == 'Classical_linear':

        GAN_imgs, dataset, idx = [], [], 0      

        dirs = os.listdir(path)

        for elem in sorted(dirs):
                             
            model = torch.load(path+elem+'/lin_gen')
            model.eval()
            
            for _ in range(n_samples):      
        
                image = model(torch.rand(1, noise_dim)).view(8,8).cpu().detach()
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
                
            for _ in range(n_samples):   

                image = model(torch.rand(1, noise_dim)).view(8,8).cpu().detach()
                
                GAN_imgs.append(image)

                label = torch.nn.functional.one_hot(torch.tensor(labels[idx]), num_classes=num_classes)
                dataset.append((image, label))
          
            idx += 1                
    
    else:

        print('Network typology not admitted.')

    return GAN_imgs, dataset
    
    
class Classificator(nn.Module):

    ''' Model for classification task.
    
        Input: number of classes.
        
        Output: tensor of dimention (number of classes). '''

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding = 'same')
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.lin = nn.Linear(576, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x,1) # flatten all dimensions except batch
        x = self.lin(x)
        return torch.sigmoid(x)



def training_classification(classificator, train_dataloader, epochs, learning_rate, batch_size, path):
    
    ''' Function for classification training.

        Input:  model, train dataloader, number of epochs, learning rate, batch size and path for saving model.
    
        Output : List of loss values mediated for each epoch. '''

    optimizer = torch.optim.SGD(classificator.parameters(),lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    episode_loss = []
    tot_loss_mean = []

    with tqdm(range(epochs)) as tepochs:

        for _ in tepochs:  

            for data in train_dataloader:
                optimizer.zero_grad()

                # forward + backward + optimization
                outputs = classificator(data[0].view(batch_size,1,8,8))
                
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
    
    ''' Function calculating model accuracy.

        Input:  trained model, test dataloader and path for loading trained model.
    
        Output : accuracy value. '''

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            
            labels = data[1]

            classificator.load_state_dict(torch.load(path + 'classificator.pt'))
            
            outputs = classificator(data[0].view(1,1,8,8))

            act_label = np.argmax(labels)
            pred_label = np.argmax(outputs) 

            if(act_label == pred_label):
                correct += 1
            
            total += 1
    
    print('Accuracy score: ', correct / total)

    return correct / total 



def training_run(model_type, classificator, dataloader, real_dataloader, learning_rate, epochs, batch_size, reset_parameters=True):
    
    ''' Function for a single run with GAN and QGAN data used as training set and real data used as test set.

        Input:  model typology, model, GAN or QGAN dataloader, dataloader from real data, learning rate, epochs, batch size, 
                reset parameters for choosing if resetting model parameters at the end of the run.
    
        Output : run accuracy and loss values. '''


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



def multiple_runs(runs, model_type, c_classificator, q_classificator, synthetic_dataloader, q_synthetic_dataloader, real_dataloader, num_labels, 
                  learning_rate, epochs, batch_size, reset_parameters=True):

    ''' Function for multiple runs with both GAN and QGAN data.

        Input:  model typology, model, GAN or QGAN trained classificators, GAN synthetic data, QGAN synthetic data, dataloader from real data, number of classes,
                learning rate, epochs, batch size, reset parameter for choosing if resetting model parameters at the end of each run.
    
        Output : list of accuracies for classical and quantum models, list of losses for classical and quantum models. '''

    acc_tot = []
    q_acc_tot = []
    loss_tot = []
    q_loss_tot = []

    for run in range(runs):

        print(f'---------------------------------------- RUN {run+1} ----------------------------------------')
        
        print('Classical GAN training: ')
        classical_accuracy, classical_loss = training_run(model_type[0], c_classificator, synthetic_dataloader, real_dataloader, learning_rate, epochs, batch_size, reset_parameters)
        #print(classical_accuracy)

        print('Quantum GAN training: ')
        quantum_accuracy, quantum_loss = training_run(model_type[1], q_classificator, q_synthetic_dataloader, real_dataloader, learning_rate, epochs, batch_size, reset_parameters)

        acc_tot.append(classical_accuracy)
        q_acc_tot.append(quantum_accuracy)
        loss_tot.append(classical_loss)
        q_loss_tot.append(quantum_loss)

    # c_acc_mean = np.mean(acc_tot)
    # c_acc_std = np.std(acc_tot)
    # q_acc_mean = np.mean(q_acc_tot)    
    # q_acc_std = np.std(q_acc_tot)
    # c_loss_mean = np.mean(loss_tot)
    # c_loss_std = np.std(loss_tot)
    # q_loss_mean = np.mean(q_loss_tot)    
    # q_loss_std = np.std(q_loss_tot)
    #delta_acc = [np.abs(acc_tot[i]-q_acc_tot[i]) for i in range(len(acc_tot))]

    dict = {'GAN_data_accuracy': [acc_tot], 'QGAN_data_accuracy': [q_acc_tot]}  
    
    # Saving results in a dataframe 
    df = pd.DataFrame(dict)         
    
    df.to_csv(f'torch_results/Metrics/Classification_accuracies_{num_labels}_labels.csv')

    return acc_tot, q_acc_tot, classical_loss, quantum_loss



def accuracy_boxplot(c_acc_final, q_acc_final, labels):

    ''' Function for plotting accuracies statistics on multiple runs. 
        
        Input: list of accuracies for both classical and quantum models, numer of label combinations tested. '''

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='oldlace')
    
    boxes = ax.boxplot(c_acc_final, patch_artist=True, showmeans = False, showfliers=False, widths = 0.12, labels = labels)
    for box in boxes["boxes"]:
        box.set(facecolor = "green")

    boxes = ax.boxplot(q_acc_final, patch_artist=True, showmeans = False, showfliers=False, widths = 0.15, labels = labels)
    for box in boxes["boxes"]:
        box.set(facecolor = "palegreen")

    fig.text(0.92, 0.73, f'GAN linear\nGen: 134 params\nDisc: 4225 params', backgroundcolor='green', color='black', weight='roman')
    fig.text(0.92, 0.63, f'QGAN linear\nGen: 120 params\nDisc: 4225 params', backgroundcolor='palegreen', color='black', weight='roman')

    ax.set_xlabel('Number of labels', fontsize=16)
    ax.set_ylabel('Accuracy score', fontsize=16)
    plt.grid()
    plt.title(f"Accuracy classification score - Handwritten 8x8 dataset", fontsize=18)
    plt.savefig(f'torch_results/Metrics/Classification_accuracies.jpeg') # Saving boxplot image
    plt.show()