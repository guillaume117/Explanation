
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import pickle
from tqdm import tqdm
import os
import time
from TRAIN.train import TrainDataset
from CNN.ResNet18 import ResNet18
from CNN.VGG import VGG
from CNN.SimpleCNN import SimpleModel
import torch.nn as nn

"""
This class aims to generate num_subset sub-datasets composed of 80% of the original dataset, 
to record them, and to record their indices.
"""



class GenerateDataset:
    def __init__(self,num_subset,name_dataset, name_folder,image_size,sparsity=0.8):
        self.name_dataset=name_dataset
        self.name_folder = name_folder
        self.image_size=image_size
        self.sparsity=sparsity

        self.num_subset = num_subset
        self.dataset_train=None
        self.dataset_val = None
        self.idx_to_class=None
        self.dataset_test = None
        self.path = None
        self.ResName = None
        self.initialise()


    def initialise(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.2])
        ])

        """
        Here you'll find a little bit tricky way to build our dataset for training, validate and test. Since I want
        to keep a good repartition of classes on all my training dataset, and because Subset doesn't inherit targets from dataset,
        I've decided to use the test as validation. The purpose of this is not to achieve a good training, but we must remind that our
        goal is to assess the quality of explanation
       
        """
      
        self.path = f'./{self.name_folder}'
        print(self.path)

        if self.name_dataset == 'FashionMNIST':
   
            os.makedirs(self.path, exist_ok=True)
            try :
                self.dataset_train = datasets.FashionMNIST(root = self.path, transform = transform, download = True, train = True)
            except Exception as e:
                print(f"Une erreur s'est produite : {e}")
                return
            try :
                self.dataset_val = datasets.FashionMNIST(root= self.path, transform = transform, download = True, train = False)
            except Exception as e :
                print(f"Une erreur s'est produite : {e}")
                return
           
        elif self.name_dataset =='MNIST':

            os.makedirs(self.path,exist_ok=True)
            try :
            
                self.dataset_train = datasets.MNIST(root=self.path, transform = transform, download = True, train = True)
            except Exception as e :
                print(f"Une erreur s'est produite : {e}")
                return
            try:
                self.dataset_val = datasets.MNIST(root= self.path, transform = transform, download = True, train = False)

            except Exception as e :
                print(f"Une erreur s'est produite : {e}")


        elif self.name_dataset == 'CIFAR10':
            transforms.Compose([
                transforms.Resize((32, 32)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])  
            ])
    
            os.makedirs(self.path, exist_ok=True)
            try :
                self.dataset_train = datasets.CIFAR10(root = self.path, transform = transform, download = True, train = True)
            except Exception as e:
                print(f"Une erreur s'est produite : {e}")
                return
            try :
                self.dataset_val = datasets.CIFAR10(root= self.path, transform = transform, download = True, train = False)
            except Exception as e :
                print(f"Une erreur s'est produite : {e}")
                return

        else :
            print("The dataset must be chosen between MNIST or FashionMNIST")
            return 

    
    def generate(self):
  

        """
        Here I build a weight in order to sample correctly all my subsets"""

        print("*"*100)
        print("Creation of the weight vector for normalizing the sub-dataset.")
        self.idx_to_class = {v: k for k, v in self.dataset_train.class_to_idx.items()}
        print("Distribution of classes: ", self.get_class_distribution())
        targets = torch.tensor(self.dataset_train.targets)

        class_count = [i for i in self.get_class_distribution().values()]
        class_weights = 1.0/torch.tensor(class_count, dtype=torch.float16) 
        class_weights_all = class_weights[targets]
        print(f"len class weight all : {len(class_weights_all)}")
        print(class_weights_all)
        print("*"*100)
        print("-"*100)
        print("Creation of global train dataset (will be use to test discrepancies)")
     

        dataloader = DataLoader(self.dataset_train, batch_size = len(self.dataset_train))
        dataglobal = []
        for batch, label in dataloader:
            dataglobal.extend(zip(batch,label))
        torch.save(dataglobal,f'./{self.name_folder}/{self.name_dataset}_dataset_global.pt')
        print(f'{self.name_dataset}_dataset_train_global.pt recorded on folder {self.path} ')
        print("-"*100)
        print("*"*100)
        print("-"*100)
        
        print("Creation of validation dataset (is the same for all sub dataset)")

        dataloader = DataLoader(self.dataset_val, batch_size = len(self.dataset_val))
        dataval = []
        for batch, label in dataloader:
            dataval.extend(zip(batch,label))
        torch.save(dataval,f'./{self.name_folder}/{self.name_dataset}_dataset_val.pt')
        print(f'{self.name_dataset}_train_dataval.pt recorded on folder {self.path} ')
        print("-"*100)

        batch_size = int(len(self.dataset_train)*self.sparsity)
        
        for i in tqdm(range(self.num_subset)):

            weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=int(len(class_weights_all)*self.sparsity),
            replacement=False
            )
            print("*"*100)
            print(f"Creation of sub dataset number {i} over {self.num_subset}")
        
            dataloader = DataLoader(self.dataset_train, sampler=weighted_sampler, batch_size=batch_size,shuffle=False)
            """All this work would be useless without retrieving indices!! """
            indice_list = [indice for indice in weighted_sampler]
            #enumerate_indice_list = [indice for indice in enumerate(weighted_sampler)]

            with open(f'{self.name_folder}/{self.name_dataset}indices_data_sampler_{i}_{self.num_subset}.pkl', 'wb') as file:
                pickle.dump(indice_list, file)
                print(f'{self.name_dataset}indices_data_sampler_{i}_{self.num_subset}.pkl recorded on folder {self.name_folder}')
            #with open(f'{self.path}/indices_data_sampler_enumerate_{i}.pkl', 'wb') as file:
            #    pickle.dump(enumerate_indice_list, file)
            #    print(f'indices_data_sampler_{i}_{self.num_subset}.pkl recorded on folder {self.name_folder}')
        
            weighted_dataset = []
        
            for batch ,label in dataloader:
               
            
                weighted_dataset.extend(list(zip(batch,label)))

            torch.save(weighted_dataset,f'{self.path}/{self.name_dataset}_train_dataset_{i}.pt')
            print(f'{self.name_dataset}_train_dataset_{i}.pt recorded on folder {self.path} ')
            print("x"*100)
       

    def get_class_distribution(self):
        count_dict = {k:0 for k,v in self.dataset_train.class_to_idx.items()} # initialise dictionary
    
        for input, label in self.dataset_train:
            label = self.idx_to_class[label]
            count_dict[label] += 1
            
        return count_dict
    

    def train_sub_dataset(self,ResName,num_epoch,batch_size,learning_rate, scheduler= True):
        self.ResName =ResName
        
        if torch.backends.mps.is_available():
             device = torch.device("mps")
        elif torch.cuda.is_available:
            device = torch.device("cuda")
        else: 
            device = torch.device("cpu")


        if self.ResName =='ResNet18':
            for i in range(self.num_subset):
                print("T"*100)
                print(f"Training of subset n {i} over {self.num_subset} on {self.ResName}")
                print("T"*100)
                train_dataset = torch.load(f'{self.name_folder}/{self.name_dataset}_train_dataset_{i}.pt')
                val_dataset= torch.load(f'./{self.name_folder}/{self.name_dataset}_dataset_val.pt')


                model = ResNet18(num_classes=10,hooked=False,num_depth=1)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
                scheduler =  lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                criterion = nn.CrossEntropyLoss()
                




                train=TrainDataset(model, criterion, optimizer, scheduler,num_epoch,learning_rate,batch_size,device = device)
                model = train.train_model(train_dataset,val_dataset)
                torch.save(model.state_dict(), f'{self.name_folder}/{self.ResName}_Trained_dataset_{i}.pth')
        
        if self.ResName =='VGG':
            
            for i in range(self.num_subset):
                print("T"*100)
                print(f"Training of subset n {i} over {self.num_subset} on {self.ResName}")
                print("T"*100)
                train_dataset = torch.load(f'{self.name_folder}/{self.name_dataset}_train_dataset_{i}.pt')
                val_dataset= torch.load(f'./{self.name_folder}/{self.name_dataset}_dataset_val.pt')


                model = VGG(num_classes=10,hooked=False,num_depth=1)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
                scheduler =  lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                criterion = nn.CrossEntropyLoss()
        
        if self.ResName =='VGG_Cifar':
            
            for i in range(self.num_subset):
                print("T"*100)
                print(f"Training of subset n {i} over {self.num_subset} on {self.ResName}")
                print("T"*100)
                train_dataset = torch.load(f'{self.name_folder}/{self.name_dataset}_train_dataset_{i}.pt')
                val_dataset= torch.load(f'./{self.name_folder}/{self.name_dataset}_dataset_val.pt')


                model = VGG(num_classes=10,hooked=False,num_depth=3)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
                scheduler =  lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                criterion = nn.CrossEntropyLoss()




                train=TrainDataset(model, criterion, optimizer, scheduler,num_epoch,learning_rate,batch_size,device = device)
                model = train.train_model(train_dataset,val_dataset)
                torch.save(model.state_dict(), f'{self.name_folder}/{self.ResName}_Trained_dataset_{i}.pth')

        if self.ResName =='SimpleCNN':
            for i in range(self.num_subset):
                print("T"*100)
                print(f"Training of subset n {i} over {self.num_subset} on {self.ResName}")
                print("T"*100)
                train_dataset = torch.load(f'{self.name_folder}/{self.name_dataset}_train_dataset_{i}.pt')
                val_dataset= torch.load(f'./{self.name_folder}/{self.name_dataset}_dataset_val.pt')


                model = SimpleModel(num_classes=10,hooked=False,num_depth=1)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
                scheduler =  lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                criterion = nn.CrossEntropyLoss()
                




                train=TrainDataset(model, criterion, optimizer, scheduler,num_epoch,learning_rate,batch_size,device = device)
                model = train.train_model(train_dataset,val_dataset)
                torch.save(model.state_dict(), f'{self.name_folder}/{self.ResName}_Trained_dataset_{i}.pth')

        else :
            print("Nom de réseau inconnu")


    def get_parameters(self):
        return self.num_subset,self.name_folder,self.name_dataset, self.ResName
    
    def add_network_type(self,ResName):
        self.ResName=ResName
        

    



  
