
import torch
from torch.utils.data import DataLoader
import time
import copy

 
"""
 This is my train method, classic. 
 Once will remarks that there are no test in this method, accordingly to the scope of our 
 subject. 
    The method is a classic train method, with the following steps:  
    - Load the data
    - Set the model to train mode
    - Set the optimizer to zero_grad
    - Set the scheduler to step
 
"""    
class TrainDataset:
    def __init__(self,model,criterion,
                 optimizer,scheduler,num_epochs,learning_rate,batch_size,device):
        self.device = device
        self.model = model
        self.criterion=criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.learning_rate =learning_rate




    def train_model(self,train_dataset,val_dataset):
    
        val_acc = []
        val_loss = []
        train_acc = []
        train_loss = []
        epoch=0

        train_loader  = DataLoader(dataset = train_dataset, batch_size=self.batch_size, shuffle =True)
        val_loader = DataLoader(dataset = val_dataset, batch_size=self.batch_size,shuffle=True)

        dataloaders = {'train': train_loader, 'val': val_loader}
        image_datasets = {'train': train_dataset, 'val': val_dataset}
        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0
        list = {'train': {'acc': train_acc, 'loss': train_loss}, 
            'val':{'acc': val_acc, 'loss': val_loss}}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            for phase in ['train','val']:
                if phase =='train':
                    self.scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()
                running_loss=0.0
                running_corrects = 0.0
                for inputs, labels in dataloaders[phase]:

                    if self.device == 'mps':
                        inputs = inputs.to(self.device).float()
                        labels = labels.to(self.device).float()
                    else :
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / dataset_sizes[phase]

                if self.device =='mps':
                    epoch_acc = running_corrects.float() / dataset_sizes[phase]
                else :
                    epoch_acc = running_corrects/ dataset_sizes[phase]
                list[phase]['loss'].append(epoch_loss)
                list[phase]['acc'].append(epoch_acc.item())

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
        
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            
            print()
            
        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        
            
        return self.model







