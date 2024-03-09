
#NotebookCase=GenerateDataset(10, name_dataset='MNIST', name_folder='Evaluation_2',image_size=28)
#ResName='SimpleCNN'



from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch
from CNN.ResNet18 import ResNet18
from CNN.SimpleCNN import SimpleModel
from CNN.VGG import VGG
import numpy as np
from tqdm import tqdm
from util.GenerateAndTrain import GenerateDataset



class ComputeHeatmap:
    def __init__(self,case,num_sample):
        self.case = case
        self.num_sample=num_sample
        self.num_subset,self.name_folder,self.name_dataset,self.ResName=self.case.get_parameters()
        if self.ResName == None:
            print('X'*100)
            print('Network type is not listed')
            print("Please use GenerateDataset.add_network_type(Network)")
            print("X*100")


        #dataloader= DataLoader(train_dataset,shuffle = False, batch_size=len(train_dataset))
        #train_dataset_unwrapped =[]
        #for batch,label in dataloader:
            #train_dataset_unwrapped.extend(list(zip(batch,label)))
        if self.ResName == 'ResNet18':
            self.model = ResNet18(num_classes=10,hooked=True)  

        elif self.ResName== 'SimpleCNN':
            self.model = SimpleModel(num_classes=10, hooked=True)

        elif self.ResName =='VGG':
            self.model =VGG(num_classes=10,hooked=True)
        self.train_dataset = torch.load(f'./{self.name_folder}/{self.name_dataset}_dataset_global.pt')

        self.num_sample = num_sample
        if self.num_sample>len(self.train_dataset):
            print(f"The number of sample must be lower than {len(self.train_dataset)}")

        self.pred_acc =np.zeros((self.num_sample,self.num_subset))
        self.heatmap_list=[]
        self.heatmap_global =[]
    



    def compute_heatmap(self):    
        for j in tqdm(range(self.num_subset)):
            self.model.load_state_dict(torch.load(f'{self.name_folder}/{self.ResName}_Trained_dataset_{j}.pth'))

            self.model.eval()
            self.heatmap_list=[]
            for i in range(self.num_sample):
                img=torch.tensor(self.train_dataset[i][0])
                label=torch.tensor(self.train_dataset[i][1])
                #noise = torch.randn_like(img) * noise_level
                #img= img+noise
                #img = torch.clamp(img, 0, 1)
                label = int(label)
                #img = F.interpolate(img, size=new_size, mode='bilinear', align_corners=False)
                img = img.float().unsqueeze(1)
                pred=self.model(img)
                P = pred.argmax().numpy()

                if P == label:
                    self.pred_acc[i,j]=1
                
                pred[:,P].backward()
                gradients = self.model.get_activations_gradient()
        
                pooled_gradients = torch.mean(gradients, dim=[2,3])

                activations = self.model.get_activations().detach()
                num_act = activations.size(1)
            
                
                for i in range(num_act):
                    activations[ :,i, :, :] *= pooled_gradients[:,i]
                    

                heatmap = torch.mean(activations, dim=1).squeeze()
                heatmap = np.maximum(heatmap, 0)
                heatmap /= torch.max(heatmap)
                heatmap = np.array(heatmap)
                self.heatmap_list.append(heatmap)
            self.heatmap_global.append(self.heatmap_list)


    def get_heatmap(self):
        return self.heatmap_global
    
    def get_pred_acc(self):
        return self.pred_acc
        
