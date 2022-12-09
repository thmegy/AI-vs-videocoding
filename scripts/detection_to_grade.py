import argparse, tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')



class CustomDataset(Dataset):
    def __init__(self, path, device):
        df = pd.read_csv(path)

        x = df.iloc[:,2:].values
        y = df['NOTE'].values
        y /= 10 # grade between 0 and 1

        self.x=torch.tensor(x,dtype=torch.float32, device=device)
        self.y=torch.tensor(y,dtype=torch.float32, device=device)[:, None]
 
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]


    
class GradeFC(nn.Module):
    def __init__(self, device):
        super(GradeFC, self).__init__()
        self.fc1 = nn.Linear(12, 64, device=device)
        self.fc2 = nn.Linear(64, 128, device=device)
        self.fc3 = nn.Linear(128, 256, device=device)
        self.fc4 = nn.Linear(256, 32, device=device)
        self.fcout = nn.Linear(32, 1, device=device)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        
        output = self.fcout(x)
        output = torch.sigmoid(output)
        
        return output



def train_loop(dataloader, model, loss_fn, optimizer, printout=False):
    num_batches = len(dataloader)
    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    if printout:
        print(f"Train Error: \n Avg loss: {train_loss:>4f}")

    return train_loss

            

def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>4f} \n")

    return test_loss



def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    model = GradeFC(device)
    learning_rate = 1e-4
    batch_size = 256
    epochs = 600

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_set = CustomDataset(args.train_set, device)
    test_set = CustomDataset(args.test_set, device)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    if args.eval:
        model.load_state_dict(torch.load('detection_to_grade.pth'))
    else:
        train_loss_list = []
        test_loss_list = []
        test_epochs = []
        for t in tqdm.tqdm(range(1, epochs+1)):
            if t%25 == 0:
                print(f"Epoch {t}\n-------------------------------")
                train_loss = train_loop(train_loader, model, loss_fn, optimizer, printout=True)
                test_loss = test_loop(test_loader, model, loss_fn)
                test_loss_list.append(test_loss)
                test_epochs.append(t)
            else:
                train_loss = train_loop(train_loader, model, loss_fn, optimizer)
            train_loss_list.append(train_loss)
        print("Done!")

        # save model
        torch.save(model.state_dict(), 'detection_to_grade.pth')
    
        # plot loss
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, epochs+1), train_loss_list, label='train')
        ax1.plot(test_epochs, test_loss_list, label='test')
        ax1.legend()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('MSE')
        fig1.savefig('loss.png')

    # evaluation
    model.eval()
    loss_fn = nn.L1Loss(reduction='none') # get loss for each individual element
    eval_diff = []
    pred_list = []
    target_list = []
    with torch.no_grad():
        for X, y in tqdm.tqdm(test_loader):
            pred = model(X)
            pred_list += (10*pred).tolist()
            eval_diff += (10*loss_fn(pred, y)).tolist()
            target_list += (10*y).tolist()

    print(np.histogram(eval_diff, bins=np.linspace(0, 10, 21)))
    print(np.histogram(pred_list, bins=np.linspace(0, 10, 21)))
    print(np.histogram(target_list, bins=np.linspace(0, 10, 21)))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-set', required=True)
    parser.add_argument('--test-set', required=True)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    main(args)
    
