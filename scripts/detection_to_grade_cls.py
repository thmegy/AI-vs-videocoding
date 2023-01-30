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
from collections import Counter
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support, top_k_accuracy_score
from torch.utils.tensorboard import SummaryWriter



class CustomDataset(Dataset):
    def __init__(self, x, y, device, augmentation=None):

        y = np.digitize(y, bins=np.linspace(0, 10, 6)[1:-1])

        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        
        self.augmentation = augmentation

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.augmentation is not None:
            return self.augmentation(self.x[idx]), self.augmentation(self.y[idx])
        else:
            return self.x[idx], self.y[idx]

    def get_balance_weigths(self):
        class_sample_count = torch.unique(self.y, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = torch.tensor([weight[t] for t in self.y])
        return samples_weight
    


def gaussian_noise(x):
    return torch.clamp(torch.normal(x, 1), min=0, max=10)


    
class GradeFC(nn.Module):
    def __init__(self, device):
        super(GradeFC, self).__init__()
        self.fc1 = nn.Linear(8, 64, device=device)
        self.fc2 = nn.Linear(64, 126, device=device)
        self.fc3 = nn.Linear(126, 252, device=device)
        self.fc4 = nn.Linear(252, 126, device=device)
        self.fc5 = nn.Linear(126, 64, device=device)
        self.fcout = nn.Linear(64, 5, device=device)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        
        output = self.fcout(x)
        output = F.softmax(output, dim=1)
        
        return output



def train_loop(dataloader, model, loss_fn, optimizer, printout=False):
    num_batches = len(dataloader)
    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y).squeeze()
        loss = loss.mean()
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
    pred_list = []
    pred_score_list = []
    target_list = []
    test_loss = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred_list += pred.argmax(dim=1).tolist()
            pred_score_list += pred.tolist()
            target_list += y.squeeze().tolist()
            test_loss += loss_fn(pred, y).item()

    num_batches = len(dataloader)
    test_loss /= num_batches

    return test_loss, pred_list, pred_score_list, target_list



def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    writer = SummaryWriter(f'log_tensorboard/detection_to_grade/cls/{args.tb_dir}')
    
    df = pd.read_csv(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,2:].values, df['NOTE'].values, test_size=0.2, random_state=1)
#    x_train, x_test, y_train, y_test = train_test_split(df['Moy_degradation'].values[:,None], df['NOTE'].values, test_size=0.2, random_state=1) # for fake data
    scaler = StandardScaler()
    x_train = scaler.fit_transform( x_train )
    x_test = scaler.transform( x_test )

    train_set = CustomDataset(x_train, y_train, device)
    test_set = CustomDataset(x_test, y_test, device)

    sample_weights = train_set.get_balance_weigths()
    train_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), sampler=train_sampler)
#    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
        
    model = GradeFC(device)
    learning_rate = 1e-4
    epochs = 3000

    train_loss_fn = nn.CrossEntropyLoss(reduction='none')
    test_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    if not args.eval:
        best_test_loss = None
        for t in tqdm.tqdm(range(1, epochs+1)):
            train_loss = train_loop(train_loader, model, train_loss_fn, optimizer)
            writer.add_scalar('train/loss', train_loss, t)
            
            test_loss, pred_list, pred_score_list, target_list = test_loop(test_loader, model, test_loss_fn)
            writer.add_scalar('test/balanced_accuracy_score', balanced_accuracy_score(target_list, pred_list), t)
            writer.add_scalar('test/top_k_accuracy_score', top_k_accuracy_score(target_list, pred_score_list), t)    
    
            if t%100 == 0:
                # save model
                if best_test_loss is None or test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(model.state_dict(), 'detection_to_grade_cls.pth')
                    print('Best epoch this far ! Saving weights.')
        print("Done!")
    
    # evaluation
    model.load_state_dict(torch.load('detection_to_grade_cls.pth'))
    model.eval()

    pred_list = []
    pred_score_list = []
    target_list = []
    with torch.no_grad():
        for X, y in tqdm.tqdm(test_loader):
            pred = model(X)
            pred_list += pred.argmax(dim=1).tolist()
            pred_score_list += pred.tolist()
            target_list += y.squeeze().tolist()

    print(balanced_accuracy_score(target_list, pred_list))
    print( confusion_matrix( target_list, pred_list ) )
    print(precision_recall_fscore_support(target_list, pred_list))
    print(top_k_accuracy_score(target_list, pred_score_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--tb-dir', required=True, help='path to tensorboad log: log_tensorboard/detection_to_grade/cls/<tb-dir>')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    main(args)
    
