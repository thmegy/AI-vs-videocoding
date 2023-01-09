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
    def __init__(self, path, device, augmentation=None):
        df = pd.read_csv(path)

        x = df.iloc[:,2:].values
        y = df['NOTE'].values
        #y /= 10 # grade between 0 and 1
        #y = np.digitize(df['NOTE'], bins=np.linspace(0, 10, 6)[1:-1])

        self.x = torch.tensor(x,dtype=torch.float32, device=device)
        self.y = torch.tensor(y,dtype=torch.float32, device=device)[:, None]
        #self.y = torch.tensor(y, device=device)
        
        self.augmentation = augmentation

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.augmentation is None:
            return self.x[idx],self.y[idx]
        else:
            return self.augmentation(self.x[idx]), self.augmentation(self.y[idx])

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
        self.fcout = nn.Linear(64, 1, device=device)
        
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
        #output = torch.sigmoid(output)
#        output = F.softmax(output, dim=1)
        
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
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
        
    model = GradeFC(device)
    learning_rate = 2e-3
    batch_size = 64
    epochs = 300

    loss_fn = nn.MSELoss()
    #loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#    train_set = CustomDataset(args.train_set, device, augmentation=gaussian_noise)
    train_set = CustomDataset(args.train_set, device)
    test_set = CustomDataset(args.test_set, device)

#    sample_weights = train_set.get_balance_weigths()
#    train_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
#    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    if args.eval:
        model.load_state_dict(torch.load('detection_to_grade.pth'))
    else:
        train_loss_list = []
        test_loss_list = []
        test_epochs = []
        best_test_loss = None
        for t in tqdm.tqdm(range(1, epochs+1)):
            if t%10 == 0:
                print(f"Epoch {t}\n-------------------------------")
                train_loss = train_loop(train_loader, model, loss_fn, optimizer, printout=True)
                test_loss = test_loop(test_loader, model, loss_fn)
                test_loss_list.append(test_loss)
                test_epochs.append(t)
                # save model
                if best_test_loss is None or test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(model.state_dict(), 'detection_to_grade.pth')
                    print('Best epoch this far ! Saving weights.')
            else:
                train_loss = train_loop(train_loader, model, loss_fn, optimizer)
            train_loss_list.append(train_loss)
        print("Done!")
    
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

#    pred_list = []
#    logit_list = []
#    with torch.no_grad():
#        for X, y in tqdm.tqdm(test_loader):
#            pred = model(X)
#            pred_list += pred.argmax(dim=1).tolist()
#
#    print(pred_list)


    
    loss_fn = nn.L1Loss(reduction='none') # get loss for each individual element
    eval_diff = []
    pred_list = []
    target_list = []
    with torch.no_grad():
        for X, y in tqdm.tqdm(test_loader):
            pred = model(X)
            pred_list += (pred.squeeze()).tolist()
            eval_diff += (loss_fn(pred, y).squeeze()).tolist()
            target_list += (y.squeeze()).tolist()

    print(np.array(eval_diff).mean())
    
    print(np.histogram(eval_diff, bins=np.linspace(0, 10, 21)))
    preds, bins_pred = np.histogram(pred_list, bins=np.linspace(0, 10, 21))
    targets, bins_target = np.histogram(target_list, bins=np.linspace(0, 10, 21))

#    print(min(pred_list))
    print(preds)
    print(targets)
    
    bins = np.cumsum(np.diff(bins_pred))
    
    fig2, ax2 = plt.subplots()
    ax2.bar(bins, preds, label='Prédiction', alpha=0.5, width=0.25)
    ax2.bar(bins, targets, label='Target', alpha=0.5, width=0.25)
    ax2.legend()
    ax2.set_xlabel('Note')
    ax2.set_ylabel('#Tronçons')
    fig2.savefig('pred_target_distribution.png')

    matrix, _, _ = np.histogram2d(pred_list, target_list, bins=(np.linspace(0, 10, 11),np.linspace(0, 10, 11)))
    matrix = matrix / matrix.sum(axis=0)

    fig3, ax3 = plt.subplots()
    c = ax3.pcolor(matrix, cmap='Greens')
    ax3.set_ylabel('Note Prédite')
    ax3.set_xlabel('Note cible')
    fig3.colorbar(c)
    fig3.savefig('pred_target_2d.png')

    fig4, ax4 = plt.subplots()
    ax4.scatter(target_list, pred_list)
    ax4.set_ylabel('Note Prédite')
    ax4.set_xlabel('Note cible')
    fig4.savefig('pred_target_scatter.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-set', required=True)
    parser.add_argument('--test-set', required=True)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    main(args)
    
