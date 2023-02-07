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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter


class CustomDataset(Dataset):
    def __init__(self, x, y, device, augmentation=None, do_lds=False, nbin=20):

        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)[:, None]
        
        self.augmentation = augmentation
        self.do_lds = do_lds
        if do_lds:
            lds_weights_per_bin = self.label_distribution_smoothing(nbin)
            self.lds_weights = torch.tensor([ lds_weights_per_bin[ np.digitize(x, bins=np.linspace(0, 10, nbin)[1:-1]) ] for x in y ], dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.augmentation is not None:
            return self.augmentation(self.x[idx]), self.augmentation(self.y[idx])
        elif self.do_lds:
            return self.x[idx], self.y[idx], self.lds_weights[idx]
        else:
            return self.x[idx], self.y[idx]

    def get_balance_weigths(self):
        class_sample_count = torch.unique(self.y, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = torch.tensor([weight[t] for t in self.y])
        return samples_weight

    def label_distribution_smoothing(self, nbin):
        '''
        See https://arxiv.org/pdf/2102.09554.pdf
        '''
        hist = np.histogram(self.y.tolist(), bins=nbin)[0]

        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
        
        eff_label_dist = convolve1d(hist, weights=lds_kernel_window, mode='constant')
        weight_per_bin = 1 / eff_label_dist

        return weight_per_bin


        
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        # kernel = gaussian(ks)
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window



def gaussian_noise(x):
    return torch.clamp(torch.normal(x, 1), min=0, max=10)


    
class GradeFC(nn.Module):
    def __init__(self, device):
        super(GradeFC, self).__init__()
        self.fc1 = nn.Linear(8, 64, device=device)
        self.fc2 = nn.Linear(64, 126, device=device)
        self.fc3 = nn.Linear(126, 252, device=device)
        self.fcout = nn.Linear(252, 2, device=device)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        
        output = self.fcout(x)
        #output = torch.sigmoid(output)
        mu = torch.sigmoid(output[:,0]) * 10
        sigma = torch.sigmoid(output[:,1])

        return mu, sigma
#        return output



def train_loop(dataloader, model, loss_fn, optimizer, printout=False):
    num_batches = len(dataloader)
    train_loss = 0
    
#    for batch, (X, y, w) in enumerate(dataloader):
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
#        pred = model(X)
#        loss = loss_fn(pred, y).squeeze()

        mu, sigma = model(X)
        loss = loss_fn(mu, y.squeeze(), sigma).squeeze()
        
#        loss *= w # reweight loss according to lds smoothed distribution
        loss = loss.mean()
#        loss *= y.shape[0] # renormalise loss
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
    pred_list = []
    target_list = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)[0]
            test_loss += loss_fn(pred, y.squeeze()).item()
            pred_list += (pred.squeeze()).tolist()
            target_list += (y.squeeze()).tolist()

    test_loss /= num_batches

    target_list = np.digitize(target_list, bins=np.linspace(0, 10, 6)[1:-1])
    pred_list = np.digitize(pred_list, bins=np.linspace(0, 10, 6)[1:-1])
    
    return test_loss, target_list, pred_list



def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    writer = SummaryWriter(f'log_tensorboard/detection_to_grade/reg/{args.tb_dir}')
        
    df = pd.read_csv(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,2:].values, df['NOTE'].values, test_size=0.2, random_state=1)
#    x_train, x_test, y_train, y_test = train_test_split(df['Moy_degradation'].values[:,None], df['NOTE'].values, test_size=0.2, random_state=1) # for fake data
#    scaler = StandardScaler()
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform( x_train )
    x_test = scaler.transform( x_test )

#    train_set = CustomDataset(x_train, y_train, device, augmentation=gaussian_noise)
#    train_set = CustomDataset(x_train, y_train, device, do_lds=True)
    train_set = CustomDataset(x_train, y_train, device)
    test_set = CustomDataset(x_test, y_test, device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
        
    model = GradeFC(device)
    learning_rate = 1e-3
    epochs = 3000

#    train_loss_fn = nn.MSELoss(reduction='none')
    train_loss_fn = nn.GaussianNLLLoss(reduction='none')
    test_loss_fn = nn.L1Loss() # get loss for each individual element

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    
    if not args.eval:
        best_test_metric = None
        for t in tqdm.tqdm(range(1, epochs+1)):
            train_loss = train_loop(train_loader, model, train_loss_fn, optimizer)
            writer.add_scalar('train/loss', train_loss, t)

            test_loss, target_list, pred_list = test_loop(test_loader, model, test_loss_fn)
            writer.add_scalar('test/loss', test_loss, t)
            writer.add_scalar('test/balanced_accuracy_score', balanced_accuracy_score(target_list, pred_list), t)
            prfs = precision_recall_fscore_support(target_list, pred_list, zero_division=0)
            writer.add_scalar(f'test/average_precision', prfs[0].mean(), t)    
            writer.add_scalar(f'test/average_recall', prfs[1].mean(), t)    
            writer.add_scalar(f'test/average_fscore', prfs[2].mean(), t)

            writer.add_scalar('learning-rate', optimizer.param_groups[0]['lr'], t)
            scheduler.step()

            if t%100 == 0:
                test_metric = prfs[2].mean()
                # save model
                if best_test_metric is None or test_metric > best_test_metric:
                    best_test_metric = test_metric
                    torch.save(model.state_dict(), 'detection_to_grade.pth')
                    print('Best epoch this far ! Saving weights.')
        print("Done!")

    # evaluation
    model.load_state_dict(torch.load('detection_to_grade.pth'))
    model.eval()

    loss_fn = nn.L1Loss(reduction='none') # get loss for each individual element

    eval_diff = []
    pred_list = []
    sigma_list = []
    target_list = []
    with torch.no_grad():
        for X, y in tqdm.tqdm(test_loader):
            pred, sigma = model(X)
            pred_list += (pred.squeeze()).tolist()
            eval_diff += (loss_fn(pred, y).squeeze()).tolist()
            target_list += (y.squeeze()).tolist()
            sigma_list += (sigma.squeeze()).tolist()

    print(np.array(eval_diff).mean())
    preds, bins_pred = np.histogram(pred_list, bins=np.linspace(0, 10, 21))
    targets, bins_target = np.histogram(target_list, bins=np.linspace(0, 10, 21))

    print( confusion_matrix( np.digitize(target_list, bins=np.linspace(0, 10, 11)[1:-1]), np.digitize(pred_list, bins=np.linspace(0, 10, 11)[1:-1]) ) )
    print(preds, max(pred_list), min(pred_list))
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

    fig5, ax5 = plt.subplots()
    ax5.scatter(target_list, sigma_list)
    ax5.set_ylabel('sigma prédit')
    ax5.set_xlabel('Note cible')
    fig5.savefig('sigma_target_scatter.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--tb-dir', required=True, help='path to tensorboad log: log_tensorboard/detection_to_grade/cls/<tb-dir>')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    main(args)
    
