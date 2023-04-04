import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_dd
import shap



class LitModel(pl.LightningModule):
    def __init__(self, model):
        super(LitModel,self).__init__()
        self.model = model
        self.concordance = torchmetrics.ConcordanceCorrCoef()
        self.cosine = torchmetrics.CosineSimilarity()
        self.explvar = torchmetrics.ExplainedVariance()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loc, scale, mixing = self.model(x)
        loss = MDNLoss(loc, scale, mixing, y)
        self.log('train/loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return [optimizer]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loc, scale, mixing = self.model(x)
        val_loss = MDNLoss(loc, scale, mixing, y)
        self.log("val_loss", val_loss)
#        concordance = self.concordance(y_hat.squeeze(), y.squeeze())
#        self.log("test/concordance_corr_coef", concordance)
#        cosine = self.cosine(y_hat.squeeze(), y.squeeze())
#        self.log("test/cosine_similarity", cosine)
#        explvar = self.explvar(y_hat.squeeze(), y.squeeze())
#        self.log("test/explained_variance", explvar)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        loc, scale, mixing = self(x)
        max_mixing_id = mixing.argmax(dim=1)
        max_mixing_id = [[i for i in range(mixing.shape[0])], max_mixing_id.tolist()]

        # define predicted mean and aleatoric and epistemic uncertainties
        mu = torch.clamp(loc[max_mixing_id], max=10)
        sigma_al = torch.sqrt( scale[max_mixing_id] )
        sigma_ep = torch.sqrt( ( mu - (loc*mixing).sum(dim=1) )**2 )

        # compute probability distribution for each data point
        eps = 10**-6
        prob_list = []
        for t in torch.linspace(0,10,41)[:-1]:
            gaussian_prob = (mixing * torch.exp(-((t-loc)**2) / (2*scale**2)) / (scale * np.sqrt(2*np.pi)) + eps).sum(dim=1)
            prob_list.append(gaussian_prob[:,None])
        prob_dist = torch.cat(prob_list, dim=1)
            
        return mu, sigma_al, sigma_ep, y, mixing[max_mixing_id], prob_dist


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)[:, None]
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


    
class GradeFC(nn.Module):
    def __init__(self, N_in, ncomp):
        super(GradeFC, self).__init__()
        self.ncomp = ncomp
        self.fc1 = nn.Linear(N_in, 64)
        self.fc2 = nn.Linear(64, 126)
        self.fc3 = nn.Linear(126, 252)
        self.fcout = nn.Linear(252, ncomp*3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        
        output = self.fcout(x)
        output = output.reshape(x.shape[0], self.ncomp, 3) # rows are MDN components, columns are parameters of components (mu, sigma, pi)
        #loc = torch.sigmoid(output[:,:,0]) * 10
        loc = torch.exp(output[:,:,0])
        
        sigma_max = 10 # arbitrary parameter
        scale = sigma_max * torch.sigmoid(output[:,:,1])
        
        maxes = output[:,:,2].max(dim=1)[0][:,None] # get max of mixings for each data point
        mixing = torch.exp(output[:,:,2] - maxes) / torch.exp(output[:,:,2] - maxes).sum(dim=1)[:,None]
        
        return loc, scale, mixing


def MDNLoss(loc, scale, mixing, target):
    '''
    loc: expectation values of gaussian components (N_batch, N_components).
    scale: standard deviation of gaussian components (N_batch, N_components).
    mixing: mixing parameter of gaussian components (N_batch, N_components).
    target: target regression value
    '''
    eps = 10**-6
    # Clamp for stability
    scale = scale.clone()
    with torch.no_grad():
        scale.clamp_(min=eps)

    # compute gaussian probability for each component of the MDN, and multiply by mixing parameter
    gaussian_prob = mixing * torch.exp(-((target-loc)**2) / (2*scale**2)) / (scale * np.sqrt(2*np.pi)) + eps

    # compute loss per data point
    loss_single = gaussian_prob.sum(dim=1)
    loss_single = torch.log(loss_single)

    # reduce loss
    loss = -loss_single.mean()
    
    return loss



def plot_scatter_2Dhist(title, array_x, xlabel, array_y, ylabel, range_hist):
    # scatter plot
    fig, ax = plt.subplots()
    ax.scatter(array_x, array_y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(f'{title}_scatter.png')

    # 2d histogram plot
    matrix, _, _ = np.histogram2d(array_y, array_x, range=range_hist)
    matrix = matrix / matrix.sum(axis=0)

    fig, ax = plt.subplots()
    c = ax.pcolor(matrix, cmap='Greens')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_ticks(np.linspace(0, 10, 6))
    ax.set_yticklabels([f'{l:.2f}' for l in np.linspace(0, range_hist[0][1], 6)], rotation=45, ha='right')   
    fig.colorbar(c)
    fig.savefig(f'{title}_2Dhist.png')

    


def main(args):
    # prepare data
    df = pd.read_csv(args.dataset)
    x_train, x_val, y_train, y_val = train_test_split(df.iloc[:,2:].values, df['NOTE'].values, test_size=0.2, random_state=42)
#    x_train, x_val, y_train, y_val = train_test_split(df['Moy_degradation'].values[:,None], df['NOTE'].values, test_size=0.2, random_state=1) # for fake data
    scaler = StandardScaler()
    x_train = scaler.fit_transform( x_train )
#    x_train = x_train[(y_train<4)|(y_train>7)]
#    y_train = y_train[(y_train<4)|(y_train>7)]
    x_val = scaler.transform( x_val )

    train_set = CustomDataset(x_train, y_train)
    val_set = CustomDataset(x_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set))

    # define model and lightning module
    model = LitModel( GradeFC(x_train.shape[1], args.ncomp) )

    checkpoint_callback = ModelCheckpoint( save_top_k=2, monitor="val_loss")
    trainer = pl.Trainer(max_epochs=1000, gpus=1, callbacks=[checkpoint_callback])
    if args.eval is None:
        trainer.fit(model, train_loader, val_loader)
        args.eval = checkpoint_callback.best_model_path

    ######################
    ##### EVALUATION #####
    ######################

    # load best checkpoint
    model = LitModel.load_from_checkpoint( args.eval, model=GradeFC(x_train.shape[1], args.ncomp) )
    model.eval()

    # load test data, by default IA_comparaison dataset
#    df = pd.read_csv(args.test_set)
#    x_test, y_test = df.iloc[:,2:].values, df['NOTE'].values
#    x_test = scaler.transform( x_test )

#    test_set = CustomDataset(x_test, y_test)
#    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))

    x_test = x_val
    test_loader = val_loader
    
    # predict
    results = trainer.predict(model, test_loader)

    pred_list = []
    sigma_al_list = []
    sigma_ep_list = []
    target_list = []
    mixing_list = []
    prob_dist_list = []
    for pred, sigma_al, sigma_ep, target, mixing, prob_dist in results:
        pred_list += pred.squeeze().tolist()
        sigma_al_list += sigma_al.squeeze().tolist()
        sigma_ep_list += sigma_ep.squeeze().tolist()
        target_list += target.squeeze().tolist()
        mixing_list += mixing.squeeze().tolist()
        prob_dist_list.append( prob_dist )

    # convert to numpy
    prob_dist = torch.cat(prob_dist_list).numpy()
    target = np.array(target_list)
    pred = np.array(pred_list)
    sigma_al = np.array(sigma_al_list)
    sigma_ep = np.array(sigma_ep_list)
    sigma_tot = np.sqrt(sigma_al**2 + sigma_ep**2)

    preds, bins_pred = np.histogram(pred, bins=np.linspace(0, 10, 21))
    targets, bins_target = np.histogram(target, bins=np.linspace(0, 10, 21))

    print('')
    print( 'Balanced accuracy: ', balanced_accuracy_score( np.digitize(target, bins=np.linspace(0, 10, 11)[1:-1]), np.digitize(pred, bins=np.linspace(0, 10, 11)[1:-1]) ) )
    print('')
    print( confusion_matrix( np.digitize(target, bins=np.linspace(0, 10, 11)[1:-1]), np.digitize(pred, bins=np.linspace(0, 10, 11)[1:-1]) ) )
#    print(min(pred))
    print(preds)
    print(targets)
    
    bins = np.cumsum(np.diff(bins_pred))
    
    fig, ax = plt.subplots()
    ax.bar(bins, preds, label='Prédiction', alpha=0.5, width=0.25)
    ax.bar(bins, targets, label='Target', alpha=0.5, width=0.25)
    ax.legend()
    ax.set_xlabel('Note')
    ax.set_ylabel('#Tronçons')
    fig.savefig('pred_target_distribution.png')

    matrix, _, _ = np.histogram2d(pred, target, bins=(np.linspace(0, 10, 11),np.linspace(0, 10, 11)))
    matrix = matrix / matrix.sum(axis=0)

    fig, ax = plt.subplots()
    c = ax.pcolor(matrix, cmap='Greens')
    ax.set_ylabel('Note Prédite')
    ax.set_xlabel('Note cible')
    fig.colorbar(c)
    fig.savefig('pred_target_2DHist.png')
    
    fig, ax = plt.subplots()
    ax.errorbar(target, pred, yerr=sigma_tot, fmt='none', color='red', label=r'$\sqrt{\sigma_{ep}^{2}+\sigma_{al}^{2}}$')
    ax.errorbar(target, pred, yerr=sigma_ep, fmt='none', label='$\sigma_{ep}$')
    ax.scatter(target, pred, marker='.', color='black')
    ax.set_ylabel('Note Prédite')
    ax.set_xlabel('Note cible')
    ax.legend(loc='upper left')
    fig.set_tight_layout(True)
    fig.savefig('pred_target_scatter.png', dpi=200)

    # epistemic and aleatoric uncertainties vs target
    plot_scatter_2Dhist('sigma_al_target', target, 'Note cible', sigma_al, r'$\sigma_{al}$', [[0, sigma_al.max()],[0, 10]])
    plot_scatter_2Dhist('sigma_ep_target', target, 'Note cible', sigma_ep, r'$\sigma_{ep}$', [[0, sigma_ep.max()],[0, 10]])
    
    # epistemic and aleatoric uncertainties vs difference between target and prediction
    plot_scatter_2Dhist('sigma_al_diff', np.abs(target-pred), '|Note cible - Note prédite|', sigma_al, r'$\sigma_{al}$', [[0, sigma_al.max()],[0, np.abs(target-pred).max()]])
    plot_scatter_2Dhist('sigma_ep_diff', np.abs(target-pred), '|Note cible - Note prédite|', sigma_ep, r'$\sigma_{ep}$', [[0, sigma_ep.max()],[0, np.abs(target-pred).max()]])
    plot_scatter_2Dhist('sigma_tot_diff', np.abs(target-pred), '|Note cible - Note prédite|', sigma_tot, r'$\sigma_{tot}$', [[0, sigma_tot.max()],[0, np.abs(target-pred).max()]])
    
    # predicted probability distribution for each test data point
    sorted_arg = np.argsort(target)
    target_sorted = target[sorted_arg]
    pred_sorted = pred[sorted_arg]
    prob_dist = prob_dist[sorted_arg]

    fig, ax = plt.subplots(figsize=(prob_dist.shape[0]*0.5, 12))
    c = ax.pcolor(prob_dist.T/prob_dist.sum(axis=1), cmap='Greens')
    ax.set_ylabel(r'Note', fontsize=15)
    ax.get_xaxis().set_visible(False)
    ax.scatter((np.linspace(1,len(target),len(target))-0.5), target_sorted*4, color='black', label='Note Cible')
    ax.scatter((np.linspace(1,len(target),len(target))-0.5), pred_sorted*4, color='red', label='Note Prédite')
    ticks = np.linspace(0, 40, 11)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks*10/ticks[-1], fontsize=15)
    ax.legend()
    fig.colorbar(c)
    fig.set_tight_layout(True)
    fig.savefig('prob_dist.png')

    # epistemic uncertainty vs training data density
    # use pca to reduce input space to 3 dimensions, and get binned data density
    pca = PCA()
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    hist3d_train, edges = np.histogramdd(x_train_pca[:,:2], bins=5)
    hist3d_train = hist3d_train / hist3d_train.sum()
    # find to which bin the test data point belong
    x_test_pca = pca.transform(x_test)
    stat, edges, binnumber = binned_statistic_dd(x_test_pca[:,:2], None, 'count', bins=5, expand_binnumbers=True)
    binnumber = binnumber - 1 # let index start at 0, not 1
    density = hist3d_train[tuple(binnumber)] # get density corresponding to test data points

    fig, ax = plt.subplots()
    ax.scatter(density, sigma_ep)
    ax.set_ylabel(r'$\sigma_{ep}$')
    ax.set_xlabel('Training data density')
    fig.savefig('sigma_ep_density_scatter.png')

    
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--test-set', default='data/detection_to_grade/AI_comparaison_AI4Cracks_notes_200.csv', help='path to test dataset')
    parser.add_argument('--eval', default=None, help='Path to checkpoint. If this argument is used, training is skipped.')
    parser.add_argument('--ncomp', type=int, default=5, help='Number of components in the MDN.')
    args = parser.parse_args()

    main(args)
    
