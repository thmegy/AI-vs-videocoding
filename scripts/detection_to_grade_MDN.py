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
from sklearn.metrics import confusion_matrix
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
        mu = loc[max_mixing_id]
        sigma_al = torch.sqrt( scale[max_mixing_id] )
        sigma_ep = torch.sqrt( ( mu - (loc*mixing).sum() )**2 )

        return mu, sigma_al, sigma_ep, y


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
        loc = torch.sigmoid(output[:,:,0]) * 10
        sigma_max = 10 # arbitrary parameter
        scale = sigma_max * torch.sigmoid(output[:,:,1])
        mixing = torch.exp(output[:,:,2] - output[:,:,2].max()) / torch.exp(output[:,:,2] - output[:,:,2].max()).sum()
        
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


    


def main(args):
    # prepare data
    df = pd.read_csv(args.dataset)
    x_train, x_val, y_train, y_val = train_test_split(df.iloc[:,2:].values, df['NOTE'].values, test_size=0.2, random_state=42)
#    x_train, x_val, y_train, y_val = train_test_split(df['Moy_degradation'].values[:,None], df['NOTE'].values, test_size=0.2, random_state=1) # for fake data
    scaler = StandardScaler()
    x_train = scaler.fit_transform( x_train )
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

    test_loader = val_loader
    
    # predict
    results = trainer.predict(model, test_loader)

    pred_list = []
    sigma_al_list = []
    sigma_ep_list = []
    target_list = []
    for pred, sigma_al, sigma_ep, target in results:
        pred_list += pred.squeeze().tolist()
        sigma_al_list += sigma_al.squeeze().tolist()
        sigma_ep_list += sigma_ep.squeeze().tolist()
        target_list += target.squeeze().tolist()
    
    preds, bins_pred = np.histogram(pred_list, bins=np.linspace(0, 10, 21))
    targets, bins_target = np.histogram(target_list, bins=np.linspace(0, 10, 21))

    print( confusion_matrix( np.digitize(target_list, bins=np.linspace(0, 10, 11)[1:-1]), np.digitize(pred_list, bins=np.linspace(0, 10, 11)[1:-1]) ) )
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

    sigma_al = np.array(sigma_al_list)
    sigma_ep = np.array(sigma_ep_list)
    sigma_tot = np.sqrt(sigma_al**2 + sigma_ep**2)
    
    fig4, ax4 = plt.subplots()
    ax4.errorbar(target_list, pred_list, yerr=sigma_tot, fmt='o', color='red')
    ax4.errorbar(target_list, pred_list, yerr=sigma_al, fmt='o')
    ax4.set_ylabel('Note Prédite')
    ax4.set_xlabel('Note cible')
    fig4.savefig('pred_target_scatter.png')

    fig5, ax5 = plt.subplots()
    ax5.scatter(target_list, sigma_al_list)
    ax5.set_ylabel(r'$\sigma_{al}$')
    ax5.set_xlabel('Note cible')
    fig5.savefig('sigma_al_target_scatter.png')

    fig6, ax6 = plt.subplots()
    ax6.scatter(target_list, sigma_ep_list)
    ax6.set_ylabel(r'$\sigma_{ep}$')
    ax6.set_xlabel('Note cible')
    fig6.savefig('sigma_ep_target_scatter.png')

#    # compute SHAP values
#    batch = next(iter(test_loader))
#    x_test, _ = batch
#
#    explainer = shap.DeepExplainer(model, x_test)
#    shap_values = explainer.shap_values(x_test)
#
#    plt.figure()
#    shap.summary_plot(shap_values, plot_type = 'bar', feature_names=df.columns[2:], show=False)
#    plt.tight_layout()
#    plt.savefig('shap_bar_plot.png')
#
#    plt.figure()
#    shap.summary_plot(shap_values, features=x_test, feature_names=df.columns[2:], show=False)
#    plt.tight_layout()
#    plt.savefig('shap_dot_plot.png')



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--test-set', default='data/detection_to_grade/AI_comparaison_AI4Cracks_notes_200.csv', help='path to test dataset')
    parser.add_argument('--eval', default=None, help='Path to checkpoint. If this argument is used, training is skipped.')
    parser.add_argument('--ncomp', type=int, default=5, help='Number of components in the MDN.')
    args = parser.parse_args()

    main(args)
    
