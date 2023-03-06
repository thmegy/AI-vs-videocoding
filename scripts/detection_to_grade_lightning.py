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
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train/loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return [optimizer]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = F.l1_loss(y_hat, y)
        self.log("val_loss", val_loss)
        concordance = self.concordance(y_hat.squeeze(), y.squeeze())
        self.log("test/concordance_corr_coef", concordance)
        cosine = self.cosine(y_hat.squeeze(), y.squeeze())
        self.log("test/cosine_similarity", cosine)
        explvar = self.explvar(y_hat.squeeze(), y.squeeze())
        self.log("test/explained_variance", explvar)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x), y


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)[:, None]
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


    
class GradeFC(nn.Module):
    def __init__(self, N_in):
        super(GradeFC, self).__init__()
        self.fc1 = nn.Linear(N_in, 64)
        self.fc2 = nn.Linear(64, 126)
        self.fc3 = nn.Linear(126, 252)
        self.fcout = nn.Linear(252, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        
        output = self.fcout(x)
        output = torch.sigmoid(output) * 10
        
        return output



def main(args):
    # prepare data
    df = pd.read_csv(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,2:].values, df['NOTE'].values, test_size=0.2, random_state=1)
#    x_train, x_test, y_train, y_test = train_test_split(df['Moy_degradation'].values[:,None], df['NOTE'].values, test_size=0.2, random_state=1) # for fake data
    scaler = StandardScaler()
    x_train = scaler.fit_transform( x_train )
    x_test = scaler.transform( x_test )

    train_set = CustomDataset(x_train, y_train)
    test_set = CustomDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))

    # define model and lightning module
    model = LitModel(GradeFC(x_train.shape[1]))

    if not args.eval:
        checkpoint_callback = ModelCheckpoint( save_top_k=2, monitor="val_loss")
        trainer = pl.Trainer(gpus=1, callbacks=[checkpoint_callback])
        trainer.fit(model, train_loader, test_loader)

    # evaluation
    model = LitModel.load_from_checkpoint(checkpoint_callback.best_model_path, model=GradeFC(x_train.shape[1]))
    model.eval()
    results = trainer.predict(model, test_loader)

    pred_list = []
    target_list = []
    for pred, target in results:
        pred_list += pred.squeeze().tolist()
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

    fig4, ax4 = plt.subplots()
    ax4.scatter(target_list, pred_list)
    ax4.set_ylabel('Note Prédite')
    ax4.set_xlabel('Note cible')
    fig4.savefig('pred_target_scatter.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    main(args)
    
