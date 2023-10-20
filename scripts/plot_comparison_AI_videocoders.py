import argparse
import json, hjson
import os

import numpy as np
import tqdm
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



score_thresholds = {
    'Arrachement_pelade' : 0.18,
    'Faiencage' : 0.32,
    'Nid_de_poule' : 0.40,
    'Transversale' : 0.28,
    'Longitudinale' : 0.37,
    'Remblaiement_de_tranchees' : 0.42,
    'Pontage' : 0.46,
    'Autre_reparation' : 0.10,
}



def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data



def plot_class_matrices(matrix, metric_name, videocoders, class_names):
    print(f'\nPlotting {metric_name}...')
    fig = plt.figure(figsize=(30,9))
    subfigs_cbar = fig.subfigures(nrows=1, ncols=2, width_ratios=(10,1))
    subfigs = subfigs_cbar[0].subfigures(nrows=2, ncols=4)
    subfigs = subfigs.flatten()

    for ic, cls in enumerate(class_names):
        opts = {'cmap': 'RdYlGn', 'vmin': 0, 'vmax': +1}
        
        ax = subfigs[ic].subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'width_ratios':(4, 1), 'wspace':0.05, 'height_ratios':(1, 4), 'hspace':0.05})
        
        heatmap = ax[1,0].pcolor(matrix[:,:,ic], **opts)
        for irow in range(matrix[:,:,ic].shape[0]):
            for icol in range(matrix[:,:,ic].shape[1]):
                ax[1,0].text(icol+0.5, irow+0.5, '{:.3f}'.format(matrix[:,:,ic][irow][icol]),
                        ha="center", va="center", color="black")

        ax[1,0].set_yticks(np.arange(0.5, matrix.shape[0], 1))
        ax[1,0].set_yticklabels(videocoders)
        ax[1,0].set_xticks(np.arange(0.5, matrix.shape[0], 1))
        ax[1,0].set_xticklabels(videocoders, rotation=45, ha='right')
        
        if ic%4 == 0:
            ax[1,0].set_ylabel('Compared')
        if ic >= 4:
            ax[1,0].set_xlabel('Reference')

        avg_line = ( (matrix[:,:,ic].sum(axis=1)-1) / (matrix[:,:,ic].shape[1]-1) ).reshape(-1,1)
        ax[1,1].pcolor(avg_line, **opts)
        for irow in range(avg_line.shape[0]):
            for icol in range(avg_line.shape[1]):
                ax[1,1].text(icol+0.5, irow+0.5, '{:.3f}'.format(avg_line[irow][icol]),
                        ha="center", va="center", color="black")

        ax[1,1].set_xticks([0.5])
        ax[1,1].set_xticklabels(['mean'], rotation=45, ha='right')

        avg_col = ( (matrix[:,:,ic].sum(axis=0)-1) / (matrix[:,:,ic].shape[0]-1) ).reshape(1,-1)
        ax[0,0].pcolor(avg_col, **opts)
        for irow in range(avg_col.shape[0]):
            for icol in range(avg_col.shape[1]):
                ax[0,0].text(icol+0.5, irow+0.5, '{:.3f}'.format(avg_col[irow][icol]),
                        ha="center", va="center", color="black")

        ax[0,0].set_yticks([0.5])
        ax[0,0].set_yticklabels(['mean'], rotation=45, ha='right') 
        ax[0,0].set_title(cls)

        ax[0,1].axis('off')
       
#        np.savetxt(f'/home/thmegy/Téléchargements/{metric_name}_{cls}.txt', matrix[:,:,ic],
#                   delimiter=',', header=','.join(videocoders))
    

    ax = subfigs_cbar[1].subplots()
    ax.axis('off')
    cax = subfigs_cbar[1].add_axes([0.4, 0.2, 0.2, 0.7])
    cbar = subfigs_cbar[1].colorbar(heatmap, cax=cax)
    cbar.set_label(metric_name)
    fig.patch.set_facecolor('white')
    fig.savefig(f'plots/{metric_name}_matrix.png')

    
    
def main(args):
    distance_thr = f'{args.distance_thr}m'
    
    f1_matrix = [] # (N_videocoders, N_videocoders, N_classes)
    precision_matrix = [] # (N_videocoders, N_videocoders, N_classes)
    recall_matrix = [] # (N_videocoders, N_videocoders, N_classes)
                          # inversion of reference is equivalent to (precision, recall) --> (recall, precision)

    for iv, videocoder_1 in enumerate(args.videocoders):
        # videocoder_1 is compared to videocoder_2
        f1_list = []
        precision_list = []
        recall_list = []
        for videocoder_2 in args.videocoders:
            # videocoder_2 is the reference
            if videocoder_1 == videocoder_2:
                f1_list_cls = [1 for _ in range(8)] # hard-coded number of class...
                precision_list_cls = [1 for _ in range(8)]
            else:
                f1_list_cls = []
                precision_list_cls = []
                if 'AI' in [videocoder_1, videocoder_2]:
                    vid = videocoder_1 if videocoder_1!='AI' else videocoder_2
                    res_ai = read_json(f'/home/theo/workdir/ai-vs-videocoding/results_comparison/det/{vid}/results.json')
                    for (cls, ai) in res_ai.items():
                        # find available score closest to inputed threshold
                        sthrs = np.array([float(t) for t in ai[distance_thr].keys() if t not in ['ap', 'ar']])
                        closest_id = np.argmin(np.abs(sthrs-score_thresholds[cls]))
                        sthr = f'{sthrs[closest_id]:.3f}'

                        f1_list_cls.append(ai[distance_thr][sthr]['f1_score'])
                        # by default, human videocoders as taken as reference when compared to AI
                        # inversion of reference is equivalent to (precision, recall) --> (recall, precision)
                        if videocoder_1=='AI':
                            precision_list_cls.append(ai[distance_thr][sthr]['precision'])
                        else:
                            precision_list_cls.append(ai[distance_thr][sthr]['recall'])

                else:
                    inv_ref = False # Who is reference in input data ?
                    try:
                        res = read_json(f'/home/theo/workdir/ai-vs-videocoding/results_comparison/videocoders/videocoding_reference_videos/{videocoder_1}_vs_{videocoder_2}/results.json')
                    except:
                        res = read_json(f'/home/theo/workdir/ai-vs-videocoding/results_comparison/videocoders/videocoding_reference_videos/{videocoder_2}_vs_{videocoder_1}/results.json')
                        inv_ref = True

                    for cls, d in res['f1_score'].items():
                        f1_list_cls.append(d[distance_thr])
                    if inv_ref:
                        for cls, d in res['precision'].items():
                            precision_list_cls.append(d[distance_thr])
                    else:
                        for cls, d in res['recall'].items():
                            precision_list_cls.append(d[distance_thr])
            f1_list.append(f1_list_cls)
            precision_list.append(precision_list_cls)

        f1_matrix.append(f1_list)
        precision_matrix.append(precision_list)

    f1_matrix = np.array(f1_matrix)
    precision_matrix = np.array(precision_matrix)


    # plot
    classes = list(res['f1_score'].keys())
    plot_class_matrices(f1_matrix, 'F1_score', args.videocoders, classes)
    plot_class_matrices(precision_matrix, 'precision_recall', args.videocoders, classes)
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videocoders', nargs='*', type=str, default=['1','4a','4b', 'AI'], help='name of videocoders we compare AI to.')
    parser.add_argument("--distance-thr", type=int, default=10,
                        help='distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive.')

    args = parser.parse_args()

    main(args)
    
