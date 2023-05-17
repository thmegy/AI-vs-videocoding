import argparse
import json, hjson
import os
import re

import cv2 as cv
import numpy as np
import tqdm
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import krippendorff
import subprocess

from utils import extract_lengths_videocoding, compute_smallest_distances



def compute_precision_recall(distances_compared, distances_ref, threshold):
    tp_compared = (distances_compared < threshold).sum()
    fp = (distances_compared > threshold).sum()
    
    tp_ref = (distances_ref < threshold).sum()
    fn = (distances_ref > threshold).sum()
    
    recall = tp_ref / (tp_ref + fn)
    precision = tp_compared / (tp_compared + fp)

    return precision, recall



def plot_distance_distributions(distances_video_1, distances_video_2, outpath, class_name):
    bins = np.linspace(0, 50, 26)
    
    fig, ax1 = plt.subplots()
    ax1.hist(np.clip(distances_video_1, bins[0], bins[-1]), bins=bins)
    ax1.set_xlabel('distance [m]')
    fig.savefig(f'{outpath}/{class_name}_videocoder_1.png') # videocoder 2 as reference
    plt.close()
                
    fig, ax1 = plt.subplots()
    ax1.hist(np.clip(distances_video_2, bins[0], bins[-1]), bins=bins)
    ax1.set_xlabel('distance [m]')
    fig.savefig(f'{outpath}/{class_name}_videocoder_2.png') # videocoder 1 as reference
    plt.close()

    

def plot_evolution(res, var, var_name, outpath):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xlabel('threshold [m]', fontsize=14)
    ax.set_ylabel(var_name, fontsize=14)

    for cls, var_dict in res[var].items():
        thrs = [t.replace('m', '') for t in var_dict.keys()]
        ax.plot(thrs, var_dict.values(), label=cls)
    
    ax.legend(fontsize=16, ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.savefig(f'{outpath}/{var}.png') # videocoder 1 as reference
    plt.close()


    
def plot_precision_recall(results, outpath):
    fig, ax = plt.subplots() # precision-recall summary plot
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')

    for ic, class_name in enumerate(results['recall'].keys()):
        precision_list = list(results['precision'][class_name].values())
        recall_list = list(results['recall'][class_name].values())
        distance_thrs = list(results['recall'][class_name].keys())

        p = ax.plot(recall_list, precision_list, label=class_name, marker='o')
        col = p[-1].get_color()
        for x, y, dthr in zip(recall_list, precision_list, distance_thrs):
            plt.text(x, y, dthr.replace('m', ''), color=col)

    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig(f'{outpath}/precision_recall.png')
    plt.close('all')


    
def plot_timeline(presence_array, videocoders, class_name, outname):
    '''
    Plot, for a given video, whether a degradation has been annotated or not by each videocoder.
    '''
    cmap_binary = mpl.colors.ListedColormap(['red', 'green'])
    cmap = mpl.colors.ListedColormap(['red', 'orange', 'yellow', 'green', 'blue'])

    fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':(2,1)})
    praw = ax[0].imshow(presence_array, aspect='auto', interpolation='none', cmap=cmap_binary, vmin=0, vmax=2)
    ax[0].set_yticks(np.arange(len(videocoders)))
    ax[0].set_yticklabels(videocoders)
    ax[0].set_ylabel('vidéocodeurs')

    psum = ax[1].imshow(presence_array.sum(axis=0).reshape(1,-1), aspect='auto', interpolation='none', cmap=cmap, vmin=0, vmax=5)
    ax[1].set_yticks([0.])
    ax[1].set_yticklabels([''])
    ax[1].set_xlabel('distance [m]')

    cbar_binary = fig.colorbar(praw, ax=ax[0], ticks=[0.5, 1.5])
    cbar_binary.ax.set_yticklabels(['0','1'])
    cbar_binary.set_label(f'Présence {class_name}')
    cbar = fig.colorbar(psum, ax=ax[1], ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
    cbar.ax.set_yticklabels(['0','1','2','3','4'])
    cbar.set_label(f'Somme')
    
    fig.set_tight_layout(True)
    fig.savefig(outname)
    plt.close('all')


    
def plot_timeline_grouped(presence_array_list, videocoders, class_name, video_list, outname):
    '''
    Plot, for all reference videos, whether a degradation has been annotated or not by each videocoder.
    '''
    cmap_binary = mpl.colors.ListedColormap(['white', 'black'])

    fig = plt.figure(figsize=(34,9))
    subfigs_cbar = fig.subfigures(nrows=1, ncols=2, width_ratios=(10,1))
    subfigs = subfigs_cbar[0].subfigures(nrows=2, ncols=7)
    subfigs = subfigs.flatten()
    
    for iv, (vid, presence_array) in enumerate(zip(video_list, presence_array_list)):
        ax = subfigs[iv].subplots(2, sharex=True, gridspec_kw={'height_ratios':(2,1)})
        praw = ax[0].imshow(presence_array, aspect='auto', interpolation='none', cmap=cmap_binary, vmin=0, vmax=2)
        ax[0].set_yticks(np.arange(len(videocoders)))
        ax[0].set_yticklabels(videocoders)
        ax[0].set_title(vid, fontsize='small')
        ax[0].set_ylabel('vidéocodeurs')

        # plot size of majority. Example for 5 videocoder: if sum={5,0} --> majority=5  ; sum={4,1} --> majority=4  ;   sum={3,2} --> majority=3
        presence_sum = presence_array.sum(axis=0)
        majority = np.abs(presence_sum - len(videocoders)/2) + len(videocoders)/2
        vmin = len(videocoders)/2 if len(videocoders)%2==0 else len(videocoders)//2 + 1
        psum = ax[1].imshow(majority.reshape(1,-1), aspect='auto', interpolation='none', cmap='RdYlGn', vmin=vmin, vmax=len(videocoders))
        ax[1].set_yticks([])
        ax[1].set_yticklabels([])
        ax[1].set_xlabel('distance [m]')

        # add hatches if sum == 0 , i.e. no annotations
        x_hatches = np.where(presence_sum==0)[0]
        x_edges_id = np.where(np.diff(x_hatches)>1)[0]
        x_edges_id = np.sort(np.concatenate([x_edges_id, x_edges_id+1]))
        x_edges = x_hatches[x_edges_id]

        if len(x_hatches>0):
            if len(x_edges>0):
                x_edges = np.insert(x_edges, 0, x_hatches[0])  # add start of first rectangle
                x_edges = np.append(x_edges, x_hatches[-1]) # add end of last rectangle
            else:
                x_edges = np.array([x_hatches[0], x_hatches[-1]]) # full size corresponds to zone with sum==0
            
            x_edges = x_edges.reshape(-1,2)
            
        for x1, x2 in x_edges:
            ax[1].add_patch( mpl.patches.Rectangle((x1,-1), x2-x1+1, 2, hatch='//', fill=False) )


    ax = subfigs_cbar[1].subplots()
    ax.axis('off')
    cax = subfigs_cbar[1].add_axes([0.6, 0.2, 0.2, 0.7])
    cbar = subfigs_cbar[1].colorbar(psum, cax=cax, ticks=np.arange(vmin, len(videocoders)+1, 1))
    cbar.ax.set_yticklabels(np.arange(vmin, len(videocoders)+1, 1).astype('str'))
    cbar.set_label('$N_{accord}$')

    cax_binary = subfigs_cbar[1].add_axes([0.2, 0.2, 0.2, 0.7])
    cbar_binary = subfigs_cbar[1].colorbar(praw, cax=cax_binary, ticks=[0.5, 1.5])
    cbar_binary.ax.set_yticklabels(['0','1'])
    cbar_binary.set_label(f'Présence {class_name}')

    cax_hatch = subfigs_cbar[1].add_axes([0.2, 0.1, 0.8, 0.1])
    cax_hatch.axis('off')
    cax_hatch.add_patch( mpl.patches.Rectangle((0,0.1), 0.1, 0.5, hatch='//', fill=False) )
    cax_hatch.text(0.15, 0.35, 'Aucune Dégradation relevée',
                   ha="left", va="center", color="black")

    fig.savefig(outname)
    plt.close('all')



def plot_competence(matrix, videocoders, classes, outname):
    '''
    Plot matrix of annotator competence for each class of degradation, as estimated with MACE.
    '''    
    fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios':(8,1), 'wspace':0.05}, figsize=(len(classes)*2, len(videocoders)*1.7))
    opts = {'cmap': 'RdYlGn', 'vmin': 0, 'vmax': +1}

    ax[0].pcolor(matrix, **opts)
    for irow in range(matrix.shape[0]):
        for icol in range(matrix.shape[1]):
            ax[0].text(icol+0.5, irow+0.5, '{:.3f}'.format(matrix[irow][icol]),
                       ha="center", va="center", color="black")

    ax[0].set_yticks(np.arange(0.5, matrix.shape[0], 1))
    ax[0].set_yticklabels(videocoders)
    ax[0].set_ylabel('vidéocodeurs')
    ax[0].set_xticks(np.arange(0.5, matrix.shape[1], 1))
    ax[0].set_xticklabels(classes, rotation=45, ha='right')

    # compute mean competence of each videocoder and add to matrix plot
    mean_competence = matrix.mean(axis=1).reshape(-1, 1)
    heatmap = ax[1].pcolor(mean_competence, **opts)
    for irow in range(mean_competence.shape[0]):
        for icol in range(mean_competence.shape[1]):
            ax[1].text(icol+0.5, irow+0.5, '{:.3f}'.format(mean_competence[irow][icol]),
                       ha="center", va="center", color="black")

    ax[1].set_xticks([0.5])
    ax[1].set_xticklabels(['Moyenne'], rotation=45, ha='right')

    fig.colorbar(heatmap, ax=ax.ravel().tolist())

    fig.savefig(outname, bbox_inches='tight')
    plt.close('all')
    
                
        
    
def main(args):
    outpath_base = f'results_comparison/videocoders/{args.videocoding_config.replace(".json", "").split("/")[-1]}'

    # load dicts with classes in json and classes used for comparaison
    cls_config =  hjson.load(open(args.cls_config, 'r'))
    classes_vid = cls_config['classes_vid']
    classes_comp= cls_config['classes_comp']
    
    videocoding_config =  hjson.load(open(args.videocoding_config, 'r'))
    inputpath = videocoding_config['inputpath']
    
    # get length for each videocoder from all videos, and unique combinations of videocoders
    length_video_dict = {}
    combinations = []
    max_distance_list = [] # list of max distance travelled in each video
    video_list = [] # name of videos
    for iv, videocoder in enumerate(args.videocoders):
        json_dict = videocoding_config[f'json_dict_{videocoder}'] # load dict to link predefined videos and their videocoding files
        length_video_list = [] # (N_videos, N_classes, N_degradations)
        for vid, jjson in json_dict.items():
            length_video, max_distance = extract_lengths_videocoding(inputpath+'/'+jjson, inputpath+'/'+vid.replace('mp4', 'csv'),
                                                                     classes_vid, classes_comp, args.process_every_nth_meter,
                                                                     return_max_distance=True)
            length_video_list.append(length_video)
            if iv==0:
                max_distance_list.append(max_distance)
                video_list = list(json_dict.keys())
            
        length_video_dict[videocoder] = length_video_list

        # get combinations
        videocoders_tmp = args.videocoders[:iv] + args.videocoders[iv+1:]
        for videocoder_other in videocoders_tmp:
            comb = {videocoder, videocoder_other}
            if comb not in combinations:
                combinations.append(comb)
                
    # loop over videocoders combinations
    for comb in combinations:
        videocoder_1, videocoder_2 = sorted(list(comb))
        length_video_list_1, length_video_list_2 = length_video_dict[videocoder_1], length_video_dict[videocoder_2]
        print(f'\nRunning comparison for {videocoder_1}_vs_{videocoder_2}')
#        print(f'{"class" : <30}{"F1_score": ^10}')
        outpath = f'{outpath_base}/{videocoder_1}_vs_{videocoder_2}'
        os.makedirs(outpath, exist_ok=True)
    
        results = {'recall':{}, 'precision':{}, 'f1_score':{}}
                   
        # compute distances for all classes          
        # loop over classes
        for ic in range(len(classes_comp)):
            class_name = list(classes_comp.keys())[ic]
            results['recall'][class_name] = {}
            results['precision'][class_name] = {}
            results['f1_score'][class_name] = {}

            distances_video_list_1 = []
            distances_video_list_2 = []

            # loop over videos
            for length_video_1, length_video_2 in zip(length_video_list_1, length_video_list_2):
                lv1 = length_video_1[ic]
                lv2 = length_video_2[ic]

                distances_video_list_1.append( compute_smallest_distances(lv1, lv2) )
                distances_video_list_2.append( compute_smallest_distances(lv2, lv1) )

            distances_video_1 = np.concatenate(distances_video_list_1)
            distances_video_2 = np.concatenate(distances_video_list_2)

            # plot distance distributions
            plot_distance_distributions(distances_video_1, distances_video_2, outpath, class_name)

            # compute precision and recall
            # number of true positives is different when taken from distances from videocoding or from AI prediction, e.g. one videocoding matches with several predictions.
            recall_dict = {}
            precision_dict = {}
            f1_dict = {}
            for dthr in args.threshold:
                precision, recall = compute_precision_recall(distances_video_2, distances_video_1, dthr)
                recall_dict[f'{int(dthr)}m'] = recall
                precision_dict[f'{int(dthr)}m'] = precision
                f1_dict[f'{int(dthr)}m'] = 2*precision*recall / (precision+recall)

            results['recall'][class_name] = recall_dict
            results['precision'][class_name] = precision_dict
            results['f1_score'][class_name] = f1_dict

            distance_thr = '10m' # threshold used to extract single summary figures
#            print(f'{class_name : <30} {results["f1_score"][class_name][distance_thr]:^10.3f}')

            plot_evolution(results, 'precision', 'Precision', outpath)
            plot_evolution(results, 'recall', 'Recall', outpath)
            plot_precision_recall(results, outpath)

            with open(f'{outpath}/results.json', 'w') as fout:
                json.dump(results, fout, indent = 6)


                
    ##### make timeline of agreement between all videocoders and use it to compute agreement metrics and annotator competence #####
    os.makedirs(f'{outpath_base}/MACE_inputs', exist_ok=True)

    assert args.process_every_nth_meter==1, '--process-every-nth-meter needs to be 1m for timeline plots !'
    distance_threshold = 10
    length_video_list_videocoders = [l for l in length_video_dict.values()]

    max_distance_list = [int(m)+1 for m in max_distance_list] # only integers
    
    # loop over classes
    mace_competence_array = []
    k_alpha_list = []
    print('\nKrippendorff\'s alpha')
    for ic in range(len(classes_comp)):
        class_name = list(classes_comp.keys())[ic]
        
        # loop over videos
        presence_array_list = []
        for iv, video in enumerate(video_list):
            presence_array = np.zeros((len(args.videocoders), max_distance_list[iv])) # 1 if degradation present, 0 otherwise
            
            # loop over videocoders
            for ivc, length_video_list in enumerate(length_video_list_videocoders):
                degr_list = np.array(length_video_list[iv][ic]).astype(int)
                # add frames in interval +- distance_threshold around each degradation
                if degr_list.size:
                    degr_list = np.unique(np.concatenate([ np.arange(d-distance_threshold, d+distance_threshold+1) for d in degr_list ]))
                    # keep frames within bounds of video
                    degr_list = degr_list[(degr_list >=0) & (degr_list<max_distance_list[iv])]
                degr_list = np.array(degr_list)
                presence_array[ivc, degr_list] = 1

            #plot_timeline(presence_array, args.videocoders, class_name, f'results_comparison/videocoders/{video.replace(".mp4", "")}_{class_name}_timeline.png')
            presence_array_list.append(presence_array)

        plot_timeline_grouped(presence_array_list, args.videocoders, class_name, video_list, f'{outpath_base}/{class_name}_timeline.png')

        # evaluate annotator competence with MACE and inter-annotator agreement with krippendorff's alpha
        presence_array_tot = np.concatenate(presence_array_list, axis=1)
        k_alpha = krippendorff.alpha(reliability_data=presence_array_tot.astype("str"), level_of_measurement="nominal")
        k_alpha_list.append(k_alpha)
        print(f'{class_name : <30} {k_alpha:^10.3f}')

        np.savetxt(f'{outpath_base}/MACE_inputs/{class_name}.csv', presence_array_tot.T, fmt='%i', delimiter=',')
        with open(f'{outpath_base}/MACE_inputs/{class_name}_count.tsv', 'w') as fprior: # save counts of 0s and 1s to be used as priors in MACE
            idx, count = np.unique(presence_array_tot, return_counts=True)
            for i, c in zip(idx, count):
                fprior.write(f'{int(i)}\t{c}\n')
        subprocess.run(
            f'./MACE/MACE --prefix MACE/results/{class_name} --priors {outpath_base}/MACE_inputs/{class_name}_count.tsv {outpath_base}/MACE_inputs/{class_name}.csv',
#            f'./MACE/MACE --prefix MACE/results/{class_name} {outpath_base}/MACE_inputs/{class_name}.csv',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        mace_competence_array.append(np.loadtxt(f'MACE/results/{class_name}.competence'))

    # print competence of videocoders for each class
    mace_competence_array = np.stack(mace_competence_array).T
    cls_str = ' '.join([f' {c[:15]:^15}' for c in classes_comp.keys()])
    print(f'{"videocoder" : <10} {cls_str} {"Average Competence":^15}')
    for videocoder, comp in zip(args.videocoders, mace_competence_array):
        comp_str = ' '.join([f' {c:^15.2f}' for c in comp])
        print(f'{videocoder : <10} {comp_str} {comp.mean():^15.2f}')

    np.savetxt(f'{outpath_base}/MACE_competence.txt', mace_competence_array, header=f'rows: {", ".join(classes_comp.keys())}\n columns: {", ".join(args.videocoders)}', fmt='%.3f')
    np.savetxt(f'{outpath_base}/k_alpha.txt', k_alpha_list, header=f'rows: {", ".join(classes_comp.keys())}', fmt='%.3f')

    plot_competence(mace_competence_array, args.videocoders, classes_comp.keys(), f'{outpath_base}/MACE_competence.png')

    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videocoders', nargs='*', type=str, default=['1','2','3','4a','4b'], help='name of videocoders we compare AI to.')
    parser.add_argument('--videocoding-config', default='configs/videocoding_reference_videos.json', help='json file with dicts to videocoding files corresponding to predefined videos.')
    parser.add_argument('--cls-config', default='configs/classes_reference_videos.json', help='json file with dicts to link classes from AI and videocoding.')
    
    parser.add_argument('--process-every-nth-meter', type=float, default=1, help='step in meters between processed frames.')
    parser.add_argument("--threshold", type=float, nargs='*', default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                        help='distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive.')
    args = parser.parse_args()

    main(args)
    
