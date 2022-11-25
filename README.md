# AI vs Videocoding

## Compare cracks AI and videocoding

Measure distance between videocoding and closest AI prediction, then deduce precision and recall:
```
python scripts/compare_AI_videocoding.py --config <config_of_trained_model> --checkpoint <trained_model> --type {cls,det,seg}
```
Use the `--inputvideo` argument to treat a single video. otherwise, the 14 pre-selected videos for comparison are used.  
If you treat a video not part of the 14 pre-selected, you should give the corresponding csv (gps) and json (videocoding) files with the `--csv` and `--json` arguments, respectively.

Strict comparison, frame by frame, with `scripts/compare_AI_videocoding_strict.py`.



## Extract images to annotate

The distance between AI prediction and videocoding annotations can be used to find frames with False Positives or False Negatives, i.e. frames with disagreement between AI and videocoding.
These frames can be extracted for annotation with:
```
python scripts/prepare_images_to_annotate.py --inputpath-mp4 <path-to-videos> --inputpath-json <path-to-videocoding-jsons> --config <AI-config> --checkpoint <AI-checkpoint> --cls-config <classes-config>
```

Additional arguments are:
- `--process-every-nth-meter` (default=3m): step in meters between processed frames;
- `--threshold-dist` (default=10m): distance (in meter) between a prediction and a videocoding below which we consider a match as a True Positive;
- `--threshold-score` (default=0.3): Ai predictions are kept if their score are above this threshold;

An Active Learning step is also applied to reduce the number of selected images to annotate. This number is controlled with the `--n-sel-al` argument (default=1000).
In this step, the AI model measures the uncertainty of its predictions for each frames, and the most uncertain frames are selected for annotation. Additionally, the uncertainty of a given frame is weighed according to the predicted/videocoded degradations: classes that have a large impact in the road grade calculation have a larger weight. These weights should be defined in the `--cls-config` config file.

### The config file

The config file given with the `--cls-config` argument should contain the following dictionnaries:
- `classes_vid`: link between videocoding classes and classes used for comparison;
- `classes_AI`: link between AI classes and classes used for comparison;
- `classes_comp`: classes used for comparison and their indices