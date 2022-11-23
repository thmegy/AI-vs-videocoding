# AI vs Videocoding

## Compare cracks AI and videocoding

Strict comparison, frame by frame, with `scripts/compare_AI_videocoding_strict.py`.

Measure distance between videocoding and closest AI prediction, then deduce precision and recall:
```
python scripts/compare_AI_videocoding.py --config <config_of_trained_model> --checkpoint <trained_model> --type {cls,det,seg}
```
Use the `--inputvideo` argument to treat a single video. otherwise, the 14 pre-selected videos for comparison are used.  
If you treat a video not part of the 14 pre-selected, you should give the corresponding csv (gps) and json (videocoding) files with the `--csv` and `--json` arguments, respectively.

