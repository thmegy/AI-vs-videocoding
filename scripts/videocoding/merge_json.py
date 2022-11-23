import json
import argparse
import os
import numpy as np


def main(args):
    # Find jsons
    jsonpaths = []
    for root, dirnames, filenames in os.walk(args.jsondir):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext != ".json":
                continue
            jsonpath = os.path.join(root, filename)
            jsonpaths.append(jsonpath)

    if args.do_split:
        dict_train = {}
        dict_val = {}
        dict_test = {}
        for jp in jsonpaths:
            with open(jp, 'r') as f_in:
                d = json.load(f_in)
            if np.random.binomial(1, 0.65, 1) == 1:  # 0.65 probability to fill train set
                dict_train.update(d)
            else:  # 0.5 probability to fill val and test sets
                if np.random.binomial(1, 0.5, 1) == 1: 
                    dict_val.update(d)
                else:
                    dict_test.update(d)
                    
        with open(f'{args.outputdir}/{args.outputname}'.replace('.json', '_train.json'), 'w') as f_out:
            json.dump(dict_train, f_out)
        with open(f'{args.outputdir}/{args.outputname}'.replace('.json', '_val.json'), 'w') as f_out:
            json.dump(dict_val, f_out)
        with open(f'{args.outputdir}/{args.outputname}'.replace('.json', '_test.json'), 'w') as f_out:
            json.dump(dict_test, f_out)
                    
    else:
        dict_full = {}
        for jp in jsonpaths:
            with open(jp, 'r') as f_in:
                d = json.load(f_in)
            dict_full.update(d)

        with open(f'{args.outputdir}/{args.outputname}', 'w') as f_out:
            json.dump(dict_full, f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsondir", required=True)
    parser.add_argument("--outputdir", default='data')
    parser.add_argument("--outputname", required=True)
    parser.add_argument("--do-split", action='store_true', help='do train/val/test split based on missions (one json input per mission)')    
    args = parser.parse_args()
 
    os.makedirs(args.outputdir, exist_ok=True)
    main(args)
