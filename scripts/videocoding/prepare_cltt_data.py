import argparse
import json
import random
from collections import Counter


classes = (
    'Affaissement de rive',
    'Arrachement',
    'Autres réparations',
    'Faïençage',
    'Fissure longitudinale',
    'Fissure transversale',
    'Glaçage - Ressuage',
    'Réparation en BB sur découpe',
    'Fissure thermique',
    'Background'
)



def main(args):
    with open(args.original, "rt") as f_in:
        images = json.load(f_in)

    print(f"There are {len(images)} images.")

    ann = {}
    for key, item in images.items():
        item = '_'.join(set(item)).replace(' ', '_')
        if item in ann.keys():
            ann[item] += 1
        else:
            ann[item] = 1

    for k,i in ann.items():
        if i > 100:
            print(k,i)
    #print("")
    #count_per_category = {c:count_per_category[c] for c in classes}
    #print(min(count_per_category.values()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Original dataset (not split)")
    args = parser.parse_args()

    main(args)
