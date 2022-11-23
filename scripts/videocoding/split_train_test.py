import argparse
import json
import random
from collections import Counter


def main(args):
    with open(args.original, "rt") as f_in:
        images = json.load(f_in)

    print(f"There are {len(images)} images.")

    ann = []
    for key, item in images.items():
        ann += item
    count_per_category = Counter(ann)

    print("")
    print(count_per_category)
    
    was_success = False
    for i in range(1000):
        # Selected random images from the dataset to be part of validation
        images_val = {}
        images_train = {}
        for im_name, im_ann in images.items():
            if random.random() < args.ratio:
                images_val[im_name] = im_ann
            else:
                images_train[im_name] = im_ann

        ann = []
        for key, item in images_val.items():
            ann += item
        count_per_category_val = Counter(ann)

        # Redraw random indices if counts aren't balanced per category
        counts_are_balanced = True
        for category_name in count_per_category.keys():
            num_total = count_per_category[category_name]
            num_val = count_per_category_val[category_name]
            current_is_balanced = (
                0.8 * args.ratio * num_total < num_val < 1.2 * args.ratio * num_total
            )
            if not current_is_balanced:
                counts_are_balanced = False
                break
        if not counts_are_balanced:
            continue

        print("")
        print("Splits validation/train/total/ratio_val_total:")
        for category_name in count_per_category.keys():
            num_total = count_per_category[category_name]
            num_val = count_per_category_val[category_name]
            num_train = num_total - num_val
            print(
                f"{category_name} : {num_val} / {num_train} / {num_total} / {num_val / num_total}"
            )

        was_success = True
        break

    # Save
    with open(args.val, "wt") as f_out:
        json.dump(images_val, f_out)
    with open(args.train, "wt") as f_out:
        json.dump(images_train, f_out)


    if not was_success:
        print("Failed to split the dataset with given ratio.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original", required=True, help="Original dataset (not split)"
    )
    parser.add_argument("--train", required=True, help="Train dataset (after split)")
    parser.add_argument("--val", required=True, help="Validation dataset (after split)")
    parser.add_argument("--ratio", type=float, required=True, help="The validation will be ratio * original examples.")
    args = parser.parse_args()

    main(args)
