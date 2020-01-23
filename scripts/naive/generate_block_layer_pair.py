import os
import pathlib
import argparse
from argparse import RawTextHelpFormatter

from tlopu.model_utils import pick_model, generate_iterator


def parse_args():
    parser = argparse.ArgumentParser(description="Generates the block-layer pairs for the cut models",
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument("model_name", help='Base model for TL.', type=str)
    parser.add_argument("first_block", help='Index of the first block to cut.', type=int)
    parser.add_argument("last_block", help='Index of the last block to cut.', type=int)
    parser.add_argument('-model_options', help='Options for the removal of specific layers in the architecture.'
                                               'Defaults to full.',
                        choices=['full', 'noavgpool', 'norelu', 'norelu_maxpool'], type=str, default="full")
    parser.add_argument("-save_path", help="Save path for the .txt file containing the block-layer models."
                                           "Defaults to /data/home/luca/", type=str, default="/data/home/luca/")
    args = parser.parse_args()
    return args


def main(args):
    pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)
    model, output_size = pick_model(model_name=args.model_name, model_options=args.model_options, device="cpu")

    print("Generating iterator...")
    block_layer_pairs = generate_iterator(model, args.first_block, args.last_block)

    print("Saving to file...")
    filename = os.path.join(args.save_path, "block_layer_pairs_{}.txt".format(args.model_name))
    file = open(filename, mode='w')

    for block, layer in block_layer_pairs:
        file.write("{}.{}\n".format(block, layer))

    file.close()
    print("Finished!")
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
