# Originally Authored by Gaurav Parmar
# Modified for use in VLR Spring 2022 by Murtaza Dalal
import os
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", required=True)
parser.add_argument("--output_folder", required=True)
parser.add_argument("--res", required=True, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    l_fnames = sorted(glob(os.path.join(args.input_folder, "*/*.jpg"), recursive=True))
    os.makedirs(args.output_folder, exist_ok=True)

    for idx, p in enumerate(tqdm(l_fnames)):
        bname = os.path.basename(p)
        outf = os.path.join(args.output_folder, bname)
        Image.open(p).resize((args.res, args.res), Image.BICUBIC).save(outf)
