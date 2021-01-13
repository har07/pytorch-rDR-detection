import argparse
import csv
import sys
from shutil import rmtree
from PIL import Image
from glob import glob
from os import makedirs, rename
from os.path import join, splitext, basename, exists
from lib.preprocess import resize_and_center_fundus

parser = argparse.ArgumentParser(description='Preprocess FGADR data set.')
parser.add_argument("--data_dir", help="Directory where FGADR resides.",
                    default="data/fgadr")

args = parser.parse_args()
data_dir = str(args.data_dir)

labels = join(data_dir, 'DR_Seg_Grading_Label_Tes.csv')

# Create directories for grades.
[makedirs(join(data_dir, str(i))) for i in [0, 1]
        if not exists(join(data_dir, str(i)))]

# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(data_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

failed_images = []

with open(labels, 'r') as f:
    reader = csv.reader(f, delimiter=',')

    for i, row in enumerate(reader):
        filename, grade = row

        # convert 5-class (international) grade into 2-class (rDR vs non-rDR)
        if int(grade) > 1:
            grade = 1
        else:
            grade = 0

        im_path = join(data_dir, "Original_Images/{}".format(filename))

        # Find contour of eye fundus in image, and scale
        #  diameter of fundus to 299 pixels and crop the edges.
        res = resize_and_center_fundus(save_path=tmp_path,
                                       image_path=im_path,
                                       diameter=299, verbosity=0)

        # Status message.
        msg = "\r- Preprocessing pair of image: {0:>7}".format(i+1)
        sys.stdout.write(msg)
        sys.stdout.flush()

        if res != 1:
            failed_images.append(filename)
            continue


        # Move the files from the tmp folder to the right grade folder.
        rename(join(tmp_path, filename),
                join(data_dir, str(grade), filename))

# Clean tmp folder.
rmtree(tmp_path)

print("Could not preprocess {} images.".format(len(failed_images)))
print(", ".join(failed_images))
