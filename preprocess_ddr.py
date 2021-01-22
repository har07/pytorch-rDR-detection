import argparse
import csv
import sys
from shutil import rmtree
from PIL import Image
from glob import glob
from os import makedirs, rename
from os.path import join, splitext, basename, exists
from lib.preprocess import resize_and_center_fundus

parser = argparse.ArgumentParser(description='Preprocess DDR data set.')
parser.add_argument("--data_dir", help="Directory where DDR Grading data resides.",
                    default="data/ddr")

args = parser.parse_args()
data_dir = str(args.data_dir)
output_dir = 'preproc'

data_roles = ['test', 'valid']

for role in data_roles:
    # Create directories for grades.
    # 2 is for ungradable images
    [makedirs(join(data_dir, output_dir, role, str(i))) for i in [0, 1, 2]
            if not exists(join(data_dir, str(i)))]

# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(data_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

for role in data_roles:
    print("Preprocessing {} images...".format(role))
    labels = join(data_dir, role + '.txt')
    failed_images = []
    with open(labels, 'r') as f:
        reader = csv.reader(f, delimiter=' ')

        for i, row in enumerate(reader):
            filename, grade = row

            # convert 6-class (international + ungradable) grade into 2-class (rDR vs non-rDR)
            if int(grade) == 5:
                grade = 2
            elif int(grade) > 1:
                grade = 1
            else:
                grade = 0

            im_path = join(data_dir, "{}/{}".format(role, filename))

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

            new_filename = "{0}.png".format(splitext(basename(im_path))[0])

            # Move the files from the tmp folder to the right grade folder.
            rename(join(tmp_path, new_filename),
                    join(data_dir, output_dir, role, str(grade), new_filename))


    print("Could not preprocess {} of {} images.".format(len(failed_images), role))
    print(", ".join(failed_images))

# Clean tmp folder.
rmtree(tmp_path)