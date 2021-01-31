import argparse
import csv
import sys
from shutil import rmtree
from PIL import Image
from glob import glob
from os import makedirs, rename
from os.path import join, splitext, basename, exists
from lib.preprocess import resize_and_center_fundus

parser = argparse.ArgumentParser(description='Preprocess EyePACS data set.')
parser.add_argument("--data_dir", help="Directory where EyePACS resides.",
                    default="data/eyepacs")

args = parser.parse_args()
data_dir = str(args.data_dir)

train_labels = join(data_dir, 'trainLabels.csv')
test_labels = join(data_dir, 'testLabels.csv')
gradability_labels = join(data_dir, 'eyepacs_gradability_grades.csv')

# Create directories for grades.
[makedirs(join(data_dir, str(i))) for i in [0, 1, 2]
        if not exists(join(data_dir, str(i)))]

# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(data_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

failed_images = []

for labels in [train_labels, test_labels]:
    with open(labels, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)

        for i, row in enumerate(reader):
            basename, grade = row[:2]

            # convert 5-class (international) grade into 2-class (rDR vs non-rDR)
            if int(grade) > 1:
                grade = 1
            else:
                grade = 0

            im_path = glob(join(data_dir, "data", "{}*".format(basename)))[0]

            # Find contour of eye fundus in image, and scale
            #  diameter of fundus to 299 pixels and crop the edges.
            res = resize_and_center_fundus(save_path=tmp_path,
                                           image_path=im_path,
                                           diameter=299, verbosity=0)

            # Status message.
            msg = "\r- Preprocessing image: {0:>7}".format(i+1)
            sys.stdout.write(msg)
            sys.stdout.flush()

            if res != 1:
                failed_images.append(basename)
                continue

            new_filename = "{0}.png".format(basename)

            # Move the file from the tmp folder to the right grade folder.
            rename(join(tmp_path, new_filename),
                   join(data_dir, str(int(grade)), new_filename))

with open(gradability_labels, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    next(reader)

    for i, row in enumerate(reader):
        basename, grade = row[:2]

        if grade == "1":
            continue

        for grade_dir in [join(data_dir, str(i)) for i in [0, 1]]:
            # TODO: 
            # if file not exist: continue
            # else: move file to ungradable (grade=2)
            new_filename = "{0}.png".format(basename)
            if not exists(join(grade_dir, new_filename)):
                continue
            rename(join(grade_dir, new_filename),
                   join(data_dir, "2", new_filename))

# Clean tmp folder.
rmtree(tmp_path)

print("Could not preprocess {} images.".format(len(failed_images)))
print(", ".join(failed_images))
