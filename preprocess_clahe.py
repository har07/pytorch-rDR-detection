import argparse
import cv2
import os

parser = argparse.ArgumentParser(description='Apply CLAHE to data set.')
parser.add_argument("--source_dir", help="Directory where the original data resides.")
parser.add_argument("--result_dir", help="Directory to store the result after applying CLAHE to the source data")

args = parser.parse_args()
source_dir = str(args.source_dir)
result_dir = str(args.result_dir)

def _apply_clahe(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clg = clahe.apply(g)
    preproc = cv2.merge([b,clg,r])
    return preproc

os.makedirs(result_dir)
for class_dir in os.listdir(source_dir):
    os.makedirs(os.path.join(result_dir, class_dir))
    class_dir_src = os.path.join(source_dir, class_dir)
    class_dir_result = os.path.join(result_dir, class_dir)
    for img_name in os.listdir(class_dir_src):
        image_path = os.path.join(class_dir_src, img_name)
        image = cv2.imread(os.path.abspath(image_path), -1)
        result = _apply_clahe(image)
        cv2.imwrite(os.path.join(class_dir_result, img_name), result)