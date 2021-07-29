import os
import fnmatch

data_dir = "/home/asheesh/Documents/Github/kaggle-rsna-miccai/data/rsna-miccai-brain-tumor-radiogenomic-classification"

global_min_images = 1000
min_dir_path = ""
for dir_name in ("train", "test"):
    root_path = os.path.join(data_dir, dir_name)
    for sub_id in os.listdir(root_path):
        sub_path = os.path.join(root_path, sub_id)
        min_images = min(
            len(fnmatch.filter(os.listdir(os.path.join(sub_path, "FLAIR")), "*.dcm")),
            len(fnmatch.filter(os.listdir(os.path.join(sub_path, "T1w")), "*.dcm")),
            len(fnmatch.filter(os.listdir(os.path.join(sub_path, "T1wCE")), "*.dcm")),
            len(fnmatch.filter(os.listdir(os.path.join(sub_path, "T2w")), "*.dcm")),
        )
        if min_images < global_min_images:
            min_dir_path = sub_path
            global_min_images = min_images


print(global_min_images)
print(min_dir_path)