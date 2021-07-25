# Before running this, copy the Kaggle API token file to ~/.kaggle/kaggle.json
# Change the file permissions so that no other users can have read access to it
# chmod 600 ~/.kaggle/kaggle.json

# Commands available

# kaggle competitions {list, files, download, submit, submissions, leaderboard}
# kaggle datasets {list, files, download, create, version, init}
# kaggle kernels {list, init, push, pull, output, status}
# kaggle config {view, set, unset}

echo "Install kaggle package via pip before running this script"
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification
