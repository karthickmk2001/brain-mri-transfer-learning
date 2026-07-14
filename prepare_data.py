import os
import shutil
import random

random.seed(42)

# Point these at wherever you extracted the Kaggle download.
# The Kaggle dataset "brain-mri-images-for-brain-tumor-detection"
# gives you two folders called 'yes' and 'no'.
SRC = {
    'tumor':    os.path.join('brain_tumor_dataset', 'yes'),
    'no_tumor': os.path.join('brain_tumor_dataset', 'no')
}

SPLIT = 0.8   # 80% train, 20% test


def main():
    # check source folders exist
    for cls, folder in SRC.items():
        if not os.path.isdir(folder):
            print("ERROR: cannot find folder ->", folder)
            print("Edit the SRC paths at the top of this file to match")
            print("where you extracted the Kaggle zip.")
            return

    # make target folders
    for split in ['train', 'test']:
        for cls in SRC:
            os.makedirs(os.path.join('brain_mri', split, cls), exist_ok=True)

    # copy and split
    for cls, folder in SRC.items():
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)
        cut = int(len(files) * SPLIT)

        for f in files[:cut]:
            shutil.copy(os.path.join(folder, f),
                        os.path.join('brain_mri', 'train', cls, f))

        for f in files[cut:]:
            shutil.copy(os.path.join(folder, f),
                        os.path.join('brain_mri', 'test', cls, f))

    # report
    print("Dataset prepared:")
    print()
    for split in ['train', 'test']:
        for cls in SRC:
            path = os.path.join('brain_mri', split, cls)
            n = len(os.listdir(path))
            print("  {:<6} {:<10} {} images".format(split, cls, n))
    print()
    print("You can now run Transfer_Learning.ipynb")


if __name__ == '__main__':
    main()
