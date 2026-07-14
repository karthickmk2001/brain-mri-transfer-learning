# Brain MRI Classification using Transfer Learning (ResNet50)

H9MLAI Machine Learning Lab
Karthick Subramanian Muthukkaruppan | x25191489

---

## What this lab does

Applies transfer learning with a pre-trained ResNet50 to classify brain MRI scans
as `tumor` or `no_tumor`, in two phases:

1. **Feature extraction** - all ResNet50 layers frozen, only a custom head is trained
2. **Fine-tuning** - top 30 ResNet50 layers unfrozen, retrained at a much lower learning rate

Results from both phases are compared and discussed.

**Status: complete.** The notebook has been run end-to-end on the real Kaggle dataset
(not synthetic data) and all outputs, including Task 11's analysis, reflect those real
results. See [Results](#results) below.

---

## Folder structure

```text
Task_09_Executable Code_Brain_MRI_Transfer_Learning/
├── Task_09_Executable Code_Brain_MRI_Transfer_Learning.ipynb   <- the lab notebook (run this)
├── prepare_data.py              <- splits the real Kaggle data into train/test
├── generate_sample_data.py      <- makes fake data to test the pipeline first (optional)
├── requirements.txt
├── README.md
├── venv/                        <- Python virtual environment (create via steps below)
├── brain_tumor_dataset/         <- raw Kaggle download, extracted (source for prepare_data.py)
│   ├── yes/                     <- 155 tumour scans
│   └── no/                      <- 98 healthy scans
└── brain_mri/                   <- train/test split actually read by the notebook
    ├── train/
    │   ├── tumor/                (124 images)
    │   └── no_tumor/             (78 images)
    └── test/
        ├── tumor/                (31 images)
        └── no_tumor/             (20 images)
```

---

## Setup in VS Code

### 1. Open the folder

File -> Open Folder -> select this project folder.

Make sure you have the **Python** and **Jupyter** extensions installed.

### 2. Create a virtual environment

Open the terminal in VS Code (Ctrl + `) and run:

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run this once then retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Mac / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of your terminal prompt.

### 3. Install packages

```bash
pip install -r requirements.txt
```

This takes a few minutes.

---

## Getting the data

The real dataset is already prepared in `brain_mri/` in this project, so you can skip
straight to [Running the notebook](#running-the-notebook). The steps below are here in
case you need to regenerate it (e.g. a different train/test split).

### Option A - test the pipeline first (optional, 30 seconds)

If you want to check everything runs before using the real data:

```bash
python generate_sample_data.py
```

This fills `brain_mri/` with synthetic MRI-like images. Good for confirming the
notebook works. **Do not submit results from this data.** Note it does not clear
the folder first - it adds synthetic images alongside whatever is already there,
so running it now would contaminate the real dataset. If you do run it, delete
the contents of `brain_mri/train/*` and `brain_mri/test/*` and re-run Option B
afterwards to restore the clean real split.

### Option B - the real dataset (required for submission)

1. Download from Kaggle:
   <https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection>

2. Extract the zip into this project folder. You should get a folder containing
   `yes` (tumour scans) and `no` (healthy scans) - already done here as
   `brain_tumor_dataset/yes` and `brain_tumor_dataset/no`.

3. Open `prepare_data.py` and check the `SRC` paths at the top match where you
   extracted the files.

4. Run it:

   ```bash
   python prepare_data.py
   ```

5. It will print the image counts for all four folders. If any show 0, your
   `SRC` paths are wrong - fix them and run again.

---

## Running the notebook

1. Open `Task_09_Executable Code_Brain_MRI_Transfer_Learning.ipynb`
2. Click **Select Kernel** (top right) -> **Python Environments** -> pick the one
   with `venv` in the path
3. Run cells top to bottom (Shift + Enter), or **Run All**

### What each task does

|Task|What happens|Notes|
|-|-|-|
|1|Loads data with ImageDataGenerator|Prints "Found 202 images..." and "Found 51 images..."|
|2|Loads ResNet50 with ImageNet weights|Downloads ~94 MB the first time|
|3|Freezes all base layers|Instant|
|4|Adds custom classifier head|Instant|
|5|Compiles the model|Instant|
|6|**Trains - feature extraction**|The slow one|
|7|Evaluates|Test accuracy: **68.63%**|
|8|**Fine-tuning** - unfreezes top 30 layers, lr=1e-5|Also slow|
|9|Evaluates again|Test accuracy: **62.75%** (fine-tuning made it worse here - see Task 11)|
|10|Plots + confusion matrix + classification report||
|11|Written analysis|Fully updated with the real results and an honest discussion of why fine-tuning underperformed|

---

## Results

|Phase|Trainable Layers|Learning Rate|Test Accuracy|Test Loss|
|-|-|-|-|-|
|Feature Extraction|4 / 179|1e-3|68.63%|0.6055|
|Fine-Tuning|34 / 179|1e-5|62.75%|0.6892|

Fine-tuning **decreased** test accuracy by 5.88 points on this dataset - a real finding
caused by overfitting on a small training set (202 images), discussed in full in Task 11
of the notebook, including a confusion-matrix-based clinical read on tumour recall.

---

## Important: CPU speed

Unless you have an NVIDIA GPU with CUDA configured, TensorFlow runs on CPU and
ResNet50 is heavy. On this dataset size (202 train images), the full 10+10 epoch run
took roughly 10-15 minutes on CPU. On a larger dataset or slower machine it could take
much longer.

**Two ways to deal with this:**

**Reduce epochs.** In Task 6 and Task 8, change `epochs=10` to `epochs=3`.
You will still see the feature-extraction vs fine-tuning difference, which is
the actual point of the lab.

**Or swap in MobileNetV2.** About 10x lighter and trains comfortably on CPU.
Only Task 2 changes:

```python
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

Everything else works unchanged. If you do this, say so in Task 11 - substituting
a lighter backbone for CPU feasibility is a reasonable engineering decision and
is fine as long as you state it openly.

---

## Before you submit

- [x] Replace any synthetic data results with real Kaggle data results
- [x] Update the results table in Task 11 with actual Task 7 and Task 9 accuracies
- [x] Report honestly - fine-tuning made things slightly worse here, and that is
      discussed as a real finding rather than hidden
- [x] All cell outputs are visible in the saved notebook
