# Brain Tumor Detection using Machine Learning

A professional ML pipeline for classifying brain tumors from MRI scans.

## Setup
```bash
git clone https://github.com/Adam-Yasser/brain-tumor-detection.git
cd brain-tumor-detection
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Data

Download the dataset from [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data) and extract it into `data/raw/`.

**Classes:** glioma, meningioma, pituitary, notumor
**Total images:** 7,200 (5,600 training + 1,600 testing)