# Class Feature Watermark (CFW)

## Overview
Machine learning models are valuable intellectual property (IP), but they are vulnerable to **Model Extraction Attacks (MEA)**. In these attacks, adversaries replicate models by exploiting black-box inference APIs, posing a serious threat to model ownership. Model watermarking has emerged as a forensic solution to verify model ownership by embedding identifiable markers. However, existing watermarking methods are often insufficiently resilient against removal attacks.

To address this gap, we propose **Class-Feature Watermarks (CFW)**, a novel approach that enhances watermark resilience by leveraging class-level artifacts. Our method creates a non-existent class using out-of-domain samples as the watermark task, making it more resistant to adaptive removal attacks such as **Watermark Removal attacK (WRK)**.

Our framework systematically disrupts state-of-the-art watermarking methods and demonstrates superior resilience in substitute models created through MEA. CFW optimizes transferability, stability, and robustness, providing a more reliable solution for model ownership protection.

## Features
- **Adaptive Watermark Removal (WRK)**: Demonstrates the vulnerability of existing watermarking methods.
- **Class-Feature Watermarking (CFW)**: Embeds resilient, task-related features for improved protection.
- **MEA Transferability Optimization**: Ensures the watermark remains effective in extracted models.
- **Domain Utility Preservation**: Maintains the model's utility while enhancing security.

## Project Structure
The project includes several key scripts to execute the entire watermarking and attack framework:
- `train_clean_model.py`: Trains a clean baseline model.
- `create_cfw_set.py`: Generates the CFW dataset.
- `train_on_cfw.py`: Trains models using the generated CFW dataset.
- `fine_tune_cfw.py`: Fine-tunes the CFW models.
- `model_extraction_attack.py`: Simulates MEA to test watermark transferability.
- `other_defense.py`: Implements WRK and other removal attacks.
- `test_acc_fid_w_acc.py`: Evaluates substitute models before and after WRK.
- `plot_shift_output.py`: Visualizes label clustering and t-SNE plots.

## How to Run
This project can be executed using the `run.py` file, which automates the training, watermarking, and attack processes.

### Step 1: Train a Clean Model
```bash
python train_clean_model.py -poison_type sub_ood_class -poison_rate 0.002
```

### Step 2: Create the CFW Dataset
```bash
python create_cfw_set.py -poison_type sub_ood_class -poison_rate 0.002
```

### Step 3: Train on CFW
```bash
python train_on_cfw.py -tri 0 -target_class 0
```

### Step 4: Fine-Tune CFW
```bash
python fine_tune_cfw.py -un_model_sufix _tr1 -poison_type sub_ood_class -poison_rate 0.002 -tri 0 -target_class 0
```

### Step 5: Perform Model Extraction
```bash
python model_extraction_attack.py -poison_type sub_ood_class -poison_rate 0.002 -model model_unlearned_sub_10_tri0_cls0_tr1.pt -mea_type pb
```

### Step 6: Run Removal Attacks
```bash
python other_defense.py -poison_type sub_ood_class -poison_rate 0.002 -defense WRK -model extract_pb_model_unlearned_sub_10_tri0_cls0_tr1.pt -wmr_lr 0.0001
```

### Step 7: Test Substitute Model's Performance
```bash
python test_acc_fid_w_acc.py -poison_type sub_ood_class -poison_rate 0.002 -model WMR_extract_pb_model_unlearned_sub_10_tri0_cls0_tr1_lr0.0001.pt -victim_model model_unlearned_sub_10_tri0_cls0_tr1.pt
```

### Step 8: Plot Label Clustering and t-SNE
```bash
python plot_shift_output.py -poison_type sub_ood_class -poison_rate 0.002 -model WMR_extract_pb_model_unlearned_sub_10_tri0_cls0_tr1_lr0.0001.pt
```

## Dependencies
Ensure the following packages are installed:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn

## Results
Our experimental results demonstrate that:
- WRK effectively disrupts existing watermarking approaches.
- CFW significantly improves MEA transferability and resilience.
- CFW preserves domain utility and withstands various removal threats.

