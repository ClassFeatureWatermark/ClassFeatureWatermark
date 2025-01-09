# Class Feature Watermark (CFW)

## Overview
Machine learning models are valuable intellectual property (IP), but they are vulnerable to **Model Extraction Attacks (MEA)**. In these attacks, adversaries replicate models by exploiting black-box inference APIs, posing a serious threat to model ownership. Model watermarking has emerged as a forensic solution to verify model ownership by embedding identifiable markers. However, existing watermarking methods are often insufficiently resilient against removal attacks. To expose this gap, we propose **Watermark Removal attacK (WRK)**, a systematic framework that adaptively disrupts SOTA watermarks by exploiting their reliance on sample-wise artifacts, which are decoupled from domain tasks.

To address these vulnerabilities, we propose **Class-Feature Watermarks (CFW)**, a novel approach that enhances watermark resilience by leveraging class-level artifacts. Our method creates a non-existent class using out-of-domain samples as the watermark task, making it more resistant to adaptive removal attacks such as **Watermark Removal attacK (WRK)**.

## Features
- **Adaptive Watermark Removal (WRK)**: Demonstrates the vulnerability of existing watermarking methods.
- **Class-Feature Watermarking (CFW)**: Embeds resilient, task-related features for improved protection.
- **MEA Transferability Optimization**: Ensures the watermark remains effective in extracted models.

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
  
Besides, it can also execute SOTA black-box watermarks:
- `train_with_entangled_wateramrk.py`: Train a EWE-watermarked model.
- `train_with_meadefender.py`: Train a Mea-defender-watermarked model.
- `train_with_MBW.py`: Train a MBW-watermarked model.
- `train_on_poisoned_set.py`: Train with backdoor methods, e.g., Blend, BadNet.

## How to Run
This project can be executed using the `run.py` file, which automates the training, watermarking, and attack processes.

### Step 1: Train a Clean Model
```bash
python train_clean_model.py -poison_type sub_ood_class -poison_rate 0.002
```

### Step 2: Create the CFW/EWE/MEA-defender/MBW Dataset

```bash
python create_cfw_set.py -poison_type {method_name} -poison_rate {watermark_rate}
```

Replace `<method_name>` with one of the following options: `CFW`, `entangled_watermark`, `MBW`, `meadefender`, or `Blend`.

### Step 3: Train on CFW/EWE/MEA-defender/MBW
After generating the dataset, run the corresponding training script to obtain watermarked models:
For example:
```bash
python train_on_cfw.py -tri 0 -target_class 0
python train_with_entangled_watermark.py
python train_with_MBW.py
python train_with_meadefender.py
python train_on_poisoned_set.py
```

### Step 4: Fine-Tune CFW
This step is specific for CFW method.
```bash
python fine_tune_cfw.py -un_model_sufix _tr1 -poison_type sub_ood_class -poison_rate 0.002 -tri 0 -target_class 0
```

### Step 5: Perform Model Extraction
```bash
python model_extraction_attack.py -poison_type {method_name} -poison_rate {watermark_rate} -model model_unlearned_sub_10_tri0_cls0_tr1.pt -mea_type pb
```

### Step 6: Run Removal Attacks
```bash
python other_defense.py -poison_type {method_name} -poison_rate {watermark_rate} -defense {removal_method} -model extract_pb_{victim_model_name}.pt -wmr_lr 0.0001
```

Replace `<method_name>` with one of the following options: `WRK`, `NC`, `I-BAU`, `FP`, `CLP`, `NAD`, `ADV`.

### Step 7: Test Substitute Model's Performance
```bash
python test_acc_fid_w_acc.py  -poison_type {method_name} -poison_rate {watermark_rate} -model {removal_method}_extract_pb_{victim_model_name}.pt -victim_model {victim_model_name}.pt
```

### Step 8: Plot Label Clustering and t-SNE
This step is specific for CFW method.
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

## Acknowledgments
This project is built on the open-source code **Fight-Poison-with-Poison** ([GitHub](https://github.com/Unispac/Fight-Poison-With-Poison))

Besides, we benchmark following black-box model watermarks against MEAs: 
- **EWE** ([GitHub](https://github.com/cleverhans-lab/entangled-watermark))
- **MBW** ([GitHub](https://github.com/matbambbang/margin-based-watermarking))
- **MEA-Defender** ([GitHub](https://github.com/lvpeizhuo/MEA-Defender))
- 


