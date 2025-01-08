import subprocess
import time
import os
import default_args

os.chdir(os.path.dirname(os.path.abspath(__file__)))
delay_seconds = 10

#TODO: run cfw. In this project, it's called sub_ood_class, as it's assigned as a sub class
poison_types = ['sub_ood_class']
poison_rates = ['0.002']

"""Train a clean model first"""
for poison_type, poison_rate in zip(poison_types, poison_rates):
    subprocess.call(["python", 'train_clean_model.py', "-poison_type", poison_type,
                     "-poison_rate", poison_rate])
    time.sleep(delay_seconds)

"""Create CFW dataset"""
for poison_type, poison_rate in zip(poison_types, poison_rates):
    subprocess.call(["python", 'create_cfw_set.py', "-poison_type", poison_type,
                     "-poison_rate", poison_rate])
    time.sleep(delay_seconds)

"""Train CFW"""
for i in range(1):
    for target_class in range(0):
        subprocess.call(["python", 'train_on_cfw.py', '-tri', str(i), '-target_class', str(target_class)])
        time.sleep(delay_seconds)

"""Fine-tune CFW"""
python_file = 'fine_tune_cfw.py'
sufix = '_tr1'
for poison_type, poison_rate in zip(poison_types, poison_rates):
    for i in range(0, 1):
        for target_class in range(1):
            subprocess.call(["python", python_file, '-un_model_sufix', sufix, "-poison_type", poison_type,
                             "-poison_rate", poison_rate,  '-tri', str(i), '-target_class', str(target_class)])
            time.sleep(delay_seconds)

"""Set victim model's name"""
victim_models = []
for i in range(1):
    for target_class in range(1):
        victim_models += [f'model_unlearned_sub_{int(10)}_tri{i}_cls{target_class}{sufix}.pt']

# victim_models = ['full_base_aug_seed=2333.pt'] #TODO: for counterparts

"""Perform model extraction"""
python_file = 'model_extraction_attack.py'
mea_types = ['pb'] #TODO: pb is pool-based active model extraction, i.e., mexmi
for model in victim_models:
        for poison_type, poison_rate in zip(poison_types, poison_rates):
            for mea_type in mea_types:
                subprocess.call(["python", python_file, "-poison_type", poison_type,
                                 "-poison_rate", poison_rate,
                                 "-model", model, '-mea_type', mea_type])
                time.sleep(delay_seconds)

"""Run remove attacks"""
python_file = "other_defense.py" #WMR, NAD, NC
defense_list = ['WRK']
# defense_list = ['NC', 'NAD','I_BAU', 'FP', 'ADV', 'CLP'] #TODO: for existing watermark removal methods
models = [f'extract_pb_{model}' for model in victim_models]
wmr_lr = '0.0001'
for poison_type, poison_rate in zip(poison_types, poison_rates):
    for model in models:
        for defense in defense_list:
            subprocess.call(["python", python_file, "-poison_type", poison_type,
                             "-poison_rate", poison_rate, "-defense", defense,
                             "-model", model, '-wmr_lr', wmr_lr])
            time.sleep(delay_seconds)

"""Test substitute model's performance (before and after wrk)"""
python_file = 'test_acc_fid_w_acc.py'
models += [f'WMR_{model[:-3]}_lr{wmr_lr}.pt' for model in models]
for poison_type, poison_rate in zip(poison_types, poison_rates):
    for model in models:
        subprocess.call(["python", python_file, "-poison_type", poison_type,
                         "-poison_rate", poison_rate,
                         "-model", model, '-victim_model', victim_models[0]])
        time.sleep(delay_seconds)

"""Calculate label clustering; plot predicted label histogram and t-sne"""
python_file = 'plot_shift_output.py'
models = [f'WMR_{model[:-3]}_lr{wmr_lr}.pt' for model in models]
for poison_type, poison_rate in zip(poison_types, poison_rates):
    for model in models:
        subprocess.call(["python", python_file, "-poison_type", poison_type,
                         "-poison_rate", poison_rate,
                         "-model", model])
        time.sleep(delay_seconds)
