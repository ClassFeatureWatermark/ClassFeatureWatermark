'''default arguments for data poisoning setup
'''
parser_choices = {
    'dataset': ['gtsrb', 'cifar10', 'ember', 'imagenet', 'speech_commands'],
    'poison_type': ['badnet', 'blend', 'dynamic', 'clean_label', 'TaCT', 'SIG', 'WaNet', 'ISSBA',
                    'adaptive_blend', 'adaptive_patch', 'none', 'trojan', 'meadefender', 'sslguard', 'entangled_watermark', 'MBW'
                    , 'mu_blindspot', 'ood_class', 'sub_ood_class'],
    'poison_rate': [i / 1000.0 for i in range(0, 500)],
    'cover_rate': [i / 1000.0 for i in range(0, 500)],
}

parser_default = {
    'dataset': 'cifar10',#'cifar20', 'speech_commands',#'Imagenet_21',#speech_commands
    'poison_type': 'sub_ood_class',#'ood_class', #'adaptive_blend',#'entangled_watermark',#'meadefender',#'blend',
    'poison_rate': 0.002,#0.1,#0.00, #0.003,#0.1
    'cover_rate': 0,
    'alpha': 0.2,
    'ood_labels': [0, 1, 2, 3, 4],#[54, 62, 70, 82, 92]
} #blend

seed = 2333 # 999, 999, 666 (1234, 5555, 777)