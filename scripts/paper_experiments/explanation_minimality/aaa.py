root = '/Users/andrea/minimality_experiments/'

import os
import shutil
datasets = ['fb15k', 'fb15k237', 'wn18', 'wn18rr', 'yago']
models = ['complex', 'conve', 'transe']

for mode in ['necessary', 'sufficient']:
    for dataset in datasets:
        for model in models:
            src_filepath = os.path.join(root, mode, 'post_training', model, dataset, 'output_end_to_end_skipping_random_facts.csv')
            target_filepath = os.path.join(f'kelpie_{mode}_{model}_{dataset}_sampled.csv')

            if not os.path.isfile(src_filepath):
                continue
            shutil.copy(src_filepath, target_filepath)
