echo "***WARNING: we heavily discourage running this script, as in our estimates replicating all of our additional experiments might take more than 2 months of uninterrupted computation time to complete. We suggest reproducing only a selection of our experiment by running the reproducibility_run_additional_experiments_selection.sh script instead.***" && \

##################################################
### NECESSARY ACCEPTANCE THRESHOLD EXPERIMENTS ###
##################################################

# Kelpie Necessary ComplEx FB15k; Relevance Threshold 1
python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_fb15k_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx FB15k; Relevance Threshold 10
python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_fb15k_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18; Relevance Threshold 1
python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_wn18_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18; Relevance Threshold 10
python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_wn18_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx FB15k-237; Relevance Threshold 1
python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_fb15k237_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx FB15k-237; Relevance Threshold 10
python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_fb15k237_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18RR; Relevance Threshold 1
python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_wn18rr_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18RR; Relevance Threshold 10
python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_wn18rr_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx YAGO3-10; Relevance Threshold 1
python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_yago310_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx YAGO3-10; Relevance Threshold 10
python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_complex_yago310_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE FB15k; Relevance Threshold 1
python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_nec.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/conve/verify_explanations.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_fb15k_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE FB15k; Relevance Threshold 10
python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_nec.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/conve/verify_explanations.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_fb15k_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE WN18; Relevance Threshold 1
python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/conve/verify_explanations.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_wn18_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE WN18; Relevance Threshold 10
python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/conve/verify_explanations.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_wn18_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE FB15k-237; Relevance Threshold 1
python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode necessary --relevance_threshold 1 && \
python scripts/conve/verify_explanations.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_fb15k237_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE FB15k-237; Relevance Threshold 10
python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode necessary --relevance_threshold 10 && \
python scripts/conve/verify_explanations.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_fb15k237_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE WN18RR; Relevance Threshold 1
python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/conve/verify_explanations.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_wn18rr_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE WN18RR; Relevance Threshold 10
python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/conve/verify_explanations.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_wn18rr_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE YAGO3-10; Relevance Threshold 1
python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/conve/verify_explanations.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_yago310_1.csv && \
rm output_*.csv && \

# Kelpie Necessary ConvE YAGO3-10; Relevance Threshold 10
python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/conve/verify_explanations.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_conve_yago310_10.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE FB15k; Relevance Threshold 1
python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/transe/verify_explanations.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_fb15k_1.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE FB15k; Relevance Threshold 10
python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/transe/verify_explanations.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_fb15k_10.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE WN18; Relevance Threshold 1
python3 scripts/transe/explain --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.01  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode necessary --relevance_threshold 1 && \
python3 end_to_end_testing/transe/verify_explanations.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_wn18_1.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE WN18; Relevance Threshold 10
python3 scripts/transe/explain --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.01  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode necessary --relevance_threshold 10 && \
python3 end_to_end_testing/transe/verify_explanations.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_wn18_10.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE FB15k-237; Relevance Threshold 1
python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode necessary --relevance_threshold 1 && \
python3 scripts/transe/verify_explanations.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_fb15k237_1.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE FB15k-237; Relevance Threshold 10
python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode necessary --relevance_threshold 10 && \
python3 scripts/transe/verify_explanations.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_fb15k237_10.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE WN18RR; Relevance Threshold 1
python3 scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode necessary --relevance_threshold 1 && \
python scripts/transe/verify_explanations.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_wn18rr_1.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE WN18RR; Relevance Threshold 10
python3 scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode necessary --relevance_threshold 10 && \
python scripts/transe/verify_explanations.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_wn18rr_10.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE YAGO3-10; Relevance Threshold 1
python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode necessary --relevance_threshold 1 && \
python3 scripts/transe/verify_explanations.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weigh 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_yago310_1.csv && \
rm output_*.csv && \

# Kelpie Necessary TransE YAGO3-10; Relevance Threshold 10
python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode necessary --relevance_threshold 10 && \
python3 scripts/transe/verify_explanations.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weigh 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --mode necessary && \
mv output_end_to_end.csv scripts/experiments/necessary_xsi_threshold/kelpie_transe_yago310_10.csv && \
rm output_*.csv && \

#######################################
### PREFILTER THRESHOLD EXPERIMENTS ###
#######################################

# Kelpie Necessary ComplEx FB15k: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_fb15k_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx FB15k: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_fb15k_30.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx FB15k: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_fb15k_10.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx FB15k: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_fb15k_30.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_wn18_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_wn18_30.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx WN18: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_wn18_10.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx WN18: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_wn18_30.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx FB15k-237: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_fb15k237_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx FB15k-237: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_fb15k237_30.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx FB15k-237: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_fb15k237_10.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx FB15k-237: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_fb15k237_30.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18RR: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_wn18rr_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18RR: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_wn18rr_30.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx WN18RR: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_wn18rr_10.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx WN18RR: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_wn18rr_30.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx YAGO3-10: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_yago310_10.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx YAGO3-10: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_necessary_complex_yago310_30.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx YAGO3-10: Prefilter Threshold 10
python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient --prefilter_threshold 10 && \
python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_yago310_10.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx YAGO3-10: Prefilter Threshold 30
python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient --prefilter_threshold 30 && \
python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_threshold/kelpie_sufficient_complex_yago310_30.csv && \
rm output_*.csv && \

##################################
### PREFILTER TYPE EXPERIMENTS ###
##################################

# Kelpie Necessary ComplEx FB15k: Typebased Prefilter
python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_necessary_complex_fb15k_typebased_prefilter.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx FB15k: Typebased Prefilter
python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_sufficient_complex_fb15k_typebased_prefilter.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18: Typebased Prefilter
python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_necessary_complex_wn18_typebased_prefilter.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx WN18: Typebased Prefilter
python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_sufficient_complex_wn18_typebased_prefilter.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx FB15k-237: Typebased Prefilter
python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_necessary_complex_fb15k237_typebased_prefilter.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx FB15k-237: Typebased Prefilter
python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_sufficient_complex_fb15k237_typebased_prefilter.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx WN18RR: Typebased Prefilter
python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_necessary_complex_wn18rr_typebased_prefilter.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx WN18RR: Typebased Prefilter
python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_sufficient_complex_wn18rr_typebased_prefilter.csv && \
rm output_*.csv && \

# Kelpie Necessary ComplEx YAGO3-10: Typebased Prefilter
python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode necessary && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_necessary_complex_yago310_typebased_prefilter.csv && \
rm output_*.csv && \

# Kelpie Sufficient ComplEx YAGO3-10: Typebased Prefilter
python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient --prefilter type_based && \
python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode sufficient && \
mv output_end_to_end.csv scripts/experiments/prefilter_type/kelpie_sufficient_complex_yago310_typebased_prefilter.csv && \
rm output_*.csv
