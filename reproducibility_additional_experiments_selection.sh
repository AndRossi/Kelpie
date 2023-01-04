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
rm output_*.csv