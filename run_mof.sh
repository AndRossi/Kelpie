

dataset=MOF-3000
mode=necessary
output_folder=results/$dataset/$mode
mkdir -p $output_folder
model_path=$output_folder/model.pt
facts_to_explain_path=input_facts/complex_wn18_random.csv
method=conve

# python scripts/complex/explain.py --dataset $dataset --model_path $model_path --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path $facts_to_explain_path --mode $mode
# python scripts/complex/verify_explanations.py --dataset $dataset --model_path $model_path --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode $mode
# mv output_end_to_end.csv $output_folder
# rm output_*.csv


python scripts/conve/explain.py --dataset $dataset --method=$method --model_path $model_path --facts_to_explain_path $facts_to_explain_path --mode $mode
