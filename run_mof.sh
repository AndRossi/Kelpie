

dataset=MOF-3000
mode=necessary
method=ConvE
output_folder=results/$dataset/${method}-${mode}
mkdir -p $output_folder
explain_path=input_facts/complex_wn18_random.csv
model_path=stored_models/"${method}_${dataset}.pt"

# python scripts/complex/explain.py --dataset $dataset --model_path $model_path --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --explain_path $explain_path --mode $mode
# python scripts/complex/verify_explanations.py --dataset $dataset --model_path $model_path --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode $mode
# mv output_end_to_end.csv $output_folder
# rm output_*.csv

python explain.py --dataset $dataset --method=$method --model_path $model_path --explain_path $explain_path --mode $mode \
        --run 111 --sort # > $output_folder/output.log
# mv output.txt $output_folder/