

dataset=MOF-3000
device=1
# mode=necessary        sufficient
# method=ConvE          TransE
# embedding_model=CompGCN       ""

explain() {
        mode=$1
        embedding_model=$2
        method=$3
        run=$4  # 111
        output_folder=results/$dataset/${method}${embedding_model}-${mode}
        mkdir -p $output_folder
        explain_path=$output_folder/explain.csv
        model_path=stored_models/"${method}${embedding_model}_${dataset}.pt"

        echo $output_folder

        CUDA_VISIBLE_DEVICES=$device python explain.py --dataset $dataset --method=$method \
                --model_path $model_path --explain_path $explain_path --mode $mode \
                --output_folder $output_folder  --run $run --specify_relation --ignore_inverse \
                --embedding_model "$embedding_model" --train_restrain --relation_path \
                --prefilter_threshold 50
                # > $output_folder/$mode.log
                # --relation_path
                # > "results/${mode}_example.log" 
}

explain necessary "" ConvE 011
# explain sufficient "" ConvE 011
# explain necessary CompGCN ConvE 011
# explain sufficient CompGCN ConvE 011