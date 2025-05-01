# python src/finetune/t5_loop.py --model_name t5-base --fold_count 2 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_base/fold_2
# python src/finetune/t5_loop.py --model_name t5-base --fold_count 3 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_base/fold_3
# python src/finetune/t5_loop.py --model_name t5-base --fold_count 4 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_base/fold_4
# python src/finetune/t5_loop.py --model_name t5-base --fold_count 5 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_base/fold_5
# python src/finetune/t5_loop.py --model_name t5-large --fold_count 2 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 8 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_large/fold_2
# python src/finetune/t5_loop.py --model_name t5-large --fold_count 3 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 8 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_large/fold_3
# python src/finetune/t5_loop.py --model_name t5-large --fold_count 4 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 8 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_large/fold_4
# python src/finetune/t5_loop.py --model_name t5-large --fold_count 5 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 8 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_large/fold_5

# python -m src.finetune.t5_loop --model_name t5-base --fold_count 2 --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_base_milu/fold_2 --dataset milu --data_path data/dataset/meta-llama_Llama-3.2-3B-Instruct_milu_results.csv
# python -m src.finetune.t5_loop --model_name t5-base --fold_count 3 --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_base_milu/fold_3 --dataset milu --data_path data/dataset/meta-llama_Llama-3.2-3B-Instruct_milu_results.csv
# python -m src.finetune.t5_loop --model_name t5-base --fold_count 4 --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_base_milu/fold_4 --dataset milu --data_path data/dataset/meta-llama_Llama-3.2-3B-Instruct_milu_results.csv
# python -m src.finetune.t5_loop --model_name t5-base --fold_count 5 --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_base_milu/fold_5 --dataset milu --data_path data/dataset/meta-llama_Llama-3.2-3B-Instruct_milu_results.csv

#!/bin/bash

# Define arrays for variables
sub_fields=("Gender_identity")
# prompt_strategies=("black_box" "persona" "pattern_recognition")
prompt_strategies=("black_box")
# selection_strategies=("select_by_context" "select_by_question" "select_by_both")
selection_strategies=("surrogate_q_gen_bounded")

# Base models
llama="meta-llama/Llama-3.1-8B-Instruct"
qwen="Qwen/Qwen2.5-7B-Instruct"

# Common parameters
# Try batch of 1 and see if the previous question influence the responses.
batch_size=1 
shot=5

# Loop through all combinations
for sub_field in "${sub_fields[@]}"; do
    for prompt_strategy in "${prompt_strategies[@]}"; do
        for selection_strategy in "${selection_strategies[@]}"; do
            
            # python -m src.surrogates.few_shot.few_shot_bbq \
            #     --dataset_name "heegyu/bbq" \
            #     --sub_field "$sub_field" \
            #     --shot $shot \
            #     --selection_strategy "$selection_strategy" \
            #     --prompt_variation "$prompt_strategy" \
            #     --create_prompt

            python -m src.surrogates.few_shot.surrogate_generation \
                --dataset_name heegyu/bbq \
                --sub_field Gender_identity \
                --shot 5 \
                --selection_strategy "$selection_strategy" \
                --surrogate "$qwen" \
                --candidate "$llama" \
                --surrogate_gen \

            # python -m src.surrogates.few_shot.surrogate_generation \
            #     --dataset_name heegyu/bbq \
            #     --sub_field Gender_identity \
            #     --shot 5 \
            #     --selection_strategy "$selection_strategy" \
            #     --surrogate "$qwen" \
            #     --candidate "$llama" \
            #     --candidate_gen \

            # python -m src.surrogates.few_shot.surrogate_generation \
            #     --dataset_name heegyu/bbq \
            #     --sub_field Gender_identity \
            #     --shot 5 \
            #     --selection_strategy "$selection_strategy" \
            #     --candidate "$qwen" \
            #     --surrogate "$llama" \
            #     --surrogate_gen \

            # python -m src.surrogates.few_shot.surrogate_generation \
            #     --dataset_name heegyu/bbq \
            #     --sub_field Gender_identity \
            #     --shot 5 \
            #     --selection_strategy "$selection_strategy" \
            #     --candidate "$qwen" \
            #     --surrogate "$llama" \
            #     --candidate_gen \
            
            
            # python -m src.surrogates.few_shot.few_shot_bbq \
            #     --sub_field "$sub_field" \
            #     --batch_size $batch_size \
            #     --shot $shot \
            #     --surrogate "$qwen" \
            #     --candidate "$llama" \
            #     --selection_strategy "$selection_strategy" \
            #     --prompt_variation "$prompt_strategy" \
            
            # python -m src.surrogates.few_shot.few_shot_bbq \
            #     --sub_field "$sub_field" \
            #     --batch_size $batch_size \
            #     --shot $shot \
            #     --surrogate "$llama" \
            #     --candidate "$qwen" \
            #     --selection_strategy "$selection_strategy" \
            #     --prompt_variation "$prompt_strategy" \
            
        done
    done
done

# python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field high_school_computer_science --batch_size 16 --shot 0
# python -m src.surrogates.few_shot.few_shot --sub_field high_school_computer_science --batch_size 16 --shot 3
# python -m src.surrogates.few_shot.few_shot --sub_field high_school_computer_science --batch_size 16 --shot 5

# python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field philosophy --batch_size 16 --shot 0
# python -m src.surrogates.few_shot.few_shot --sub_field philosophy --batch_size 16 --shot 3
# python -m src.surrogates.few_shot.few_shot --sub_field philosophy --batch_size 16 --shot 5

# python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field public_relations --batch_size 16 --shot 0
# python -m src.surrogates.few_shot.few_shot --sub_field public_relations --batch_size 16 --shot 3
# python -m src.surrogates.few_shot.few_shot --sub_field public_relations --batch_size 16 --shot 5
