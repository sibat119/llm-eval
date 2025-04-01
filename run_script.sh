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


python -m src.surrogates.few_shot.get_candidate_response --model_name Qwen/Qwen2.5-7B-Instruct --sub_field high_school_computer_science --batch_size 16 --shot 0
python -m src.surrogates.few_shot.get_candidate_response --model_name Qwen/Qwen2.5-7B-Instruct --sub_field high_school_computer_science --batch_size 16 --shot 3
python -m src.surrogates.few_shot.get_candidate_response --model_name Qwen/Qwen2.5-7B-Instruct --sub_field high_school_computer_science --batch_size 16 --shot 5

python -m src.surrogates.few_shot.get_candidate_response --model_name Qwen/Qwen2.5-7B-Instruct --sub_field philosophy --batch_size 16 --shot 0
python -m src.surrogates.few_shot.get_candidate_response --model_name Qwen/Qwen2.5-7B-Instruct --sub_field philosophy --batch_size 16 --shot 3
python -m src.surrogates.few_shot.get_candidate_response --model_name Qwen/Qwen2.5-7B-Instruct --sub_field philosophy --batch_size 16 --shot 5

python -m src.surrogates.few_shot.get_candidate_response --model_name Qwen/Qwen2.5-7B-Instruct --sub_field public_relations --batch_size 16 --shot 0
python -m src.surrogates.few_shot.get_candidate_response --model_name Qwen/Qwen2.5-7B-Instruct --sub_field public_relations --batch_size 16 --shot 3
python -m src.surrogates.few_shot.get_candidate_response --model_name Qwen/Qwen2.5-7B-Instruct --sub_field public_relations --batch_size 16 --shot 5



python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field high_school_computer_science --batch_size 16 --shot 0
python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field high_school_computer_science --batch_size 16 --shot 3
python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field high_school_computer_science --batch_size 16 --shot 5

python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field philosophy --batch_size 16 --shot 0
python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field philosophy --batch_size 16 --shot 3
python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field philosophy --batch_size 16 --shot 5

python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field public_relations --batch_size 16 --shot 0
python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field public_relations --batch_size 16 --shot 3
python -m src.surrogates.few_shot.get_candidate_response --model_name meta-llama/Llama-3.1-8B-Instruct --sub_field public_relations --batch_size 16 --shot 5