python src/finetune/t5_loop.py --model_name t5-small --fold_count 2 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_small/fold_2
python src/finetune/t5_loop.py --model_name t5-small --fold_count 3 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_small/fold_3
python src/finetune/t5_loop.py --model_name t5-small --fold_count 4 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_small/fold_4
python src/finetune/t5_loop.py --model_name t5-small --fold_count 5 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_small/fold_5
python src/finetune/t5_loop.py --model_name t5-large --fold_count 2 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_large/fold_2
python src/finetune/t5_loop.py --model_name t5-large --fold_count 3 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_large/fold_3
python src/finetune/t5_loop.py --model_name t5-large --fold_count 4 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_large/fold_4
python src/finetune/t5_loop.py --model_name t5-large --fold_count 5 --dataset meta-llama/Llama-3.2-3B-evals --batch_size 32 --epochs 3 --learning_rate 5e-5 --save_path ./output/t5_large/fold_5
