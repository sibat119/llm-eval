# llm-eval

We need to train a surrogate model (T5) from LLaMA model. 
We need to choose two datasets on same task.
For each dataset we will split the data 50-50 and then train on the first half and test on the second half. 
we will pre-train and finetune T5 model. 
We need to select two dataset such that: one dataset is popular and other dataset is new and not very much used.
Once fixed the dataset do the following: 
1. Split the dataset with k-fold cross validation, k=2,3,4,5
2. start the finetuning script
3. start the pretraining script
4. test the data
5. create training vs. testing plot
Another thing we need to do is look at current works on model evaluation and find any similar works.
TODO:
1. [ ] select two datasets on gender bias.
	1. [ ] popular:
	2. [ ] unknown:
2. [ ] select the version of llama that has been updated before the release of the unknown dataset.
	1. [ ] LLaMA version: 
3. [ ] create the finetuning script
4. [ ] create pretraining script
5. [ ] create evaluation pipeline
6. [ ] create plot generation pipeline.
7. [ ] do a literature survey:
	1. [ ] llm evaluation
	2. [ ] noise propagation