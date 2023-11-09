# question_answering_model

the SQuAD dataset was downloaded from : https://www.kaggle.com/datasets/buildformacarov/squad-20

I started by loading and pre-processing the SQuAD dataset so it would match the input format, then i fine-tuned the distilbert-base-cased model on a crop of the dataset using these hyperparameters during the training process:
    num_train_epochs=3
    gradient_accumulation_steps=2        
    per_device_train_batch_size=32
    per_device_eval_batch_size=32
    warmup_steps=20                
    weight_decay=0.01
