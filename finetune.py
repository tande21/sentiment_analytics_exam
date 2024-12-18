"""
file: finetune.py
purpose: Handle the class from finNER_finetuner.py
    to execute and fine tune the model. 
"""
from finNER_finetuner import FiNER_finetune
def main():
    
    # "dslim/distilbert-NER"
    # "ProsusAI/finbert"
    # Specify the model checkpoint
    model_checkpoint = "ProsusAI/finbert"
    training_args = {
            "output_dir": "distilbert-finNER",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "num_train_epochs": 10,
            "weight_decay": 0.01,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",  # Use F1 score for model selection
            "lr_scheduler_type": "linear",  # Linear learning rate decay
            "warmup_ratio": 0.1,  # 10% warmup steps
            "report_to": "none",
            "save_total_limit": 2,  # Keep last 2 model checkpoints
        }
    
    # Initialize the fine-tuner
    fine_tuner = FiNER_finetune(model_checkpoint, **training_args)
    
    try:
        # Load the dataset
        fine_tuner.load_data()
        
        # Process the data
        fine_tuner.process_data()
        
        # Setup the model
        fine_tuner.setup_model()
        
        # Train the model
        fine_tuner.train()
        
    except Exception as e:
        print(f"An error occurred during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
