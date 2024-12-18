# Project
In this project we are aiming to creating a machine learning pipeline. It is meant to be a proof of concept, on how to create a pipeline for bigger projects, that depends on huge amount of data, that needs to be handled. 

## File Descriptions
main.py: Executes the entire pipeline (currently commented out).
data_preprocessing.py: Prepares and processes Reddit posts for the pipeline.
visualization.py: Provides data analysis and visualization tools.
sentiment.py: Runs sentiment analysis using a DistilBERT model.
finNER_finetuner.py: Fine-tunes the NER model.
finetune.py: Executes the fine-tuning of the NER model.
helper.py: Contains small utility functions used across other files.
mapping.py: Initial attempt to map company names to tickers (abandoned due to computational limitations).
ner_output_processing.py: Processes outputs from the NER model.
record_linking.py: Additional tools for entity mapping.
