# Project
In this project we are aiming to creating a machine learning pipeline. It is meant to be a proof of concept, on how to create a pipeline for bigger projects, that depends on huge amount of data, that needs to be handled. 

## Folders
- **DATA** - Contains all the data sets used in the code
- **DATA/PLOTS** - Contains plots and visualizations of data processing used in the report
- **DATA/TIMESERIES** - Contains timeseries data for specific stocks
- **Sentiment analysis** - Contins the files for testing each sentiment model. The underlying data it is trained on can also be found in DATA -> stockIt_posts_dataset and is hard coded in the files
- **sentiment analysis/Results** - This folder is used to to test multiple sentiment classifiers, to see what we should use. 

## Data
- **Final dataframe product** - This is the final dataframe product after dataprocessing. The file DATA/COMPLETED_DF_SENTIMENT_PROCESSED_CSV. The csv file can be found inside COMPLETE_DF_SENTIMENT.zip. This is the product which we had "hoped" to work a bit more with. But given the time, only basic evaluation was executed. 

## File Descriptions
- main.py: Executes the entire pipeline (currently commented out).
- data_preprocessing.py: Prepares and processes Reddit posts for the pipeline.
- visualization.py: Provides data analysis and visualization tools.
- sentiment.py: Runs sentiment analysis using a DistilBERT model.
- finNER_finetuner.py: Fine-tunes the NER model.
- finetune.py: Executes the fine-tuning of the NER model.
- helper.py: Contains small utility functions used across other files.
- mapping.py: Initial attempt to map company names to tickers (abandoned due to - computational limitations).
- ner_output_processing.py: Processes outputs from the NER model.
- record_linking.py: Additional tools for entity mapping.
