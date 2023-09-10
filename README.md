# Overview

This repository contains code for sentiment analysis on text data, specifically classifying text reviews as either "Negative" or "Positive" sentiments using a binary classification model. The main components of this project include:

`model_test.ipynb`:

A Jupyter Notebook that demonstrates the training and evaluation of the sentiment analysis model.

`inference.py`: 

A Python script for performing sentiment analysis inference on new text data using a pre-trained model.

## `model_test.ipynb`
## Code Report: Sentiment Analysis Model Training and Evaluation

This Jupyter Notebook is designed to walk you through the process of training and evaluating a sentiment analysis model. The key steps in this notebook are as follows:

## Data Preprocessing

1. Load two CSV files, 'reviews.csv' and 'labels.csv', into Pandas DataFrames (df_text and df_labels).
2. Create a mapping to convert sentiment labels ('Negative' -> 0, 'Positive' -> 1).
3. Merge the text data and labels on the 'id' column to create a single DataFrame (merged_df).
4. Split the data into training and testing sets (80% training, 20% testing).
5. Tokenize the text data using the Keras Tokenizer class, with a maximum vocabulary size of 10,000 words. The tokenizer is fitted on the training data.
6. Convert text data into a binary matrix representation for both training and testing sets.

## Model Training and Evaluation

1. Build a binary classification model using Keras.
2. Train the model on the training data.
3. Evaluate the model's performance on the testing data, including metrics such as accuracy, precision, recall, and F1-score.
4. Save the trained model for future use.

## `inference.py`

This Python script provides a convenient way to perform sentiment analysis inference on text data using a pre-trained model. It accepts the following command-line arguments:

`input_file` (path to the input file containing test reviews): Provide the path to the text data you want to analyze.

`output_file` (path to the output file for predicted labels and decoded reviews): Specify the path where you want to save the results of the sentiment analysis.

## Usage 

To use `inference.py`, run the script from the command line with the required arguments as follows:

```bash
python inference.py --input_file input_data.csv --output_file output_results.csv
```

The script will tokenize and preprocess the input data, make sentiment predictions, and save the results to the specified CSV file. You can provide your own input data, and the code will generate predicted sentiment labels and decoded reviews for further analysis.

Feel free to explore and use this code for your sentiment analysis tasks!


