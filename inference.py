import argparse
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer

# Create argument parser to handle input and output file paths
parser = argparse.ArgumentParser(description='Perform sentiment analysis inference on text data.')
parser.add_argument('input_file', type=str, help='Path to the input file (test_reviews.csv)')
parser.add_argument('output_file', type=str, help='Path to the output file (test_labels_pred.csv)')
args = parser.parse_args()

# Load the pre-trained sentiment analysis model
model = load_model('model.h5') 

# Read input data from the specified CSV file
input_df = pd.read_csv(args.input_file)

# Define the maximum number of words to tokenize
max_words = 10000

# Initialize a tokenizer and fit it on the input text data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(input_df['text']) 

# Lists to store predicted sentiments and decoded reviews
predicted_sentiments = []
decoded_reviews = []

# Loop through each review in the input data
for review_text in input_df['text']:
    print("\nReview : {}".format(review_text))

    # Tokenize and decode the review text
    chosen_text_data = tokenizer.texts_to_sequences([review_text])
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    decoded_review = ' '.join([reverse_word_index.get(word_index, '?') for word_index in chosen_text_data[0]])
    decoded_reviews.append(decoded_review.replace('?', ''))

    # Tokenize the review for model input and predict sentiment
    x_test = tokenizer.texts_to_matrix([review_text], mode='binary')
    pred = model.predict(x_test)
    example = pred[0][0]
    result = round(example)
    predicted_sentiments.append(result)

    # Print the predicted sentiment (Negative or Positive)
    if result == 0:
        print("\nReview is NEGATIVE")
    else:
        print("\nReview is POSITIVE")

# Create a DataFrame with id, predicted sentiment, and decoded reviews
output_df = pd.DataFrame({'id': input_df['id'], 'sentiment': predicted_sentiments, 'text': decoded_reviews})

# Save the DataFrame to the specified output CSV file
output_df.to_csv(args.output_file, index=False)
