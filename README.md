# Weekend Updates Sentiment Analysis

This project implements sentiment analysis on "Weekend Updates" texts using multiple approaches, including custom LSTM models and a fine-tuned BERT model.

## Project Overview

The goal of this project is to classify text data into three sentiment categories:
- Positive
- Negative
- Neutral

Three different models are developed and compared:
1. A basic LSTM model (referred to as "old model")
2. An improved LSTM model (referred to as "new model") 
3. A fine-tuned BERT pre-trained model

## Dataset

The project uses a custom dataset of Weekend Updates texts with manually labeled sentiments. The dataset contains 50 original entries which are augmented using NLP techniques to create a more robust training set.

Dataset statistics after augmentation and balancing:
- Positive samples: 400
- Negative samples: ~370
- Neutral samples: ~290

## Project Structure

The repository contains the following key components:

- `weekend_updates.py`: Main implementation file containing data processing, model training, and evaluation
- `sentiment_analysis_in_lacrime.py`: An earlier version of the implementation
- `weekendUpdates.xlsx`: The original dataset (not included in the repository)

## Methodology

### Data Preprocessing

1. **Text Cleaning**
   - Removal of special characters and emoji conversion
   - Word tokenization
   - Stopword removal
   - Stemming using Porter Stemmer
   - Lemmatization using WordNet

2. **Data Augmentation**
   - Using EasyDataAugmenter to create variations of existing sentences
   - Helps address class imbalance and improves model generalization

3. **Vectorization and Embedding**
   - Text vectorization for converting words to numerical indices
   - GloVe word embeddings (100-dimensional) for semantic representation

### Models

#### LSTM Models
Both custom models use LSTM (Long Short-Term Memory) architecture with the following key differences:

1. **Old Model**:
   - LSTM with dropout of 0.5
   - SpatialDropout1D of 0.5
   - ReLU activation for LSTM
   - 16 LSTM units

2. **New Model**:
   - LSTM with dropout of 0.2
   - Tanh activation for LSTM
   - 16 LSTM units
   - No spatial dropout

#### BERT Model
- Uses the pre-trained `bert-base-uncased` model
- Fine-tuned for the sentiment classification task
- Implements proper tokenization and encoding specific to BERT

## Results

The models are evaluated on a test set with the following metrics:
- Accuracy
- Confusion Matrix

Overall performance comparison:
- Old LSTM Model: ~65% accuracy
- New LSTM Model: ~70% accuracy
- BERT Model: ~80% accuracy

The confusion matrices show that the BERT model performs significantly better, especially on the "Neutral" class which proved challenging for the custom LSTM models.

## Usage

1. **Environment Setup**
   ```
   pip install -q -U locale tensorflow keras nltk emoji torch transformers textattack
   ```

2. **Data Preparation**
   - Place `weekendUpdates.xlsx` in the appropriate location
   - Run the data preprocessing steps in `weekend_updates.py`

3. **Model Training**
   - The script includes model definition and training for all three models
   - Training hyperparameters can be adjusted as needed

4. **Evaluation**
   - The script includes code to generate confusion matrices and accuracy plots
   - An additional test set of new sentences is also evaluated

## Dependencies

- TensorFlow/Keras
- NLTK
- TextAttack
- Transformers (HuggingFace)
- Pandas
- NumPy
- Matplotlib
- Emoji
- PyTorch

## Future Improvements

- Expand the dataset with more diverse examples
- Experiment with other pre-trained models like RoBERTa or XLNet
- Implement more advanced data augmentation techniques
- Explore ensemble methods combining the strengths of different models
- Add cross-validation for more robust evaluation

## License

[Specify your license information here]

## Contributors

[List the contributors to the project]
