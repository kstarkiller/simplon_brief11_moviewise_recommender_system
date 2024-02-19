# MOVIEWISE : Movies recommendation system

This project is a movie recommender system written in Python.
Here's the project steps:

1. Data Preprocessing and Cleaning: The data is preprocessed and cleaned to ensure quality and consistency.

2. Train-Test Split: The data is split into 'train' and 'test' sets to train the NMF model.

3. Model Training: The NMF model is trained an result scores are normalized on a scale of 1 to 5, which matches the actual user ratings.

4. Performance Evaluation: A table of various performance and relevance indicators is created to evaluate the model.

5. Logging with ML Flow: The performance table is logged using ML Flow, allowing for easy tracking and comparison of different model versions.

## Libraries Used

The following libraries were used in this project:

- pandas
- numpy
- sklearn
- pymongo
- mlflow