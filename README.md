# Hybrid-Recommendation-System
# Overview

The Yelp Recommendation System is designed to deliver tailored business recommendations to users, leveraging their preferences and historical interactions. The system utilizes a combination of collaborative filtering methods and machine learning algorithms to enhance the accuracy of predictions.

Technologies Used

Programming Languages: Python, Scala
Libraries: PySpark (RDD), XGBoost
Frameworks: Apache Spark
Dataset: Subset of Yelp review data
Implementation

The project involved developing a recommendation engine using Locality Sensitive Hashing (LSH) and collaborative filtering techniques. It was divided into two main components:

Task 1: Jaccard-Based Locality Sensitive Hashing (LSH)

Objective:
Implement LSH with Jaccard similarity on the Yelp dataset to identify similar businesses based on user ratings, focusing on binary ratings.

Methods Used:

Locality Sensitive Hashing (LSH): Implemented in Python using Apache Spark RDD to efficiently find candidate pairs of similar businesses.
Jaccard Similarity: Calculated as the ratio of intersection to union of characteristic sets.
Hash Functions: Developed to create consistent row permutations in the characteristic matrix.
Signature Matrix: Constructed and divided into bands to efficiently identify candidate pairs.
Implementation Steps:

Preprocessed Yelp data, designed and implemented hash functions, created a signature matrix, and calculated Jaccard similarity for candidate pairs. The output was a CSV file of similar business pairs.
Task 2: Recommendation System Development

Objective:
Build various recommendation systems using collaborative filtering and machine learning techniques.

Methods Used:

Collaborative Filtering: Employed to recommend items based on user interactions and preferences.
Pearson Similarity: Used in item-based collaborative filtering to measure rating correlations.
XGBoost Regressor: Trained to predict ratings using dataset features.
Hybrid Approach: Combined collaborative filtering and model-based methods to enhance accuracy.
Implementation Steps:

Calculated Pearson similarity, developed a model-based system with XGBoost, and combined results using a hybrid approach to improve recommendations.
Dataset and Evaluation

Dataset: Yelp review subset with user-business interactions.
Metrics: Precision, recall, and RMSE for performance evaluation.
Conclusion

The Yelp Recommendation System effectively uses collaborative filtering and machine learning for personalized business suggestions. The implementation of a hybrid approach, along with the use of Spark RDD and PySpark, enables efficient processing of large-scale datasets, leading to accurate predictions.

Improvements

To enhance the recommendation system's performance, the following strategies were implemented:

Hyperparameter Tuning: Optimized XGBoostRegressor by exploring various hyperparameter combinations to minimize RMSE.
Feature Enrichment: Incorporated additional features from user and business metadata, improving predictive accuracy and system efficiency.
Error Distribution and Performance

Error Distribution: Detailed analysis provided, showing the distribution of prediction errors.
Execution Time: The system demonstrated efficient processing within the specified execution time.
Overall, these improvements contributed to a more accurate and efficient collaborative filtering model, providing users with high-quality, personalized recommendations.
