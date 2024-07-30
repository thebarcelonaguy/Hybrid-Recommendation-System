from pyspark import SparkContext, SparkConf

import sys
import time, math
import numpy as np
import json, csv
import xgboost as xgb


# METHOD DESCRIPTION
# I used a weighted hybrid as my final recommendation system where I gave more weights to model based CF and less weights to item based CF.
# I implemented a comprehensive recommendation system that combines collaborative filtering and XGBoost to predict user ratings for businesses.
# The workflow starts by initializing a Spark context, loading training and test datasets, and creating mappings between users and businesses.
# Collaborative filtering predictions are generated using a series of functions that calculate relationships and similarities between businesses and users, similar to a previous implementation.
# I decided to use the same code as assignment 2 for Collaborative filtering.
# Then I checked the RMSE, it was around 0.987 and I clearly needed to improve my recommendation system.
# So I decided to use model based CF and train it on various features.
# Features are extracted and processed from user, business, and review data.
# User-related features(extracted from user.json) include metrics like useful votes, compliments, and review counts; Check line number 526 for my complete user vector
# business-related features(extracted from business.json) consist of attributes such as review counts, stars, is_open, latitudem longitude. and various characteristics like credit card acceptance and parking options;
# Please refer to line number 84 for my complete business feature vector.
# and review-related features include ratings for stars, usefulness, humor, and coolness. These features, along with averages for both users and businesses, are combined to form training and test datasets.
# An XGBoost model is then trained using these features, with hyperparameters tuned via Optuna and 10-fold cross-validation.
# The resulting XGBoost predictions are then combined with collaborative filtering results to form final predictions.
# To evaluate the system's performance, the RMSE and error distribution are calculated for these combined predictions using a dedicated function.
# This integrated approach captures user preferences, business characteristics, and review sentiments more effectively, resulting in a precise recommendation system.


# Error Distributions:
# >=0 and <1: 102631
# >=1 and <2: 32427
# >=2 and <3: 6144
# >=3 and <4: 841
# >=4 and <inf: 1

# RMSE:
# 0.9764325318478704


def save_to_csv(filepath, data, header=None):
    # Open the specified file in write mode
    with open(filepath, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        if header:
            writer.writerow(header)

        for row in data:
            # Ensure all elements are correctly formatted before writing
            formatted_row = [str(row[0]), str(row[1]), float(row[2])]
            writer.writerow(formatted_row)


def extract_review_features(review_data):

    return (
        review_data.map(json.loads)
        .map(
            lambda r: (
                r["user_id"],
                (r["stars"], r["useful"], r["funny"], r["cool"]),
            )
        )
        .collectAsMap()
    )


def get_business_features(business_data_rdd):
    def extract_attributes(attributes, key, mapping_func=None):
        if not attributes:
            return 0
        value = attributes.get(key, 0)
        return mapping_func(value) if mapping_func else int(value)

    def convert_to_binary(value):
        return 1 if value == "True" else 0

    def map_business_info(business):
        business_info = json.loads(business)
        business_id = business_info.get("business_id")

        attributes = business_info.get("attributes", {})

        feature_vector = (
            business_info.get("review_count", 0),
            business_info.get("stars", 0),
            business_info.get("is_open", 0),
            business_info.get("latitude", 0.0),
            business_info.get("longitude", 0.0),
            extract_attributes(attributes, "RestaurantsPriceRange2"),
            extract_attributes(
                attributes, "BusinessAcceptsCreditCards", convert_to_binary
            ),
            extract_attributes(attributes, "BikeParking", convert_to_binary),
            extract_attributes(attributes, "OutdoorSeating", convert_to_binary),
            extract_attributes(
                attributes, "RestaurantsGoodForGroups", convert_to_binary
            ),
            extract_attributes(attributes, "RestaurantsDelivery", convert_to_binary),
            extract_attributes(attributes, "Caters", convert_to_binary),
            extract_attributes(attributes, "HasTV", convert_to_binary),
            extract_attributes(
                attributes, "RestaurantsReservations", convert_to_binary
            ),
            extract_attributes(
                attributes, "RestaurantsTableService", convert_to_binary
            ),
            extract_attributes(attributes, "ByAppointmentOnly", convert_to_binary),
            extract_attributes(attributes, "RestaurantsTakeOut", convert_to_binary),
            extract_attributes(attributes, "AcceptsInsurance", convert_to_binary),
            extract_attributes(attributes, "WheelchairAccessible", convert_to_binary),
            extract_attributes(attributes, "GoodForKids", convert_to_binary),
        )

        return (business_id, feature_vector)

    return business_data_rdd.map(map_business_info).collectAsMap()


def preprocess_data(record, user_features, business_features, review_features, is_test):
    """
    Processes a given record by combining its features with the corresponding features from user, business, and review datasets.

    Parameters:
        record (list): A list containing [user_id, business_id, rating (if available)].
        user_features (dict): Dictionary mapping user IDs to their features.
        business_features (dict): Dictionary mapping business IDs to their features.
        review_features (dict): Dictionary mapping user IDs to review-based features.
        is_test (bool): Indicates whether the record is from the test dataset.

    Returns:
        list: Combined list of features for the record.
    """

    user_id, business_id = record[0], record[1]
    rating = -1 if is_test else float(record[2]) if len(record) > 2 else None

    if (
        user_id in user_features
        and business_id in business_features
        and user_id in review_features
    ):
        combined = (
            list(user_features[user_id])
            + list(business_features[business_id])
            + list(review_features[user_id])
        )
        if rating is not None:
            combined.append(rating)

        return [user_id, business_id] + combined
    else:
        # Creating the same number of None elements as there are in the features sets plus one for the rating if not test.
        return [user_id, business_id] + [None] * (
            len(user_features[user_id])
            + len(business_features[business_id])
            + len(review_features[user_id])
            + (1 if not is_test else 0)
        )


def create_dataset(data, user_features, business_features, review_features, is_test):
    preprocessed_data = data.map(
        lambda record: preprocess_data(
            record, user_features, business_features, review_features, is_test
        )
    )

    # Cache the preprocessed data to avoid recomputation
    preprocessed_data.cache()

    feature_count = len(preprocessed_data.first())
    x = np.array(
        preprocessed_data.map(lambda record: record[2 : feature_count - 1]).collect(),
        dtype="float",
    )
    y = np.array(
        preprocessed_data.map(lambda record: record[-1]).collect(), dtype="float"
    )

    # Unpersist preprocessed_data after use
    preprocessed_data.unpersist()

    return x, y


def pre_process_data(file_path):

    data_rdd = sc.textFile(file_path)
    header = data_rdd.first()

    return data_rdd.filter(lambda x: x != header).map(lambda x: x.strip().split(","))


def preparing_dataset_dictionary(train_rdd, test_rdd):
    buckets_user_businesss = train_rdd.map(lambda x: x[0])
    buckets_user_businesss_unique = buckets_user_businesss.distinct()
    test_users = test_rdd.map(lambda x: x[0])
    test_users_unique = test_users.distinct()
    all_users = buckets_user_businesss_unique.union(test_users_unique).distinct()
    all_users_index = all_users.zipWithIndex()

    group_businessinesses = train_rdd.map(lambda x: x[1])
    group_businessinesses_unique = group_businessinesses.distinct()

    test_businesses = test_rdd.map(lambda x: x[1])
    test_businesses_unique = test_businesses.distinct()

    all_businesses = group_businessinesses_unique.union(
        test_businesses_unique
    ).distinct()
    all_businesses_index = all_businesses.zipWithIndex()

    dict_all_businesses = all_businesses_index.collectAsMap()

    dict_all_users = all_users_index.collectAsMap()
    return dict_all_businesses, dict_all_users


def get_business_user_rating_map(
    train_rdd, dictionary_all_businesses, dictionary_all_users
):
    transformed_data = train_rdd.map(
        lambda x: (
            dictionary_all_businesses[x[1]],
            (
                dictionary_all_users[x[0]],
                float(x[2]),
            ),
        )
    )

    grouped_data = transformed_data.groupByKey().mapValues(list)

    business_user_rating_map = grouped_data.collectAsMap()

    return business_user_rating_map


def calculate_average_business_ratings(train_rdd, all_bus_dict):

    business_ratings = train_rdd.map(lambda x: (all_bus_dict[x[1]], float(x[2])))

    sum_and_count = business_ratings.mapValues(lambda rating: (rating, 1)).reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )

    average_ratings = sum_and_count.mapValues(lambda x: x[0] / x[1])

    return average_ratings.collectAsMap()


def calculate_average_user_ratings(train_rdd, dictionary_all_users):

    user_ratings = train_rdd.map(lambda x: (dictionary_all_users[x[0]], float(x[2])))

    sum_and_count = user_ratings.mapValues(lambda rating: (rating, 1)).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1])
    )

    average_user_ratings = sum_and_count.mapValues(lambda x: x[0] / x[1])

    return average_user_ratings.collectAsMap()


def map_users_to_businesses_and_ratings(
    train_rdd, dictionary_all_users, dictionary_all_businesses
):

    user_business_ratings = train_rdd.map(
        lambda x: (
            dictionary_all_users[x[0]],
            (
                dictionary_all_businesses[x[1]],
                float(x[2]),
            ),
        )
    )

    grouped_by_user = user_business_ratings.groupByKey().mapValues(list)

    user_to_businesses_and_ratings_map = grouped_by_user.collectAsMap()

    return user_to_businesses_and_ratings_map


import math


# def optimize_xgb_model(x_train, y_train):
#     def objective(trial):
#         # Define hyperparameter search space
#         params = {
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#             "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
#             "max_depth": trial.suggest_int("max_depth", 3, 12),
#             "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#             "gamma": trial.suggest_float("gamma", 0.0, 1.0),
#             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
#             "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
#         }

#         model = xgb.XGBRegressor(**params)

#         # Perform 10-fold cross-validation
#         kf = KFold(n_splits=10, shuffle=True, random_state=42)
#         cv_scores = cross_val_score(
#             model, x_train, y_train, cv=kf, scoring="neg_mean_squared_error"
#         )
#         mean_rmse = np.sqrt(np.abs(np.mean(cv_scores)))

#         return mean_rmse

#     # Create a study object and optimize the objective function
#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=100)

#     best_params = study.best_params

#     # Print the best parameters found
#     print("Best parameters:", best_params)

#     # Train the model with the best parameters found
#     best_model = xgb.XGBRegressor(**best_params)
#     best_model.fit(x_train, y_train)

#     return best_model, best_params


def train_xgb_model(x_train, y_train):
    # The above code gave me the best parameters:
    # Define model hyperparameters. used Optuna to find the best parameters for my XGBOOSTmodel
    params = {
        "learning_rate": 0.02,  # Learning rate for gradient boosting
        "n_estimators": 867,  # Number of boosting rounds
        "max_depth": 9,  # Maximum tree depth
        "min_child_weight": 9,  # Minimum leaf node weight
        "subsample": 0.73,  # Subsample ratio
        "colsample_bytree": 0.66,  # Column sampling ratio per tree
        "gamma": 0.26,  # Minimum loss reduction required for split
        "reg_alpha": 0.83,  # L1 regularization term
        "reg_lambda": 0.43,  # L2 regularization term
    }

    # Train model
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(x_train, y_train)

    return xgb_model


def predict_xgb_model(xgb_model, x_test):
    # Predict using the model
    return xgb_model.predict(x_test)


def calculate_average_rating(ratings):
    """Calculate the average rating from a list of ratings."""
    return sum(ratings) / len(ratings) if ratings else 3.5


def calculate_pearson_correlation(
    common_users, user_ratings, target_user_avg, business_ratings, target_business_avg
):
    """Calculate the Pearson correlation coefficient between two sets of ratings."""
    numerator = sum(
        (user_ratings[user_id] - target_user_avg)
        * (business_ratings[user_id] - target_business_avg)
        for user_id in common_users
    )
    denominator_term1 = sum(
        (user_ratings[user_id] - target_user_avg) ** 2 for user_id in common_users
    )
    denominator_term2 = sum(
        (business_ratings[user_id] - target_business_avg) ** 2
        for user_id in common_users
    )

    if denominator_term1 == 0 or denominator_term2 == 0:
        return 0
    return numerator / (math.sqrt(denominator_term1) * math.sqrt(denominator_term2))


def get_common_users(business1_users, business2_users):
    """Return a list of common users between two businesses."""
    return [user_id for user_id in business1_users if user_id in business2_users]


def calculate_predicted_rating(
    b, u, group_business, buckets_user_business, business_avg_ratings, user_avg_ratings
):
    """Calculate the predicted rating for a user and business based on collaborative filtering."""
    if b not in group_business and u not in buckets_user_business:
        return b, u, 3.75

    if b not in group_business:
        return b, u, user_avg_ratings.get(u, 3.5)

    if u not in buckets_user_business:
        return b, u, business_avg_ratings.get(b, 3.5)

    all_user_ratings = buckets_user_business[u]
    target_business_ratings = {user_id: rating for user_id, rating in group_business[b]}
    target_business_avg = business_avg_ratings.get(b, 3.5)

    similar_businesses = []
    for business_id, user_rating in all_user_ratings:
        if business_id in group_business:
            business_ratings = {
                user_id: rating for user_id, rating in group_business[business_id]
            }
            common_users = get_common_users(
                business_ratings.keys(), target_business_ratings.keys()
            )

            if len(common_users) > 2:
                pearson_correlation = calculate_pearson_correlation(
                    common_users,
                    business_ratings,
                    business_avg_ratings.get(business_id, 3.5),
                    target_business_ratings,
                    target_business_avg,
                )
                similar_businesses.append((business_id, pearson_correlation))
            else:
                # Estimate similarity for businesses with few common users
                estimated_similarity = (
                    5
                    - abs(
                        business_avg_ratings.get(business_id, 3.5) - target_business_avg
                    )
                ) / 5
                similar_businesses.append((business_id, estimated_similarity))

    similar_businesses.sort(key=lambda x: x[1], reverse=True)
    top_similar_businesses = similar_businesses[:10]

    predicted_numerator = 0
    predicted_denominator = 0
    for business_id, weight in top_similar_businesses:
        user_rating_for_business = dict(buckets_user_business[u])[business_id]
        predicted_numerator += weight * user_rating_for_business
        predicted_denominator += abs(weight)

    if predicted_denominator == 0:
        return b, u, 3.5
    predicted_rating = predicted_numerator / predicted_denominator
    return b, u, predicted_rating


if __name__ == "__main__":
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_filepath = sys.argv[3]

    start = time.time()

    conf = SparkConf().setAppName("553Competition")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    train_rdd = pre_process_data(folder_path + "/yelp_train.csv")
    test_rdd = pre_process_data(test_file)
    dictionary_all_businesses, dictionary_all_users = preparing_dataset_dictionary(
        train_rdd, test_rdd
    )
    # group_business1(tarin_bus)
    group_business = get_business_user_rating_map(
        train_rdd, dictionary_all_businesses, dictionary_all_users
    )
    # business_avg_ratings1(bus_avg)
    business_avg_ratings = calculate_average_business_ratings(
        train_rdd, dictionary_all_businesses
    )

    # user_avg_ratings1
    user_avg_ratings = calculate_average_user_ratings(train_rdd, dictionary_all_users)

    # buckets_user_business1

    buckets_user_business = map_users_to_businesses_and_ratings(
        train_rdd, dictionary_all_users, dictionary_all_businesses
    )

    test_data = test_rdd.map(
        lambda x: (dictionary_all_businesses[x[1]], dictionary_all_users[x[0]])
    )


# Incorporate the calculate_prediction function into the workflow
predictions = test_data.map(
    lambda x: calculate_predicted_rating(
        x[0],
        x[1],
        group_business,
        buckets_user_business,
        business_avg_ratings,
        user_avg_ratings,
    )
).collect()


def reverse_dictionary(input_dict):
    """Reverse the keys and values of a given dictionary."""
    return {value: key for key, value in input_dict.items()}


for a, b, c in predictions:
    collaborative_filtering_results = np.asarray(c, dtype="float")


def load_data_files(folder_path, sc):
    user_data = sc.textFile(folder_path + "/user.json")
    business_data = sc.textFile(folder_path + "/business.json")
    review_train_data = sc.textFile(folder_path + "/review_train.json")

    return user_data, business_data, review_train_data


# Load different data files
user_data, business_data, review_train_data = load_data_files(folder_path, sc)

# Create feature mappings
user_features = user_data.map(json.loads).map(
    lambda i: (
        i["user_id"],
        (
            i["useful"],
            i["compliment_hot"],
            i["fans"],
            i["review_count"],
            i["average_stars"],
            i["compliment_funny"],
            i["compliment_more"],
            i["compliment_cool"],
            i["compliment_profile"],
            i["compliment_note"],
            i["compliment_cute"],
            i["compliment_list"],
            i["compliment_plain"],
            i["compliment_writer"],
            i["compliment_photos"],
        ),
    )
)
user_features = user_features.collectAsMap()

business_features = get_business_features(business_data)

review_train_features = extract_review_features(review_train_data)

# Only used the training dataset to train the model
x_train, y_train = create_dataset(
    train_rdd, user_features, business_features, review_train_features, False
)
x_test, y_test = create_dataset(
    test_rdd, user_features, business_features, review_train_features, True
)


test_dataset = test_rdd.map(
    lambda record: preprocess_data(
        record, user_features, business_features, review_train_features, True
    )
).collect()


xgb_model = train_xgb_model(x_train, y_train)

# Predict on test set
xgb_model_results = predict_xgb_model(xgb_model, x_test)

# Assigned more weights to xgboost since it have me more accurate predictions
combined_results = (
    0.99999999 * xgb_model_results + (1 - 0.99999999) * collaborative_filtering_results
)
test_dataset_array = np.array(test_dataset)
final_results_combined = np.c_[test_dataset_array[:, :2], combined_results]


save_to_csv(
    output_filepath,
    final_results_combined,
    header=["user_id", "business_id", "prediction"],
)


def compute_rmse_and_error_distribution(test_file, output_file):

    def preprocess_file(file_path):
        data = sc.textFile(file_path)
        header = data.first()
        return (
            data.filter(lambda row: row != header)
            .map(lambda row: row.strip().split(","))
            .map(lambda row: ((row[0], row[1]), float(row[2])))
            .sortByKey()
        )

    # Preprocess test and output files
    test_pairs = preprocess_file(test_file)
    output_pairs = preprocess_file(output_file)

    differences = test_pairs.join(output_pairs).map(
        lambda row: abs(row[1][0] - row[1][1])
    )

    # Compute RMSE
    mse = differences.map(lambda diff: diff**2).reduce(lambda x, y: x + y)
    rmse = math.sqrt(mse / differences.count())

    # Error distribution
    distribution_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, float("inf"))]

    print("Error Distributions:")
    for lower_bound, upper_bound in distribution_ranges:
        count = differences.filter(
            lambda diff: lower_bound <= diff < upper_bound
        ).count()
        print(f">={lower_bound} and <{upper_bound}: {count}")
    print("")
    print(f"RMSE:")
    print(rmse)


compute_rmse_and_error_distribution(test_file, output_filepath)
