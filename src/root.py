import pandas as pd
from sklearn import svm
from sklearn import neural_network as nn
from sklearn.externals import joblib

import datetime
import os.path

# path for dataset files
file_train_path = "../data/train/train_new.csv"
file_test_path = "../data/test/test_new.csv"

# path for saved models
linearSVR_model_path = "./models/linearSVR.sav"
MLPRegressor_model_path = "./models/MLPRegressor.sav"

# path for outputted results
linearSVR_output_path = "./output/linearSVR_submission.csv"
MLPRegressor_output_path = "./output/MLPRegressor_submission.csv"


# Load/Read the training and testing data
time_start = datetime.datetime.now()
print("Reading training data...")
time_start_read_train = datetime.datetime.now()
train_original = pd.read_csv(file_train_path)
time_end_read_train = datetime.datetime.now()
print(f"Done reading training data. This took {time_end_read_train - time_start_read_train} seconds")

print("Reading testing data...")
time_start_read_test = datetime.datetime.now()
test_original = pd.read_csv(file_test_path)
time_end_read_test = datetime.datetime.now()
print(f"Done reading training data. This took {time_end_read_test - time_start_read_test} seconds")

# Features deemed not relevant/necessary to fit and test
droppedFeatures = ["Id", "groupId", "matchId", "killPoints", "maxPlace", "roadKills", "swimDistance", "teamKills",
                   "vehicleDestroys", "winPoints", "numGroups"]

# Setting up the training dataset

# Remove unwanted features from the training data
print("Cleaning train")
train_cleanedFeatures = train_original.drop(droppedFeatures, axis=1).copy()
# Separate the features from the label. Replace null/NaNs with negligible value.
#   This shouldn't change much as there is expected to only be one or two of such values in the dataset
print("getting label_train")
label_train = train_cleanedFeatures["winPlacePerc"].to_frame().fillna(0.0001)
print("getting features_train")
features_train = train_cleanedFeatures.drop("winPlacePerc", axis=1)

# Setting up the testing dataset

# Remove unwanted features from the testing data
print("Cleaning test")
test_cleanedFeatures = test_original.drop(droppedFeatures, axis=1).copy()
# Separate the features from the label
print("getting features_test")
features_test = test_cleanedFeatures


# See if linearSVR model already exists
if os.path.isfile(linearSVR_model_path):
    clf_linearSVR = joblib.load(linearSVR_model_path)
    print(f"Loaded classifier from {linearSVR_model_path}")
else:
    # Define the classifier
    clf_linearSVR = svm.LinearSVR(verbose=1)
    print(clf_linearSVR)

    # Fit linearSVR model to training data
    print("Starting fit...")
    time_start_fit = datetime.datetime.now()
    clf_linearSVR.fit(features_train, label_train)
    time_end_fit = datetime.datetime.now()
    print(f"Finished fitting. This took {time_end_fit - time_start_fit} seconds")
    print(f"Dumping classifier into {linearSVR_model_path}")
    joblib.dump(clf_linearSVR, linearSVR_model_path)

# Get score
linearSVR_train_score = clf_linearSVR.score(features_train, label_train)
print(f"LinearSVR score on training data: {linearSVR_train_score}")

# Predict test
print("Starting test...")
time_start_test = datetime.datetime.now()
label_test_predicted = clf_linearSVR.predict(features_test)
time_end_test = datetime.datetime.now()
print(f"Done testing. This took {time_end_test - time_start_test} seconds")


# See if MLPRegressor model already exists
if os.path.isfile(MLPRegressor_model_path):
    clf_MLPRegressor = joblib.load(MLPRegressor_model_path)
    print(f"Loaded classifier from {MLPRegressor_model_path}")
else:
    # Define the classifier
    clf_MLPRegressor = nn.MLPRegressor(hidden_layer_sizes=(100, 100), activation="relu", solver="adam", verbose=True)
    print(clf_MLPRegressor)

    # Build it yo!
    print("Starting MLPRegressor fit...")
    time_start_fit = datetime.datetime.now()
    clf_MLPRegressor.fit(features_train, label_train)
    time_end_fit = datetime.datetime.now()
    print(f"Finished fitting. This took {time_end_fit - time_start_fit} seconds")
    print(f"Dumping classifier into {MLPRegressor_model_path}")
    joblib.dump(clf_MLPRegressor, MLPRegressor_model_path)

# Get score for MLPRegressor
MLPRegressor_train_score = clf_MLPRegressor.score(features_train, label_train)
print(f"MLPRegressor score on training data: {MLPRegressor_train_score}")

# Predict test
print("Starting linearSVR test...")
time_start_test = datetime.datetime.now()
linearSVR_label_test_predicted = clf_linearSVR.predict(features_test)
time_end_test = datetime.datetime.now()
print(f"Done testing. This took {time_end_test - time_start_test} seconds")
linearSVR_output = pd.concat([test_original.Id, pd.Series(linearSVR_label_test_predicted)], axis=1)
linearSVR_output = linearSVR_output.rename(columns={"Id": "Id", 0: "winPlacePerc"})
linearSVR_output.to_csv(linearSVR_output_path, index=False)
print(f"Saved output from linearSVR to{linearSVR_output_path}")

# Predict test
print("Starting MLPRegressor test...")
time_start_test = datetime.datetime.now()
MLPRegressor_label_test_predicted = clf_MLPRegressor.predict(features_test)
time_end_test = datetime.datetime.now()
print(f"Done testing. This took {time_end_test - time_start_test} seconds")
MLPRegressor_output = pd.concat([test_original.Id, pd.Series(MLPRegressor_label_test_predicted)], axis=1)
MLPRegressor_output = MLPRegressor_output.rename(columns={"Id": "Id", 0: "winPlacePerc"})
MLPRegressor_output.to_csv(MLPRegressor_output_path, index=False)
print(f"Saved output from MLPRegressor to{MLPRegressor_output_path}")
