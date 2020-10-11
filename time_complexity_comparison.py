"""
Implementation of the classic KNN-prediction algorithm from "A Course in Machine Learning" by Hal Daumé III (p. 33),
modified to be more efficient by sorting the distances only once and using them for the different K values.

Its purpose is to helps us understand if the custom locality-preserving hashing (LPH) algorithm is more time-efficient
than the classic KNN-prediction algorithm or not.

Author: Nasanbayar Ulzii-Orshikh
Date: 09/15/2020
"""
import knn

def sorted_distances(train_vectors, train_labels_vector, test_vectors):
    # Holder list for sorted distances for each of the test vectors.
    sorted_distances = []

    # For each of the test vectors, finds its distance from every training vector.
    for test_vector in test_vectors:
        # Holder for distances.
        distances = []
        # Given a test_vector, finds its distance from each training vector and appends the tuple of the distance and
        # training example label to the holder for distances.
        for i in range(len(train_vectors)):
            distances.append((knn.distance(test_vector, train_vectors[i]), train_labels_vector[i]))
        # Sorts the distances: since the distance precedes the label in the tuple, the sorting is done on the distances.
        distances.sort()
        # Appends the sorted holder of distances to the holder of sorted distances.
        sorted_distances.append(distances)

    # Returns the list of sorted distances.
    return sorted_distances

def knn_classic(sorted_distances, k_value, test_vectors, digit_list):
    # Holder for the predictions of the digit classes.
    prediction_vector = []

    # For each of test vectors, perform the voting procedure.
    for i in range(len(test_vectors)):
        # Holder for votes for each class, chosen by the user.
        votes = {j: 0 for j in digit_list}
        # Takes the closest K training example neighbors and looks up their label.
        for k in range(k_value):
            # In case the number of available neighbors in the bucket is smaller than the K value, handles the error.
            try:
                # Increments the vote count, corresponding to the training example's label.
                votes[sorted_distances[i][k][1]] += 1
            except:
                break
        # Appends each prediction into the prediction vector.
        prediction_vector.append(max(votes, key=votes.get))

    # Returns the prediction vector.
    return prediction_vector
