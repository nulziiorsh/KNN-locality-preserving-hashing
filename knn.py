"""
Full experimentation pipeline for a K-Nearest Neighbor model,
trained with a custom locality-preserving hashing (LPH) algorithm.
Author: Nasanbayar Ulzii-Orshikh
Date: 09/15/2020
"""
import numpy as np
import sys
import locality_preserving_hashing as lph
import math
import matplotlib.pyplot as plt
import time
import time_complexity_comparison as tcc

TRAIN_DATA = np.loadtxt("../cs360_data_zip/zip.train")
TEST_DATA = np.loadtxt("../cs360_data_zip/zip.test")
TRAIN_DATA_SHAPE = (7291, 1 + 256)
TEST_DATA_SHAPE = (2007, 1 + 256)

# The functions are ordered following a pre-order traversal of the function tree.
# The script was written with the Top-Down Design approach.

def main():
    """The driver function: takes inputs and handles the training and classification processes."""
    # Prints the loaded array's dimensions and checks if they match those of the original data set.
    print("\nK-Nearest Neighbor model input:")
    print(" " * 4 + "train_data shape:", TRAIN_DATA.shape)
    print(" " * 4 + "test_data shape:", TEST_DATA.shape)
    if TRAIN_DATA.shape != TRAIN_DATA_SHAPE or TEST_DATA.shape != TEST_DATA_SHAPE:
        raise ValueError("the dimensions of the loaded array don't match those of the original dataset")
    else:
        print(" " * 4 + "The dimensions of the loaded array match those of the original dataset.")

    # Settings for the digit_list, training_data_volume, K hyper-parameter, and comparison_mode.
    # In addition, for each of these items, when appropriate, TypeError is automatically raised
    # when converting the inputs into int type and ValueError with manually written conditions.
    print("\nSettings to be used for the script:")

    # Settings for the digit_list:
    if sys.argv[1] == "-1":
        digit_list = [i for i in range(10)]
    else:
        digit_list = [int(i) for i in sys.argv[1][1:-1].split(",")]
        num_of_classes = len(digit_list)
        if num_of_classes < 2:
            raise ValueError("digit_list must have at least 2 classes")
        for i in digit_list:
            if i > 9 or i < 0:
                raise ValueError("digit must be within 0 and 9, both included")
    print(" " * 4 + "digit_list =", digit_list)

    # Settings for the training_data_volume:
    if sys.argv[2] == "-1":
        training_data_volume = len(TRAIN_DATA)
    else:
        training_data_volume = int(sys.argv[2])
        if training_data_volume > TRAIN_DATA_SHAPE[0] or training_data_volume < 1:
            raise ValueError("training_data_volume must be within 1 and {}, both included".format(TRAIN_DATA_SHAPE[0]))
    print(" " * 4 + "training_data_volume =", training_data_volume)

    # Settings for the K hyper-parameter:
    if sys.argv[3] == "-1":
        k = [i for i in range(1,11)]
    else:
        k = int(sys.argv[3])
        if k < -1 or k == 0:
            raise ValueError("k must be either -1, or a positive number")
    print(" " * 4 + "k =", k)

    # Settings for the comparison_mode:
    if sys.argv[4] == "0" or sys.argv[4] == "1":
        comparison_mode = int(sys.argv[4])
    else:
        raise ValueError("comparison_mode must be either 0, or 1")
    print(" " * 4 + "comparison_mode =", comparison_mode)

    # ++++++++++ TRAINING BEGINS ++++++++++ #

    # Filters both the training and test data for the digit classes that the user specified in digit_list.
    train_data_filtered = filter_data(digit_list, TRAIN_DATA, training_data_volume)
    test_data_filtered = filter_data(digit_list, TEST_DATA, TEST_DATA_SHAPE[0])
    print("\nFiltered data shapes:")
    print(" " * 4 + "Filtered train data shape:", train_data_filtered.shape)
    print(" " * 4 + "Filtered test data shape:", test_data_filtered.shape)

    # Partitions the data set into vectors and labels.
    train_vectors, train_labels_vector = partition_labels(train_data_filtered)
    test_vectors, true_labels_vector = partition_labels(test_data_filtered)
    print("\nPartitioned data shapes:")
    print(" " * 4 + "train_vectors shape:", train_vectors.shape)
    print(" " * 4 + "train_labels_vector shape:", train_labels_vector.shape)
    print(" " * 4 + "test_vectors shape:", test_vectors.shape)
    print(" " * 4 + "true_labels_vector shape:", true_labels_vector.shape)

    # Trains the KNN model using the LPH algorithm, breaking the vector space into buckets of vectors existing
    # in each other's vicinity, to increase the speed of the classification process.
    # That is, to find the K nearest neighbors for a test vector, we don't need to calculate the distance
    # between our test example and every vector in the vector space, but only between those that are
    # in the same bucket (of the vector space) as our test example.
    feature_space = train_knn(train_vectors, train_labels_vector)
    print("\nFeature space:", feature_space)
    # print(" " * 4 + "Feature space array:", feature_space.array)
    print(" " * 4 + "Feature space radius_threshold:", feature_space.radius_threshold)
    print(" " * 4 + "Feature space number of radius buckets:", feature_space.num_of_radius_buckets)
    print(" " * 4 + "Feature space direction_threshold_1:", feature_space.direction_threshold_1)
    print(" " * 4 + "Feature space direction_threshold_2:", feature_space.direction_threshold_2)
    print(" " * 4 + "Feature space number of direction buckets:", feature_space.num_of_direction_buckets)

    # Prints the numbers of training instances in each hash in the hashing grid,
    # into which the vector space is broken down and bucketed.
    feature_space.print()

    # ++++++++++ CLASSIFICATION BEGINS ++++++++++ #

    # Turns the mental switch for the experiment based on K from the input:
    # if type(k) == list then the experiment is about finding the optimal K from the range of candidate values [1..10];
    # elif type(k) == int then it's assumed that the optimal K is found, and we'd like to find its accuracy.

    # We first check if it's the former case:
    if type(k) == list:
        # Starts the timer for measuring how long the classification based on the LPH algorithm takes.
        tic = time.perf_counter()
        print("\nLOCALITY-HASHED KNN:")

        # Pre-classifies the test_vectors: instead of finding distances between vectors and sorting them for each K
        # value, the hash function each assigns each test_vector into its corresponding bucket,
        # computes the distances between each example in it and the test_vector,
        # sorts them once and for all, and returns the list of the closest 10 examples.
        print("\nPre-classification running...")
        pre_classified_list = knn_pre_classify(feature_space, test_vectors)
        print("Pre-classification is complete")

        # Holder for the accuracy rate for each K value.
        accuracy_by_k = []

        print("\nProgress:")
        # For each K value, computes the prediction vector and adds its accuracy rate into the holder.
        for i in range(len(k)):
            # Prediction vector is calculated based on a voting procedure, since we want our pipeline to be able to
            # handle more than 2 classes of digits.
            prediction_vector = knn_classify(pre_classified_list, digit_list, k[i])
            print(" " * 4 + "Prediction {}/{} completed".format(i + 1, len(k)))

            # Accuracy is the ratio between the number of correctly predicted digits and the number of test cases used.
            # We use an indicator variable to compute it: that is, comparing component to component
            # whether the digits between the vectors match.
            accuracy_by_k.append(quantify_accuracy(prediction_vector, true_labels_vector))

        # Reports the accuracy by each K, rounded to 3 decimal places.
        print("\nNearest Neighbors:")
        for i in range(len(accuracy_by_k)):
            print("K={}, {}%".format(i + 1, '{:.3f}'.format(round(accuracy_by_k[i], 3))))

        # Stops the timer that measures how long the classification based on the LPH algorithm takes.
        toc = time.perf_counter()
        # Calculates the difference between the starting and stopping times.
        duration_lh = toc - tic
        # Reports the duration of the classification process based on the LPH algorithm.
        print(f"\nLOCALITY-HASHED KNN run in {duration_lh:0.3f} seconds")

        # Plots the classification results on an Accuracy vs. K graph.
        plot("Locality-Hashed", accuracy_by_k)

        # Comparison mode is made for analyzing the differences in time and accuracy between the classifications
        # performed by our custom LPH-based prediction algorithm and the classic KNN-prediction algorithm from
        # "A Course in Machine Learning" by Hal Daumé III (p. 33), modified to be more efficient
        # by similarly sorting the distances only once and using them for the different K values.
        if comparison_mode:
            # Similarly, starts the timer for the classification process based on the classic KNN-prediction algorithm.
            tic = time.perf_counter()
            print("\nCLASSIC KNN:")

            # Holder for the accuracy rate for each K value.
            accuracy_by_k_classic = []
            print("\nProgress:")
            # Instead of sorting the distances between test_vectors and training vectors for each k,
            # sorts it once and for all and uses it for the prediction.
            sorted_distances = tcc.sorted_distances(train_vectors, train_labels_vector, test_vectors)
            # For each K value, computes the prediction vector and adds its accuracy rate into the holder.
            for i in range(len(k)):
                # Prediction vector is similarly based on a voting procedure, since we want our pipeline to be able to
                # handle more than 2 classes of digits.
                prediction_vector = tcc.knn_classic(sorted_distances, k[i], test_vectors, digit_list)
                print(" " * 4 + "Prediction {}/{} completed".format(i + 1, len(k)))

                # Accuracy is similarly calculated as the ratio between the number of correctly predicted digits and
                # the number of test cases used.
                accuracy_by_k_classic.append(quantify_accuracy(prediction_vector, true_labels_vector))

            # Reports the accuracy by each K, rounded to 3 decimal places.
            print("\nNearest Neighbors:")
            for i in range(len(accuracy_by_k_classic)):
                print("K={}, {}%".format(i + 1, '{:.3f}'.format(round(accuracy_by_k_classic[i], 3))))

            # Stops the timer for the classification.
            toc = time.perf_counter()
            # Calculates the difference between the starting and stopping times.
            duration_classic = toc - tic
            # Reports the duration of the classification process based on the classic KNN-prediction algorithm.
            print(f"\nCLASSIC KNN run in {duration_classic:0.3f} seconds")

            # Reports by how much the two algorithms differ in time efficiency.
            print("\nLOCALITY-HASHED KNN is {}% faster than CLASSIC KNN".format('{:.3f}'.format(round((1 - duration_lh / duration_classic) * 100, 3))))

            # Plots the classification results on the same Accuracy vs. K graph.
            plot("Classic", accuracy_by_k_classic)

            # Saves the graphs in one file for a visual comparison.
            save_plot("comparison", digit_list)
        else:
            # If the mode is not comparison, no further change is made to the graph, and it is saved as it is.
            # This is all possible because plot() function plots all the graphs in the same field.
            save_plot("single", digit_list)

    # Then, check if it's latter case, where it's assumed that the optimal K is found,
    # and we'd like to find its accuracy.
    elif type(k) == int:
        # Starts the timer for the classification process based on the LPH algorithm.
        tic = time.perf_counter()
        print("\nLOCALITY-HASHED KNN:")

        # Uses the same procedure of pre-classification, described above.
        print("\nPre-classification running...")
        pre_classified_list = knn_pre_classify(feature_space, test_vectors)
        print("Pre-classification is complete")

        # The same procedures of making a prediction based on LPH and finding the accuracy.
        prediction_vector = knn_classify(pre_classified_list, digit_list, k)
        accuracy = quantify_accuracy(prediction_vector, true_labels_vector)

        # Reports the accuracy for the given K, rounded to 3 decimal places.
        print("\nNearest Neighbors:")
        print("K={}, {}%".format(k, '{:.3f}'.format(round(accuracy, 3))))

        # Stops the timer for classification.
        toc = time.perf_counter()
        # Calculates the difference between the starting and stopping times.
        duration_lh = toc - tic
        # Reports the duration of the classification process based on the LPH algorithm.
        print(f"\nLOCALITY-HASHED KNN run in {duration_lh:0.3f} seconds")

        # The same check for comparison_mode as described above.
        if comparison_mode:
            # Starts the timer for the classification process based on the classic KNN-prediction algorithm.
            tic = time.perf_counter()
            print("\nCLASSIC KNN:")

            # Similarly, instead of sorting the distances between test_vectors and training vectors for each k,
            # sorts it once and for all and uses it for the prediction.
            sorted_distances = tcc.sorted_distances(train_vectors, train_labels_vector, test_vectors)

            # The same procedures of making a classic prediction and finding the accuracy.
            prediction_vector = tcc.knn_classic(sorted_distances, k, test_vectors, digit_list)
            accuracy = quantify_accuracy(prediction_vector, true_labels_vector)

            # Reports the accuracy for the given K, rounded to 3 decimal places.
            print("\nNearest Neighbors:")
            print("K={}, {}%".format(k, '{:.3f}'.format(round(accuracy, 3))))

            # Stops the timer for the classification.
            toc = time.perf_counter()
            # Calculates the difference between the starting and stopping times.
            duration_classic = toc - tic
            # Reports the duration of the classification process based on the classic KNN-prediction algorithm.
            print(f"\nCLASSIC KNN run in {duration_classic:0.3f} seconds")

            # Reports by how much the two algorithms differ in time efficiency.
            print("\nLOCALITY-HASHED KNN is {}% faster than CLASSIC KNN".format(
                '{:.3f}'.format(round((1 - duration_lh / duration_classic) * 100, 3))))

def filter_data(digit_list, data, num_of_rows):
    """Filters data for the digit classes, chosen by the user."""
    # Holder for the filtered data list.
    filtered_data = []
    # Filters each of the vectors in the data using the digit classes, chosen by the user.
    for i in range(num_of_rows):
        # If the label of the vector is in the digit classes, includes the vector in the filtered data list.
        if int(data[i][0]) in digit_list:
            filtered_data.append(data[i])
    # Transforms the filtered data list into an numpy array.
    filtered_data = np.array(filtered_data)
    # Returns the filtered data.
    return filtered_data

def partition_labels(data):
    """Partitions the data into labels and vectors."""
    # Copies the column of labels from the data.
    labels = data[:, 0]
    # Deletes that column from the data, updating the same "data" pointer to save memory space.
    data = np.delete(data, 0, axis=1)
    # Returns the partitioned vectors and labels.
    return data, labels

def train_knn(data, labels):
    """Trains the KNN model using the LPH algorithm."""
    # Creates a custom KNN hash table that will contain the vector space, broken into buckets.
    knn_hash_table = lph.KnnHashTable(data)
    # Assigns each training vector into one of these buckets using the hashing function.
    for i in range(len(data)):
        knn_hash_table.add(data[i], labels[i])
    # Returns the hash table as the trained KNN model.
    return knn_hash_table

def knn_pre_classify(feature_space, test_vectors):
    """Pre-classifies the test_vectors.
    Details: instead of finding distances between vectors and sorting them for each K value,
    the hash function each assigns each test_vector into its corresponding bucket,
    computes the distances between each example in it and the test_vector,
    sorts them once and for all, and returns the list of the closest 10 examples."""
    # Holder for each test vector and 10 examples the closest to it.
    pre_classified_list = []

    # For each test vector, finds the radius and direction bucket indices using the hash function.
    for test_vector in test_vectors:
        radius_bucket_index, direction_bucket_index = feature_space.hash(test_vector)

        # Holder for all calculated distances between training examples and the test vector as well as the training
        # example's corresponding label in a tuple to make sure the vector-label correspondence is preserved.
        distance_label_tuples_list = []

        # Calculates and appends these elements mentioned above into the holder.
        # In the feature_space, each training example is represented by a tuple consisting of
        # the vector and the corresponding label.
        for a_tuple in feature_space.get(radius_bucket_index, direction_bucket_index):
            distance_label_tuples_list.append((distance(test_vector, a_tuple[0]), a_tuple[1]))

        # Sorts the examples: since in a tuple from the distance_label_tuples_list the distance precedes,
        # the sorting is done on the distances.
        distance_label_tuples_list.sort()

        # Holder for the closest 10 examples.
        closest_10_max_list = []

        # Appends the closest 10 examples to the holder.
        for i in range(10):
            try:
                closest_10_max_list.append(distance_label_tuples_list[i])
            except:
                break

        # Appends the pair of test vector and the closest 10 training examples to it to the relevant holder.
        pre_classified_list.append((test_vector, closest_10_max_list))

    # Returns the pre-classified list.
    return pre_classified_list

def knn_classify(pre_classified_list, digit_list, k_value):
    """Classifies each vector by the digit classes based on a voting procedure."""
    # Holder for the predictions of the digit classes.
    prediction_vector = []

    # For each test vector in the pre_classified_list, performs the voting procedure.
    for i in range(len(pre_classified_list)):
        # Holder for votes for each class, chosen by the user.
        votes = {j: 0 for j in digit_list}
        # Takes the closest K training example neighbors and looks up their label.
        for k in range(k_value):
            # In case the number of available neighbors in the bucket is smaller than the K value, handles the error.
            try:
                # Increments the vote count, corresponding to the training example's label.
                votes[pre_classified_list[i][1][k][1]] += 1
            except:
                break
        # Appends each prediction into the prediction vector.
        prediction_vector.append(max(votes, key=votes.get))

    # Returns the prediction vector.
    return prediction_vector

def distance(vector_i, vector_j):
    """Finds the distance between two vectors."""
    # Accumulator variable for the sums of the squares of the differences between the vectors' components.
    a_sum = 0
    # For each component of one of the vectors:
    for k in range(len(vector_i)):
        # Adds the squares of the components' differences to the sum.
        a_sum += (vector_i[k] - vector_j[k])**2
    # Returns the square-root of the sum, that is, the distance between the two vectors.
    return math.sqrt(a_sum)

def quantify_accuracy(prediction_vector, true_labels_vector):
    """Calculates the accuracy component by component between prediction and true labels vectors."""
    # Accumulator variable for the components that match in the two vectors.
    indicator_sum = 0
    # For each component of the vector:
    for i in range(len(true_labels_vector)):
        # If the components match then increments the accumulator variable.
        if prediction_vector[i] == true_labels_vector[i]:
            indicator_sum += 1
    # Returns the ratio between the Accumulator variable and the total number of components using percentage.
    return indicator_sum / len(true_labels_vector) * 100

# Adapted from https://stackoverflow.com/questions/20130227/matplotlib-connect-scatterplot-points-with-line-python
def plot(indicator, accuracy_by_k):
    """Plots the accuracies in a scatter-plot"""
    # Places the data of the plot into lists.
    x_data = [i + 1 for i in range(len(accuracy_by_k))]
    y_data = accuracy_by_k
    # Plots the data.
    plt.scatter(x_data, y_data)
    # Gives the lot a label to be put in the legend.
    plt.plot(x_data, y_data, label=indicator)
    # Writes each K value on the x-axis.
    plt.xticks(x_data)
    return

# Adapted from https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
def save_plot(indicator, digit_list):
    """Assigns relevant information to the graphs and saves the figure."""
    # Adds relevant information to the figure.
    plt.title("Accuracy (%) vs. Value of K for the K-Nearest Neighbor Model\nClasses used: {}".format(digit_list))
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy (%)")
    plt.legend(framealpha=1, frameon=True)
    # Transforms the digit_list into a string to be used in the name of the figure.
    digit_list_string = "".join([str(i) for i in digit_list])
    # Names the figure with a unique name.
    plt.savefig("graph_outputs/{}_K_{}.png".format(indicator, digit_list_string), bbox_inches='tight')
    return

if __name__ == "__main__":
    main()

