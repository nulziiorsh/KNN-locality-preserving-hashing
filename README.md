# KNN-locality-preserving-hashing

Author: Nasanbayar Ulzii-Orshikh

Date: 09/15/2020

**META-DESCRIPTION:**

Inevitably, I learned from the very first lab of the Machine Learning course how to better implement KNN, namely,
non-arbitrarily compute the hyperparameter K, that Iris Form bases upon. Additionally, to achieve a better time complexity, I
implemented a Locality-preserving hashing (LPH) algorithm that in a nutshell breaks the vector space into buckets and given a feature
vector accesses a bucket of its nearest neighbors in O(1).

**DESCRIPTION:**

knn.py is an implementation of the full experimentation pipeline for a K-Nearest Neighbor model,
trained with a custom locality-preserving hashing algorithm, available in locality_preserving_hashing.py.

**DESIGN:**

Locality-preserving hashing (LPH) algorithm consists of two parts: breaking the feature space into buckets and given
a vector, hashing it to access only the "relevant" bucket of the space -- a group of vectors that are the closest
to the given vector.

To break the space into buckets, [1] uses a vector, the components of which is selected randomly from a Gaussian
distribution, to compute its dot products with the training examples and find the threshold probability above which two
vectors are likely to be in the same part of the feature space. However, when I tried to use its core concept without
the probability thresholds, it wasn't possible: if we have the "template" vector [1,1] and find its dot products with
[3,-2] and [0,1], both would result in 1, although these two vectors likely represent two very different instances.

At the same time, [2] proposes breaking the space radially. However, for instance, in the two-dimensional space, doing so
would mean we'd have to find the distances between a given vector in the first quarter with the vectors in the same quarter,
as well as the second, third, and fourth quarters, although vectors there likely represent completely different instances
from the that of the given vector. Further, this would blow up the complexity in higher dimensions.

Instead, I propose to first hold the directionality as "constant" (without considering it for hashing) and bucket the
vectors radially by dividing the radius range into several sections, then for each radius bucket, hold the radius as "constant" (as the radii of the vectors in a given
bucket would be essentially the same) and bucket the vectors directionally by using a few "template" vectors. In the script,
I use 10 radial buckets and 2 "template vectors" that each break the dot-product range into two sections, resulting in
total of 40 buckets. The details of how, what I call, the granularity and other properties of the parts in the vector
space was computed can be read in locality_preserving_hashing,py, and the hand notes about the process can be found in lph_notes.pdf.

To implement this algorithm, I use a custom KnnHashTable data structure and hashing function. The reason I decided to
go with these is because the way that the vectors are hashed is not individual vector key leading to some value as is in a Python dictionary,
but the vector that’s being hashed becomes part of an iterable that the vector itself leads to. If I use a dictionary as a data structure based on hashing,
it must preserve both the key and the values. That is, for each vector of the same bucket in the feature space,
I’d have to assign the same bucket of vectors in the dictionary, which then would have to be updated each
time there is a new vector in it for each of the vector keys that you can access it with. That is computationally
too expensive. Instead, the custom data structure leverages on the collision of keys, where two vectors can lead to the same bucket.
One might argue that it is the same as the value in the dictionary being a pointer to some list. However, even if it was,
a Python dictionary does not allow a custom hashing function, instead hashing each vector to an automatic, unique bucket, in this case, of a pointer, while the custom
data structure allows such a two-step hashing as described above and directly leads to the specific bucket of interest
without storing a pointer for each of the vector keys.

**EXTENSIONS IMPLEMENTED:**

1. Performs a multi-class classification: generally, adding another class seems to make a small bump in the middle of the graph
more evident or even appear, increasing the value of the optimal K.
2. Creates a plot of Accuracy vs. K.
3. Takes command line arguments in the following order:

    1. Digit classes: -1 makes the algorithm take all classes in [0..9]; "[2,3,4]" (with quote marks and
    no space after the comma) makes the algorithm take class 2, 3, and 4.
    2. Training data volume: -1 means to take all of the training data; 4500 (integer) means take 4500 of the training
    examples.
    3. K value: -1 means run on values of K in [1..10]; 5 (integer) means run on a single value of K = 5.
    The choice represents a mental switch in the experimentation pipeline, where in the former situation,
    we’re trying to find the best K within the range of candidates from the partitioned training data,
    which the problem set refers to as the “test data”; or in the latter situation, it’s assumed that we’ve already
    chosen the most optimal k and thus want to simply calculate the accuracy of it when applied to that “test data”.
    However, one could argue that we might want to find the most optimal k within a smaller range of candidates —
    I think that would be totally valid; we’d then have to change how the input is processed and propagates throughout
    the pipeline.
    4. Comparison mode: 0 means not a comparison mode and 1 means indeed a comparison mode, which means that
    the KNN-model trained with a custom locality-preserving hashing (LPH) algorithm would be compared  in terms of its time efficiency to the classic
    KNN-prediction algorithm from "A Course in Machine Learning" by Hal Daumé III (p. 33), modified to be
    more efficient by sorting the distances only once and using them for the different K values. This is because
    it would be unfair if the hashing method uses the same distance array for finding K neighbors in range(K),
    while the classic one has to calculate distances for each test vector each time it receives a new K.
    Instead, both algorithms use their own and the same distance array every time they have to find K nearest neighbors
    for different K values. The difference in time then really comes from (not) breaking the vector space into buckets and (not) using hashing
    function for each test vector to get neighbors only from a set of vectors that are in the vicinity of the hashed vector.

Example inputs to the terminal:

- Python3 knn.py "[2,3]" 4000 7 0
- Python3 knn.py "[1,7,8]" 5124 -1 0
- Python3 knn.py "[4,0]" -1 -1 1
- Python3 knn.py "[6,7]" -1 4 1
- Python3 knn.py -1 -1 -1 1

**RESULTS:**

Even despite the classic model made more efficient, the KNN model based on the
the custom locality-preserving hashing algorithm is on average 91% faster than the classic model, while its
accuracy does not differ from the classic one by more than 8%, keeping the overall accuracy higher than 87%
throughout its performance. This implies that the model has well learnt the classification task.

Although depending on the context in which the model is being used -- for instance, in the case of helping a visually
impaired person --, the minimum 87% accuracy might not be suitable, for other contexts where the user is trying to get a sense
of the data as fast as they can, being able to perform the classification task at such an accuracy level within 1/10th
of the classic time is indeed very useful.