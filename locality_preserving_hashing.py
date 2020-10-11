"""
Implementation of a KnnHashTable class, entailing a custom locality-preserving hashing (LPH) algorithm.
Author: Nasanbayar Ulzii-Orshikh
Date: 09/15/2020
"""
import knn
import math
import numpy as np

RADIUS_BUCKET_NUM = 10
DIRECTION_BUCKET_NUM = 4

class KnnHashTable:
    """KnnHashTable class, an instance of which represents the hash grid, or the vector space broken into parts,
    enabling us for each given vector access only the "relevant", that is, the closest to it vectors in the feature space.

    In essence, the class's implementation entails the locality-preserving hashing algorithm in two components:
    1. The method that breaks the vector space into parts.
    2. The method that gets a "relevant" part of the vector space when given a vector."""

    def __init__(self, data):
        """Initializes the KnnHashTable class."""
        # Sets the default length of the vectors at 256.
        self._num_of_rows = knn.TRAIN_DATA_SHAPE[1] - 1
        # Finds the mid-length of the default vector length to be used in making the template vectors
        # perpendicular to each other.
        self._num_of_shaped_rows = int(self._num_of_rows / 2)

        # Creates "template" vectors, completely perpendicular to each other, to be used
        # for hashing in terms of a vector's direction.
        self.template_vector_1 = np.zeros(self._num_of_rows, )
        for i in range(self._num_of_shaped_rows):
            # In doing so, makes the components equal to -0.5, a number comparable to those in the components
            # of the data set's vectors.
            self.template_vector_1[i] = -0.5

        self.template_vector_2 = np.zeros(self._num_of_rows, )
        for i in range(self._num_of_shaped_rows):
            # Similarly, makes the components equal to -0.5.
            self.template_vector_2[self._num_of_shaped_rows + i] = -0.5

        # Calculates the values for the following variables:
        # radius_threshold: a length threshold that divides the entire range of the radii of the
        #                   training vectors into radius buckets.
        # min_radius: minimum radius among the training vectors.
        # direction_threshold_1: a length threshold, multiples of which divide the entire range of direction values
        #                        that are dot products between the training vectors and the FIRST "template" vector.
        #                        Using this threshold we get the first coordinate of the direction buckets.
        # min_dir_1: minimum direction value among the dot products with the FIRST "template" vector.
        # direction_threshold_1: a length threshold, multiples of which divide the entire range of direction values
        #                        that are dot products between the training vectors and the SECOND "template" vector.
        #                        Using this threshold we get the SECOND coordinate of the direction buckets.
        # min_dir_2: minimum direction value among the dot products with the second "template" vector.
        self.radius_threshold, self.min_radius, self.direction_threshold_1, self.direction_threshold_2, self.min_dir_1, self.min_dir_2 = self.break_vector_space(data)

        # Sets the number of radius buckets to RADIUS_BUCKET_NUM controller.
        self.num_of_radius_buckets = RADIUS_BUCKET_NUM
        # Sets the number of direction buckets to DIRECTION_BUCKET_NUM controller.
        self.num_of_direction_buckets = DIRECTION_BUCKET_NUM

        # Creates an array -- hash grid that breaks the vector space into RADIUS_BUCKET_NUM by DIRECTION_BUCKET_NUM
        # buckets -- to be used to store the vectors that are in vicinity to each other.
        self.array = [[[] for i in range(self.num_of_direction_buckets)] for i in range(self.num_of_radius_buckets)]

    def break_vector_space(self, data):
        """The first part of the LPH algorithm that breaks the vector space into buckets using 1 radius and 2 direction
        coordinates for finding the corresponding buckets. While the radius coordinate is found directly through the
        radius of the vector itself, the direction coordinates are found through the dot products of the vector
        with 2 perpendicular to each other "template" vectors that the algorithm entails."""
        print("\nBreaking the vector space:")
        # Creates max_radius abd min_radius variables and assigns the radius of the first vector to them
        # in order to make comparisons of radii exclusively among the data set vectors.
        max_radius = self.radius(data[0])
        min_radius = max_radius

        # Holder for all radius values.
        radii_list = []

        # Will not slice data[1:] because the complexity would be O(n-1), where n = len(data).
        for vector in data:
            # Calculates the radius of the given vector.
            current_radius = self.radius(vector)
            # Appends the current_radius to the radii_list.
            radii_list.append(current_radius)
            # Updates the min and max radii.
            if current_radius < min_radius:
                min_radius = current_radius
            elif current_radius > max_radius:
                max_radius = current_radius
        print(" " * 4 + "Radius range: [{}, {}]".format(min_radius, max_radius))

        # Divides the radius range into RADIUS_BUCKET_NUM buckets to find the radius threshold length.
        radius_threshold = (max_radius - min_radius) / RADIUS_BUCKET_NUM
        print(" " * 4 + "Radius threshold = {}".format(radius_threshold))

        # Radius threshold along two direction thresholds complete the hashing grid, and together they create a
        # granularity of buckets.

        # From the radially middle bucket, we finalize the granularity to be applied throughout the vector space:
        # we think that the middle bucket size provides a good average granularity for the vector space,
        # including the most radially closest and distant parts of the space.
        middle_bucket = []
        # Calculates the multiple-threshold length, above which the middle radial bucket is placed.
        middle_bucket_min_radius = max_radius - (RADIUS_BUCKET_NUM // 2) * radius_threshold
        # Calculates the multiple-threshold length, below which the middle radial bucket is placed.
        middle_bucket_max_radius = middle_bucket_min_radius + radius_threshold

        # Places all vectors that belong in the middle bucket into the bucket itself.
        for i in range(len(radii_list)):
            if radii_list[i] > middle_bucket_min_radius and radii_list[i] < middle_bucket_max_radius:
                middle_bucket.append(data[i])

        # Similarly, calculate the max and min values for the range of direction values, found as
        # dot products between the training vectors and the first/second "template" vectors.

        # Creates max and min variables for each direction coordinates and assigns the values from the dot products
        # with the first vector in order to make comparisons of values exclusively among the data set vectors.
        direction_coordinate_1_max = self.dot_product(middle_bucket[0], self.template_vector_1)
        direction_coordinate_1_min = direction_coordinate_1_max

        direction_coordinate_2_max = self.dot_product(middle_bucket[0], self.template_vector_2)
        direction_coordinate_2_min = direction_coordinate_2_max

        # For each vector in the middle bucket, finds the two direction coordinates as dot products with the two
        # "template" vectors.
        for vector in middle_bucket:
            direction_coordinate_1 = self.dot_product(vector, self.template_vector_1)
            direction_coordinate_2 = self.dot_product(vector, self.template_vector_2)

            # Updates the max and min coordinates for each direction range.
            if direction_coordinate_1 > direction_coordinate_1_max:
                direction_coordinate_1_max = direction_coordinate_1
            elif direction_coordinate_1 < direction_coordinate_1_min:
                direction_coordinate_1_min = direction_coordinate_1

            if direction_coordinate_2 > direction_coordinate_2_max:
                direction_coordinate_2_max = direction_coordinate_2
            elif direction_coordinate_2 < direction_coordinate_2_min:
                direction_coordinate_2_min = direction_coordinate_2

        print(" " * 4 + "Direction 1 range: [{}, {}]".format(direction_coordinate_1_min, direction_coordinate_1_max))
        print(" " * 4 + "Direction 2 range: [{}, {}]".format(direction_coordinate_2_min, direction_coordinate_2_max))

        # Calculates the threshold lengths for finding the coordinates of direction buckets,
        # setting the granularity for the hash-grid.
        # In doing so, divides the direction value range into two sections,
        # creating 4 direction buckets for each of the 10 radius buckets.
        # Changing the divisor to be more or less than 2 changes the distribution of vectors into buckets,
        # made visible by using the print method of the KnnHashTable class.
        direction_threshold_1 = (direction_coordinate_1_max - direction_coordinate_1_min) / (DIRECTION_BUCKET_NUM / 2)
        direction_threshold_2 = (direction_coordinate_2_max - direction_coordinate_2_min) / (DIRECTION_BUCKET_NUM / 2)
        print(" " * 4 + "Direction 1 threshold = {}".format(direction_threshold_1))
        print(" " * 4 + "Direction 2 threshold = {}".format(direction_threshold_2))

        # Returns the "relevant" elements.
        return radius_threshold, min_radius, direction_threshold_1, direction_threshold_2, direction_coordinate_1_min, direction_coordinate_2_min

    def dot_product(self, vector_i, vector_j):
        """Finds the dot product between two vectors."""
        # Accumulator sum for the products of the vectors' components.
        a_sum = 0
        # For each component of the vector, finds its product with a corresponding component of the other vector.
        for k in range(len(vector_i)):
            # Adds the product to the accumulator variable.
            a_sum += vector_i[k] * vector_j[k]
        # Returns the accumulator variable, that is, the dot product.
        return a_sum

    def radius(self, vector):
        """Finds the radius of a vector."""
        # Accumulator sum for the squares of the vector's components.
        a_sum = 0
        # For each component, finds its square.
        for i in vector:
            # Adds the product to the accumulator variable.
            a_sum += i**2
        # Finds the radius by taking the square root of the accumulator variable.
        radius = math.sqrt(a_sum)
        # Returns the radius.
        return radius

    def add(self, vector, label):
        """Adds a new vector into the hash grid using the hashing function."""
        # Finds the radius bucket.
        radius_bucket = self.radius_hash(vector)
        # Find the direction bucket.
        direction_bucket = self.direction_hash(vector)
        # Appends the vector to the array using the bucket-indices.
        self.array[radius_bucket][direction_bucket].append((vector, label))
        return

    def radius_hash(self, vector):
        """Finds the radius bucket of a vector."""
        # Finds the radius of the vector.
        vector_radius = self.radius(vector)
        # Find the radius bucket using the radius threshold.
        bucket_num = int((vector_radius - self.min_radius) // self.radius_threshold)
        # Due to truncation and division of decimal numbers, sometimes we get values that are just above the
        # max multiple of the radius threshold. In that case, assigns such vectors to the most distant bucket,
        # making it a bucket of vectors of radii = min_radius + BUCKET_PORTION * radius_threshold OR bigger.
        if bucket_num == RADIUS_BUCKET_NUM:
            bucket_num -= 1
        # Returns the radius bucket.
        return bucket_num

    def direction_hash(self, vector):
        """Finds the direction bucket of a vector."""
        # Find the dot products of the vector with the "template" vectors.
        vector_direction_coordinate_1 = self.dot_product(vector, self.template_vector_1)
        vector_direction_coordinate_2 = self.dot_product(vector, self.template_vector_2)
        # Finds the direction buckets in each of the directions, where in each direction a vector can
        # be mapped into only one of the two sections, divided by the direction thresholds.
        bucket_x = (vector_direction_coordinate_1 - self.min_dir_1) // self.direction_threshold_1
        bucket_y = (vector_direction_coordinate_2 - self.min_dir_2) // self.direction_threshold_2

        # Due to the granularity becoming too small for some of the radius buckets, some vectors get mapped into
        # the 3rd or 4th section of a direction. In such a case, similar how it was with for the radius buckets,
        # such vectors get assigned to the 1st section, or a direction coordinate of 1.
        # If the section is negative then section 0 is assigned as a direction coordinate to the vector.
        if bucket_x >= (DIRECTION_BUCKET_NUM / 2):
            bucket_x = 1
        elif bucket_x < 0:
            bucket_x = 0
        if bucket_y >= (DIRECTION_BUCKET_NUM / 2):
            bucket_y = 1
        elif bucket_y < 0:
            bucket_y = 0

        # Since in each direction the coordinates can be either 1 or 0, we have 4 buckets, encoded with a binary pair.
        if bucket_x == 0 and bucket_y == 0:
            bucket_num = 0
        elif bucket_x == 0 and bucket_y == 1:
            bucket_num = 1
        elif bucket_x == 1 and bucket_y == 0:
            bucket_num = 2
        elif bucket_x == 1 and bucket_y == 1:
            bucket_num = 3
        else:
            # In case the coordinates are not 0 and 1: it was useful for the initial debugging.
            # There shouldn't be such cases now.
            raise ValueError("bucket_num index is out of range with bucket_x = {} and bucket_y = {}".format(bucket_x, bucket_y))
        # Returns the direction bucket.
        return bucket_num

    def get(self, radius_bucket_num, direction_bucket_num):
        """ The second part of the LPH algorithm that takes a vector and gets a part of the vector space around it,
        that is, a list of vectors existing in its vicinity."""
        # Accesses the array -- hash grid -- using the by the radius and direction bucket numbers and gets
        # the corresponding part of the vector space.
        list_of_vectors = self.array[radius_bucket_num][direction_bucket_num]
        # Returns the that part of the vector space -- list of vectors existing in each other's vicinity.
        return list_of_vectors

    def hash(self, vector):
        """Returns the radius and direction bucket numbers of a given vector."""
        return self.radius_hash(vector), self.direction_hash(vector)

    def print(self):
        """Prints out the lengths of elements in the array -- hash grid -- of the KnnHashTable object."""
        # Creates a container list for the lengths of the elements in the array.
        length_list = [[[] for i in range(self.num_of_direction_buckets)] for i in range(self.num_of_radius_buckets)]
        # For each radius bucket:
        for i in range(len(self.array)):
            # For each direction bucket:
            for j in range(len(self.array[i])):
                # Append the length of the current direction bucket to the container.
                length_list[i][j].append(len(self.array[i][j]))

        print("\nTrained hash table's length list:")
        # Print each row, that is, radius bucket of the container.
        for i in length_list:
            print(" " * 4, i)
        return
