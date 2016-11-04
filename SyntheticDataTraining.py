import numpy

from common_functions import *
import random
import re

M = 12


def calculate_values_synthetic_data():
    count = 0  # row counter
    file = open("input.csv")
    numline = len(file.readlines())

    feature_matrix = numpy.zeros(shape=(numline, 10))  # training set has 10 features in total
    target_list = list()  # this list will store the data of target values ,which are discrete in nature

    with open("input.csv", "r") as f:
        for line in f:
            feature_array = line.strip().split(',')
            feature_values = list()
            for i in range(0, 10):
                number = re.sub('[^0-9]', '', feature_array[i])
                feature_values.append(number)
            feature_matrix[count] = feature_values
            count += 1

    with open("output.csv", "r") as f:
        for line in f:
            num = re.sub('[^0-9]', '', line)
            target_list.append(int(num))
        target_matrix = numpy.array(target_list)
        target_matrix = target_matrix.astype(int)

        '''
        new_array = numpy.column_stack((feature_matrix, target_matrix))
        numpy.random.shuffle(new_array)
        feature_matrix = new_array[:, :len(new_array[0]) - 1]
        target_matrix = new_array[:, len(new_array[0]) - 1]
        '''

        size_training_data = int(numline * 0.80)
        training_data = feature_matrix[0:size_training_data, :]
        validation_data = feature_matrix[size_training_data:size_training_data + round(numline * 0.10), :]
        testing_data = feature_matrix[round(numline * 0.9):numline, :]
        size_validation_data = len(validation_data)
        size_testing_data = len(testing_data)

        # select m random rows as center of basis function
        random_rows = numpy.zeros(shape=(M, 10))
        for index in range(0, M):
            # row_index = randint(0, size_training_data - 1)
            row_index = random.randrange(0, size_training_data - 1)
            random_rows[index] = training_data[row_index, :]

        covariance_matrix_training = calculate_covarincematrix(training_data, 10)
        # print('design matrix for training data ', covariance_matrix_training)
        inverse_covariance_matrix_training = numpy.linalg.inv(covariance_matrix_training)
        basis_matrix_training = calculate_basis_matrix(training_data, random_rows, inverse_covariance_matrix_training,
                                                       size_training_data, M)

        weights_training, lambda_training, e_rms_training = calculate_least_error_values(basis_matrix_training, M,
                                                                                         target_matrix[
                                                                                         0:size_training_data - 1],
                                                                                         len(training_data),'synthetic', 'training')
        print('Values for Synthetic data set ')
        print('-----------------------------------------------------------')
        print('Minimum ERMS value for training  data is ', e_rms_training)

        covariance_matrix = calculate_covarincematrix(validation_data, 10)
        inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)
        # print('design matrix for validation  data ', covariance_matrix)
        basis_matrix_validation = calculate_basis_matrix(validation_data, random_rows, inverse_covariance_matrix,
                                                         len(validation_data), M)
        target_matrix_validation = target_matrix[size_training_data:size_training_data + len(validation_data) - 1]
        weights_validation, lambda_validation, e_rms_validation = calculate_least_error_values(basis_matrix_validation, M,
                                                                                               target_matrix_validation,
                                                                                               len(validation_data), 'synthetic', 'validation')
        print('Minimum ERMS value for validation data is ', e_rms_validation)
        # testing set calculation

        lambda_val = lambda_validation
        lambda_values = list()
        starting_index = len(training_data) + len(validation_data)
        covariance_matrix = calculate_covarincematrix(testing_data, 10)
        # print('design matrix for testing  data ', covariance_matrix)
        inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)
        matrix_basis_testing = calculate_basis_matrix(testing_data, random_rows, inverse_covariance_matrix,
                                                      len(testing_data), M)
        weight_matrix = weights_validation
        error_sum = 0

        for h in range(0, size_testing_data - 1):
            data_vector = testing_data[h]
            basis_mat = matrix_basis_testing[h]
            predicted_value = numpy.dot(weight_matrix.T, basis_mat)
            actual_value = target_matrix[starting_index + h]
            error_sum += (actual_value - predicted_value) ** 2
            basis_mat = list()
        error_sum *= 0.5
        error_sum += (lambda_val * 0.5) * numpy.dot(weight_matrix.T, weight_matrix)
        term = (2 * error_sum) / size_testing_data
        e_rms_value = math.sqrt(term)
        print('Minimum ERMS value for testing  data is ', e_rms_value)

        minimum_e_rms_training = calculate_sgd_values(training_data, target_matrix[0:len(training_data)-1], basis_matrix_training, M)
        print('Minimum e_rms value for SGD over training set is ', minimum_e_rms_training)

        minimum_e_rms_validation = calculate_sgd_values(validation_data, target_matrix_validation, basis_matrix_validation, M)
        print('Minimum e_rms value for SGD over validation set is ', minimum_e_rms_validation)

        target_matrix_testing = target_matrix[starting_index:len(target_matrix)-1]
        minimum_e_rms_testing = calculate_sgd_values(testing_data, target_matrix_testing, matrix_basis_testing, M)
        print('Minimum e_rms value for SGD over testing set  is ', minimum_e_rms_testing)
        print('------------------------------------------------------------------')