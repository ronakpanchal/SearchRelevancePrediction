

from random import randint
from common_functions import *

M = 10


def calculate_values_letor():
    count = 0  # row counter
    file = open("Querylevelnorm.txt")
    numline = len(file.readlines())
    feature_matrix = numpy.zeros(shape=(numline, 46))  # training set has 46 features in total
    target_list = list()  # this list will store the data of target values ,which are discrete in nature

    with open("Querylevelnorm.txt", "r") as f:
        for line in f:
            feature_arr = line.split(' ')
            target_list.append(feature_arr[0])
            feature_values = list()
            for i in range(2, 48):
                feature_val = feature_arr[i].split(':')
                feature_values.append(feature_val[1])
            feature_matrix[count] = feature_values
            count += 1

        # shuffle the features data
        # numpy.random.permutation(feature_matrix)
        target_matrix = numpy.array(target_list)
        target_matrix = target_matrix.astype(int)
        '''
           Get the first 80% rows of data for training  model
        '''
        new_array = numpy.column_stack((feature_matrix, target_matrix))
        numpy.random.shuffle(new_array)
        feature_matrix = new_array[:, :len(new_array[0])-1]
        target_matrix = new_array[:, len(new_array[0])-1]
        size_training_data = int(int(numline * 0.80))
        training_data = feature_matrix[0:size_training_data, :]
        validation_data = feature_matrix[size_training_data:size_training_data + round(numline * 0.10), :]
        testing_data = feature_matrix[round(numline * 0.9):numline, :]
        size_validation_data = len(validation_data)
        size_testing_data = len(testing_data)

        random_rows = numpy.zeros(shape=(M, 46))
        for index in range(0, M):
            row_index = randint(0, size_training_data - 1)
            random_rows[index] = training_data[row_index, :]

        covariance_matrix_training = calculate_covarincematrix(training_data, 46)
        inverse_covariance_matrix_training = numpy.linalg.inv(covariance_matrix_training)
        # print('design matrix for training  data ', covariance_matrix_training)
        basis_matrix_training = calculate_basis_matrix(training_data, random_rows,
                                                       inverse_covariance_matrix_training, size_training_data, M)

        weights_training, lambda_training, e_rms_training = calculate_least_error_values(basis_matrix_training, M,
                                                                                         target_matrix[
                                                                                         0:size_training_data - 1],
                                                                                         len(training_data), 'letor', 'training')

        covariance_matrix = calculate_covarincematrix(validation_data, 46)
        inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)
        # print('design matrix for validation  data ', covariance_matrix)
        matrix_basis_validation = calculate_basis_matrix(validation_data, random_rows, inverse_covariance_matrix,
                                                         len(validation_data), M)

        target_matrix_validation = target_matrix[size_training_data:size_training_data + len(validation_data) - 1]
        weights_validation, lambda_validation, e_rms_validation = calculate_least_error_values(matrix_basis_validation,
                                                                                               M,
                                                                                               target_matrix_validation,
                                                                                               len(validation_data), 'letor', 'validation')
        print('Values for LETOR data set ')
        print('--------------------------------------------------------------')
        print('Minimum ERMS value for training  data is ', e_rms_training, 'lambda=', str(round(lambda_training,2)), 'M= ', str(M))
        print('Minimum ERMS value for validation data is ', e_rms_validation, 'lambda=', str(round(lambda_validation,2)), 'M= ', str(M))
        # test set calculation
        starting_index = len(training_data) + len(validation_data)
        covariance_matrix = calculate_covarincematrix(testing_data, 46)
        # print('design matrix for testing  data ', covariance_matrix)
        inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)
        matrix_basis_testing = calculate_basis_matrix(testing_data, random_rows, inverse_covariance_matrix, len(testing_data),
                                              M)
        weight_matrix = weights_validation
        error_sum = 0
        lambda_val = lambda_validation
        for h in range(0, len(testing_data) - 1):
            basis_mat = matrix_basis_testing[h]
            predicted_value = numpy.dot(weight_matrix.T, basis_mat)
            actual_value = target_matrix[starting_index + h]
            error_sum += (actual_value - predicted_value) ** 2
        error_sum *= 0.5
        error_sum += (round(lambda_val, 1) * 0.5) * numpy.dot(weight_matrix.T, weight_matrix)
        term = (2 * error_sum) / size_testing_data
        e_rms_value = math.sqrt(term)
        print('Minimum ERMS value for testing data is ', e_rms_value, 'lambda=', str(round(lambda_validation,2)), 'M= ', str(M))

        # SGD Calculation
        # training data

        minimum_e_rms_training,min_sgd_lambda = calculate_sgd_values(training_data, target_matrix[0:len(training_data)-1], basis_matrix_training, M)
        print('Minimum e_rms value for SGD over training set is ', minimum_e_rms_training, 'at lambda = ', str(round(min_sgd_lambda,2)), 'M=', M)

        minimum_e_rms_validation,min_sgd_lambda = calculate_sgd_values(validation_data, target_matrix_validation, matrix_basis_validation, M)
        print('Minimum e_rms value for SGD over validation set is ', minimum_e_rms_validation, 'at lambda = ',  str(round(min_sgd_lambda,2)), 'M=', M)

        target_matrix_testing = target_matrix[starting_index:len(target_matrix)-1]
        minimum_e_rms_testing,min_sgd_lambda = calculate_sgd_values(testing_data, target_matrix_testing, matrix_basis_testing, M)
        print('Minimum e_rms value for SGD over testing set  is ', minimum_e_rms_testing, 'at lambda = ', str(round(min_sgd_lambda,2)), 'M=', M)
        print('------------------------------------------------------------------')
