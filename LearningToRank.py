
import numpy
from pylab import *
import math
from random import randint

M = 5
'''
This method calculates the covariance matrix of training data features
'''


def calculate_covarincematrix(training_set, no_of_features):
    sigma = numpy.zeros(shape=(no_of_features, no_of_features))
    for ind in range(0, no_of_features):
        variance = numpy.var(training_set[0:len(training_set)-1, ind])
        sigma[ind][ind] = variance/10
    return sigma


def calculate_basis_matrix(training_set, mu_rows, covariance_matrix_inverse, training_data_size, no_of_basis_functions):
    basis_matrix = numpy.zeros(shape=(training_data_size-1, no_of_basis_functions))
    for k in range(0, training_data_size-1):
        for index_row in range(0, no_of_basis_functions):
            diff_vector = training_set[k] - mu_rows[index_row]
            mat_a = numpy.dot(diff_vector, covariance_matrix_inverse)
            mat_b = numpy.dot(mat_a, diff_vector.T)
            scalar_value = numpy.exp(-0.5*mat_b)
            basis_matrix[k][index_row] = scalar_value
    return basis_matrix
'''
    This code will read data from Querylevelnorm file which contains the date for training the model
'''
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
    '''
     Assign very small values to columns which have 0 values all through out the column
     there are 6 such columns in total.
     This will prevent singular matrix creation,while calculating the covariance matrix for features.
    '''

    feature_matrix[:, 5] = 0.0001
    feature_matrix[:, 6] = 0.0001
    feature_matrix[:, 7] = 0.0001
    feature_matrix[:, 8] = 0.0001
    feature_matrix[:, 9] = 0.0001
    feature_matrix[:, 45] = 0.0001

    '''
    Get the first 80% rows of data for training  model
    '''
    size_training_data = round(numline*0.80)
    training_data = feature_matrix[0:size_training_data-1, :]
    validation_data = feature_matrix[size_training_data:size_training_data+round(numline*0.10), :]
    testing_data = feature_matrix[round(numline*0.9):numline-1, :]
    size_validation_data = len(validation_data)
    size_testing_data = len(testing_data)

    target_matrix = numpy.array(target_list)
    target_matrix = target_matrix.astype(int)
    covariance_matrix = calculate_covarincematrix(training_data, 46)
    inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)

    '''
    initially pick up M as 6
    M is the number of basis functions
    pick 6 random rows
    store the rows in numpy array
    '''
    random_rows = numpy.zeros(shape=(M, 46))
    for index in range(0, M):
        row_index = randint(0, size_training_data-1)
        random_rows[index] = training_data[row_index, :]

    matrix_basis = calculate_basis_matrix(training_data, random_rows, inverse_covariance_matrix, size_training_data, M)
    lambda_val = 0

    '''
    trying different values of lambda between 0 and 1 with variable increments in lambda value
    '''
    e_rms_values = list()
    lambda_values = list()
    min_rms = sys.maxsize
    min_rms_weight_matrix = numpy.zeros(shape=(M, 1))
    min_rms_lambda = 0
    weight_matrix = numpy.zeros(shape=(M, 1))
    covariance_matrix = calculate_covarincematrix(validation_data, 46)
    inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)
    for j in range(0, 70):
            mat_c = lambda_val*numpy.eye(M)
            phi_mat = numpy.dot(matrix_basis.T, matrix_basis)
            mat_d = mat_c + phi_mat
            mat_d_inverse = numpy.linalg.inv(mat_d)
            mat_e = numpy.dot(mat_d_inverse, matrix_basis.T)
            actual_values = target_matrix[0:size_training_data - 1]
            weight_matrix = numpy.dot(mat_e, actual_values)
            error_sum = 0
            for l in range(0, size_validation_data-1):
                data_vector = validation_data[l]
                basis_mat = list()
                for i in range(0, M):
                    vector_a = data_vector - random_rows[i]
                    vector_b = numpy.dot(vector_a, inverse_covariance_matrix)
                    vector_c = numpy.dot(vector_b, vector_a.T)
                    basis_mat.append(numpy.exp(-0.5*vector_c))
                basis_mat = numpy.array(basis_mat)
                predicted_value = numpy.dot(weight_matrix.T, basis_mat)
                error_sum += (target_matrix[l + size_training_data] - predicted_value) ** 2
            error_sum *= 0.5
            lambda_values.append(lambda_val)
            lambda_val += 5

            error_sum += (round(lambda_val, 1) * 0.5) * numpy.dot(weight_matrix.T, weight_matrix)
            term = (2 * error_sum) / size_validation_data
            e_rms_value = math.sqrt(term)
            e_rms_values.append(e_rms_value)
            if e_rms_value < min_rms:
                min_rms = e_rms_value
                min_rms_weight_matrix = weight_matrix
                min_rms_lambda = lambda_val
    print('Minimum E_RMS value for Validation Set  is ', min_rms)

    figure()
    plot(lambda_values, e_rms_values, 'r')
    xlabel('Regularization Parameter')
    ylabel('E_RMS')
    title('Regularization parameter VS E_RMS')
    # show()
    e_rms_array = [0.656, 0.582, 0.578, 0.575, 0.579, 0.578, 0.575, 0.580, 0.581, 0.580]
    basis_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    figure()
    plot(basis_values, e_rms_array, 'r')
    xlabel('Number of basis functions')
    ylabel('E_RMS')
    title('Number of basis functions VS E_RMS')
    # show()

    basis_mat = list()
    covariance_matrix = calculate_covarincematrix(testing_data, 46)
    inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)
    starting_index = size_training_data + size_validation_data + 2
    # test set calculation
    weight_matrix = min_rms_weight_matrix
    for h in range(0, size_testing_data):
        data_vector = testing_data[h]
        for i in range(0, M):
            vector_a = data_vector - random_rows[i]
            vector_b = numpy.dot(vector_a, inverse_covariance_matrix)
            vector_c = numpy.dot(vector_b, vector_a.T)
            basis_mat.append(numpy.exp(-0.5 * vector_c))
        basis_mat = numpy.array(basis_mat)
        predicted_value = numpy.dot(weight_matrix.T, basis_mat)
        actual_value = target_matrix[starting_index+h]
        error_sum += (actual_value - predicted_value) ** 2
        basis_mat = list()
        error_sum *= 0.5
        lambda_values.append(lambda_val)

    error_sum += (round(lambda_val, 1) * 0.5) * numpy.dot(weight_matrix.T, weight_matrix)
    term = (2 * error_sum) / size_testing_data
    e_rms_value = math.sqrt(term)
    print('E_RMS value for Testing  Set is ', e_rms_value)

    '''
    # SGD calculation starts here
    lambda_sgd = 0.1
    sgd_weight_matrix = numpy.random.random(M)
    sgd_weight_matrix = numpy.array(sgd_weight_matrix)
    learning_rate = 0.001
    last_delta_E = 1000
    for k in range(0, size_training_data-1):
        for i in range(0, M):
            vector_a = training_data[k] - random_rows[i]
            vector_b = numpy.dot(vector_a, inverse_covariance_matrix)
            vector_c = numpy.dot(vector_b, vector_a.T)
            basis_mat.append(numpy.exp(-0.5 * vector_c))
        basis_mat = numpy.array(basis_mat)
        delta_E = target_matrix[k] - numpy.dot(sgd_weight_matrix.T, basis_mat)
        delta_E = delta_E * basis_mat
        delta_E = numpy.linalg.norm(delta_E)
        print('delta E value is ', delta_E)
        if last_delta_E - delta_E < 0.001:
            break
        if last_delta_E - delta_E < 0:
                learning_rate *= 10
        last_delta_E = delta_E
        # delta_E = delta_E * basis_mat
        delta_E_d = delta_E+sgd_weight_matrix.T * lambda_sgd
        sgd_weight_matrix_updated = sgd_weight_matrix + (-learning_rate * delta_E_d)
        sgd_weight_matrix = sgd_weight_matrix_updated
        basis_mat = list()
    print('sgd weight matrix is ', sgd_weight_matrix)
    '''
