
import numpy
import math


def calculate_covarincematrix(data_set, no_of_features):
    sigma = numpy.zeros(shape=(no_of_features, no_of_features))
    for ind in range(0, no_of_features):
        variance = numpy.var(data_set[0:len(data_set) - 1, ind])
        if variance == 0:
            sigma[ind][ind] = 0.001
        else:
            sigma[ind][ind] = variance/10
    return sigma


def calculate_basis_matrix(training_set, mu_rows, covariance_matrix_inverse, training_data_size, no_of_basis_functions):
    basis_matrix = numpy.zeros(shape=(training_data_size-1, no_of_basis_functions))
    for p in range(0, training_data_size-1):
        for index_row in range(0, no_of_basis_functions):
            if index_row == 0:
                basis_matrix[p][index_row] = 1
            else:
                diff_vector = training_set[p] - mu_rows[index_row]
                mat_a = numpy.dot(diff_vector, covariance_matrix_inverse)
                mat_b = numpy.dot(mat_a, diff_vector.T)
                scalar_value = numpy.exp(-0.5*mat_b)
                basis_matrix[p][index_row] = scalar_value
    return basis_matrix


def calculate_least_error_values(matrix_basis_training, m, target_matrix, size):

    lambda_val = 0.1
    lambda_values = list()
    e_rms_values = list()
    min_rms = 1000
    min_rms_weight_matrix = numpy.zeros(shape=(m, 1))

    for j in range(0, 10):
        mat_c = lambda_val * numpy.eye(m)
        transpose_base_matrix = matrix_basis_training.T
        phi_mat = numpy.dot(transpose_base_matrix, matrix_basis_training)
        mat_d = mat_c + phi_mat
        mat_d_inverse = numpy.linalg.inv(mat_d)
        mat_e = numpy.dot(mat_d_inverse, matrix_basis_training.T)
        actual_values = target_matrix
        weight_matrix = numpy.dot(mat_e, actual_values)
        error_sum = 0
        for l in range(0, size - 1):
            basis_mat = matrix_basis_training[l]
            predicted_value = numpy.dot(weight_matrix.T, basis_mat)
            error_sum += (target_matrix[l] - predicted_value) ** 2
        error_sum *= 0.5
        lambda_values.append(lambda_val)
        lambda_val += 0.1

        error_sum += (round(lambda_val, 1) * 0.5) * numpy.dot(weight_matrix.T, weight_matrix)
        term = (2 * error_sum) / size
        e_rms_value = math.sqrt(term)

        e_rms_values.append(e_rms_value)

        if e_rms_value < min_rms:
            min_rms = e_rms_value
            min_rms_weight_matrix = weight_matrix
            min_rms_lambda = lambda_val
    return min_rms_weight_matrix, min_rms_lambda, min_rms


def get_sum_of_squared_errors(data_training, target_data, weights_matrix, basis_matrix):
    sum_of_errors = 0
    for counter in range(0, len(data_training) - 1):
        real_value = target_data[counter]
        value_predicted = numpy.dot(weights_matrix.T, basis_matrix[counter])
        difference = real_value - value_predicted
        sum_of_errors += (difference ** 2)
    sum_of_errors *= 0.5
    return sum_of_errors
