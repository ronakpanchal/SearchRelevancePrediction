import numpy
from pylab import *
from LearningToRank import calculate_basis_matrix
from LearningToRank import calculate_covarincematrix

M = 5

count = 0  # row counter
file = open("input.csv")
numline = len(file.readlines())


feature_matrix = numpy.zeros(shape=(numline, 10))  # training set has 46 features in total
target_list = list()  # this list will store the data of target values ,which are discrete in nature


with open("input.csv", "r") as f:
    for line in f:
        feature_array = line.split(',')
        feature_values = list()
        for i in range(0, 10):
            if i == 5:
                feature_array[i].rstrip('\n')
            feature_values.append(feature_array[i])
        feature_matrix[count] = feature_values
        count += 1

with open("output.csv", "r") as f:
    for line in f:
        target_list.append(int(line))
    target_matrix = numpy.array(target_list)

    target_matrix = target_matrix.astype(int)

    size_training_data = round(numline * 0.80)
    training_data = feature_matrix[0:size_training_data, :]
    validation_data = feature_matrix[size_training_data:round(size_training_data + (numline*0.10)), :]
    testing_data = feature_matrix[round(numline*0.90):numline-1, :]

    # calculate covariance matrix
    covariance_matrix = calculate_covarincematrix(training_data, 10)
    inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)

    # select m random rows as center of basis function
    random_rows = numpy.zeros(shape=(M, 10))
    for index in range(0, M):
        row_index = randint(0, size_training_data - 1)
        random_rows[index] = training_data[row_index, :]

    matrix_basis = calculate_basis_matrix(training_data, random_rows, inverse_covariance_matrix, size_training_data, M)

    e_rms_values = list()
    lambda_val = 0
    lambda_values = list()
    min_rms = sys.maxsize
    min_rms_weight_matrix = numpy.zeros(shape=(M, 1))
    min_rms_lambda = 0
    weight_matrix = numpy.zeros(shape=(M, 1))
    size_validation_data = len(validation_data)

    # ERMS calculation for training data
    for j in range(0, 70):
        mat_c = lambda_val * numpy.eye(M)
        phi_mat = numpy.dot(matrix_basis.T, matrix_basis)
        mat_d = mat_c + phi_mat
        mat_d_inverse = numpy.linalg.inv(mat_d)
        mat_e = numpy.dot(mat_d_inverse, matrix_basis.T)
        actual_values = target_matrix[0:size_training_data - 1]
        weight_matrix = numpy.dot(mat_e, actual_values)
        error_sum = 0
        for l in range(0, size_training_data):
            data_vector = training_data[l]
            basis_mat = list()
            for i in range(0, M):
                vector_a = data_vector - random_rows[i]
                vector_b = numpy.dot(vector_a, inverse_covariance_matrix)
                vector_c = numpy.dot(vector_b, vector_a.T)
                basis_mat.append(numpy.exp(-0.5 * vector_c))
            basis_mat = numpy.array(basis_mat)
            predicted_value = numpy.dot(weight_matrix.T, basis_mat)
            # print('predicted value: ', predicted_value, 'actual value', target_matrix[l + size_training_data])
            error_sum += (target_matrix[l] - predicted_value) ** 2
        error_sum *= 0.5
        lambda_values.append(lambda_val)
        lambda_val += 5

        error_sum += (round(lambda_val, 1) * 0.5) * numpy.dot(weight_matrix.T, weight_matrix)
        term = (2 * error_sum) / size_training_data
        e_rms_value = math.sqrt(term)
        e_rms_values.append(e_rms_value)
        if e_rms_value < min_rms:
            min_rms = e_rms_value
            min_rms_weight_matrix = weight_matrix
            min_rms_lambda = lambda_val
    print('minimum e_rms value for synthetic data training set is ', min_rms)

    # ERMS calculation for validation  data

    e_rms_values = list()
    lambda_values = list()
    min_rms = sys.maxsize
    min_rms_weight_matrix = numpy.zeros(shape=(M, 1))
    min_rms_lambda = 0
    lambda_val = 0
    weight_matrix = numpy.zeros(shape=(M, 1))
    covariance_matrix = calculate_covarincematrix(validation_data, 10)
    inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)
    for j in range(0, 70):
        mat_c = lambda_val * numpy.eye(M)
        phi_mat = numpy.dot(matrix_basis.T, matrix_basis)
        mat_d = mat_c + phi_mat
        mat_d_inverse = numpy.linalg.inv(mat_d)
        mat_e = numpy.dot(mat_d_inverse, matrix_basis.T)
        actual_values = target_matrix[0:size_training_data - 1]
        weight_matrix = numpy.dot(mat_e, actual_values)
        error_sum = 0
        for l in range(0, size_validation_data - 1):
            data_vector = validation_data[l]
            basis_mat = list()
            for i in range(0, M):
                vector_a = data_vector - random_rows[i]
                vector_b = numpy.dot(vector_a, inverse_covariance_matrix)
                vector_c = numpy.dot(vector_b, vector_a.T)
                basis_mat.append(numpy.exp(-0.5 * vector_c))
            basis_mat = numpy.array(basis_mat)
            predicted_value = numpy.dot(weight_matrix.T, basis_mat)
            # print('predicted value: ', predicted_value, 'actual value', target_matrix[l + size_training_data])
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

    print('minimum e_rms value for synthetic data validation set is ', min_rms)

    # testing set calculation
    lambda_val = 0
    basis_mat = list()
    size_testing_data = len(testing_data)
    covariance_matrix = calculate_covarincematrix(testing_data, 10)
    inverse_covariance_matrix = numpy.linalg.inv(covariance_matrix)
    starting_index = size_training_data + size_validation_data + 2
    for h in range(0, size_testing_data):
        data_vector = testing_data[h]
        for i in range(0, M):
            vector_a = data_vector - random_rows[i]
            vector_b = numpy.dot(vector_a, inverse_covariance_matrix)
            vector_c = numpy.dot(vector_b, vector_a.T)
            basis_mat.append(numpy.exp(-0.5 * vector_c))
        basis_mat = numpy.array(basis_mat)
        predicted_value = numpy.dot(weight_matrix.T, basis_mat)
        actual_value = target_matrix[starting_index + h-1]
        error_sum += (actual_value - predicted_value) ** 2
        basis_mat = list()
        error_sum *= 0.5
        lambda_values.append(lambda_val)
        lambda_val += 5

    error_sum += (round(lambda_val, 1) * 0.5) * numpy.dot(weight_matrix.T, weight_matrix)
    term = (2 * error_sum) / size_testing_data
    e_rms_value = math.sqrt(term)
    print('e_rms value for testing data is ', e_rms_value)

