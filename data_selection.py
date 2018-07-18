import sys
import csv
import random
import numpy as np


def data_selection(file_name):
    with open(str(file_name), 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=';')
        header = next(datareader)
        data = []

        for row in datareader:
            row_of_floats = list(map(float, row))
            data.append(row_of_floats)

    # data is  of type list
    data_as_array = np.array(data)

    with open('all_data.csv', 'w') as csvfile4:
        writer = csv.writer(csvfile4, delimiter=',')
        writer.writerow(header)
        for num in range(0, len(data)):
            writer.writerow(data[num])

    training_data_list = []
    test_data_list = []
    count = 0

    np.random.seed(5)

    while (count < (0.9 * 1599)):
        x = random.choice(data)
        training_data_list.append(x)
        data.remove(x)
        count = count + 1

    while (count >= 0.9 * 1599 and count < 1599):
        z = random.choice(data)
        test_data_list.append(z)
        data.remove(z)
        count = count + 1

    training_data = np.array(training_data_list)
    test_data = np.array(test_data_list)

    # writing the training data to CSV
    with open('final_training_data.csv', 'w') as csvfile2:
        writer = csv.writer(csvfile2, delimiter=',')
        writer.writerow(header)
        for num in range(0, len(training_data)):
            writer.writerow(training_data[num])

    # writing the validation data to CSV
    with open('final_test_data.csv', 'w') as csvfile4:
        writer = csv.writer(csvfile4, delimiter=',')
        writer.writerow(header)
        for num in range(0, len(test_data)):
            writer.writerow(test_data[num])

    # writing the validation data to CSV
