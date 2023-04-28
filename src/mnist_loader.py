import pandas as pd
import numpy as np

def load_data():
    training_data_df = pd.read_pickle("./training.pkl")
    validation_data_df = pd.read_pickle("./validation.pkl")
    test_data_df = pd.read_pickle("./testing.pkl")

    print(len(training_data_df))

    training_data = training_data_df["unit_data"].to_list()
    validation_data = validation_data_df["unit_data"].to_list()
    test_data = test_data_df["unit_data"].to_list()

    training_data_labels = training_data_df["symbol_id"].to_list()
    validation_data_labels = validation_data_df["symbol_id"].to_list()
    test_data_labels = test_data_df["symbol_id"].to_list()


    id_map = {183:0,88:1,944:2,82:3,166:4,603:5}

    for i in range(len(training_data_labels)):
        training_data_labels[i] = id_map[training_data_labels[i]]

    for i in range(len(validation_data_labels)):
        validation_data_labels[i] = id_map[validation_data_labels[i]]

    for i in range(len(test_data_labels)):
        test_data_labels[i] = id_map[test_data_labels[i]]

    final_training_data = [training_data, training_data_labels]
    final_validation_data = [validation_data, validation_data_labels]
    final_testing_data = [test_data,test_data_labels]

    return (final_training_data, final_validation_data, final_testing_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (1024, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (1024, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (1024, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((6, 1))
    e[j] = 1.0
    return e


