from sys import argv
from Preprocess_data import preprocess_test_reindex
from agoda_prediction import evaluate_and_export, load_data
from agoda_cancellation_estimator import AgodaCancellationEstimator
from agoda_selling_amount_estimator import AgodaSellingAmountEstimator


#  commend line args: data\\agoda_cancellation_train.csv data\\Agoda_Test_1.csv data\\Agoda_Test_2.csv

if __name__ == '__main__':
    # block 1
    filename = argv[1]
    X_train_1, y_train_1, _ = load_data(filename, is_cancellation=True)
    model_cancellation_1 = AgodaCancellationEstimator().fit(X_train_1, y_train_1)
    X_train_2, y_train_2, _ = load_data(filename, is_cancellation=False)
    model_cancellation_2 = AgodaSellingAmountEstimator().fit(X_train_2, y_train_2)

    # block 2
    test_filename_1 = argv[2]
    X_test_1, _, h_booking_id = load_data(test_filename_1, is_cancellation=True)
    X_test_1 = preprocess_test_reindex(X_train_1, X_test_1)
    evaluate_and_export(model_cancellation_1, X_test_1, h_booking_id, "task1.csv", is_cancellation=True)

    test_filename_2 = argv[3]
    X_test_2, _, h_booking_id = load_data(test_filename_2, is_cancellation=False)
    X_test_2 = preprocess_test_reindex(X_train_2, X_test_2)
    evaluate_and_export(model_cancellation_2, X_test_2, h_booking_id, "task2.csv", is_cancellation=False)

    # block 3
