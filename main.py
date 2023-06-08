from sys import argv
import agoda_cancellation_prediction
from agoda_cancellation_estimator import AgodaCancellationEstimator

if __name__ == '__main__':
    # block 1
    filename = argv[1]
    X_train, y_train, h_booking_id_save = agoda_cancellation_prediction.load_data(filename)
    model = AgodaCancellationEstimator().fit(X_train, y_train)


    # block 2
    test_filename = argv[2]
    X_train, y_train, h_booking_id_save = agoda_cancellation_prediction.load_data(filename)