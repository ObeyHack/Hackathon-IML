import pandas as pd

CATEGORICAL_FEATURES = ["charge_option", "customer_nationality",
                        "guest_nationality_country_name", "origin_country_code"
                       "language", "original_payment_method", "original_payment_type"]

def preprocess():
    # Read data
    data = pd.read_csv('data\\agoda_cancellation_train.csv')
    # Split the data into X and y
    X_train = data.drop('cancellation_datetime', axis=1)
    y_train = data['cancellation_datetime']
    # binary the y_train
    y_train = y_train.apply(lambda x: 0 if pd.isnull(x) else 1)





    # make date to datetime
    X_train['checkin_date'] = pd.to_datetime(X_train['checkin_date'])
    X_train['checkout_date'] = pd.to_datetime(X_train['checkout_date'])
    X_train['booking_datetime'] = pd.to_datetime(X_train['checkout_date'])


    # h_booking_id - delete from train, save from output
    h_booking_id_save = X_train['h_booking_id']
    X_train = X_train.drop('h_booking_id', axis=1)

    # booking_datetime - parse_dates=["booking_datetime"] in read_csv DONE IN LINE 23
    # checkin_date - parse_dates=["checkin_date"] in read_csv DONE IN LINE 21
    # checkout_date - delete from train, will be used to calculate stay_duration
    X_train["stay_duration"] = (X_train['checkout_date'] - X_train['checkin_date']).days


    return X_train, y_train


def main():
    X, y = preprocess()




if __name__ == '__main__':
    main()
