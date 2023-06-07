import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def important_data():
    X, y = preprocess()
    # for each cancellation_policy_code find the number of cancellation

    cancellation_policy_code = X['cancellation_policy_code'].unique()
    cancellation_policy_code_hash = {cancellation_policy_code[i]: 0 for i in range(len(cancellation_policy_code))}
    for row in range(len(y)):
        cancellation_policy_code_hash[X['cancellation_policy_code'][row]] += y[row]

    for key in cancellation_policy_code_hash:
        print(key, cancellation_policy_code_hash[key])
    # print max of cancellation_policy_code_hash
    print("max: ", max(cancellation_policy_code_hash, key=cancellation_policy_code_hash.get),
          cancellation_policy_code_hash[max(cancellation_policy_code_hash, key=cancellation_policy_code_hash.get)])
    #a lot cancel while using 1D1N_1 N 2854

def piercing_correlation():
    # make  piercing correlation graph for each feature with y (cancellation)
    X, y = preprocess()
    # for each feature make a plot of the feature with y
    for feature in X:
        print(feature)
        # make a plot of the feature with y


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
    return X_train, y_train


def main():
    X, y = preprocess()




if __name__ == '__main__':
    main()
