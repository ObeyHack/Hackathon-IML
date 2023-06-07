from Preprocess_data import preprocess


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
    #a lot cancel while using 1D1N_1N 2854

def piercing_correlation():
    # make  piercing correlation graph for each feature with y (cancellation)
    X, y = preprocess()
    # for each feature make a plot of the feature with y
    for feature in X:
        print(feature)
        # make a plot of the feature with y
