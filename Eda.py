from matplotlib import pyplot as plt

from Preprocess_data import preprocess
from scipy.stats import pearsonr

L_features = ['no_of_adults', 'no_of_children', 'no_of_room', 'no_of_extra_bed', 'hotel_star_rating',
              'is_first_booking', 'cancellation_policy_code', 'is_user_logged_in', 'guest_nationality_country_name',
              'original_payment_method', 'original_payment_type','charge_option','original_payment_currency','hotel_id']



def important_data():
    X, y = preprocess()
    # for each cancellation_policy_cod e find the number of cancellation

    cancellation_policy_code = X['cancellation_policy_code'].unique()
    cancellation_policy_code_hash = {cancellation_policy_code[i]: 0 for i in range(len(cancellation_policy_code))}
    for row in range(len(y)):
        cancellation_policy_code_hash[X['cancellation_policy_code'][row]] += y[row]

    for key in cancellation_policy_code_hash:
        print(key, cancellation_policy_code_hash[key])
    # print max of cancellation_policy_code_hash
    print("max: ", max(cancellation_policy_code_hash, key=cancellation_policy_code_hash.get),
          cancellation_policy_code_hash[max(cancellation_policy_code_hash, key=cancellation_policy_code_hash.get)])
    # a lot cancel while using 1D1N_1N 2854


def cheack_guest_nationality_country_name(final_vals_for_unique_values, unique_values, sumuniqe):
    print('guest_nationality_country_name up to 50 %')
    # print all contries with 0.75< in final_vals_for_unique_values and number of people in each country
    for i in range(len(final_vals_for_unique_values)):
        if final_vals_for_unique_values[i] > 0.50:
            print(unique_values[i], " number of overall reservation: ", sumuniqe[i], "cancellation probabilty : ",
                  final_vals_for_unique_values[i])

    print('guest_nationality_country_name  under 20 %')

    for i in range(len(final_vals_for_unique_values)):
        if final_vals_for_unique_values[i] < 0.2:
            print(unique_values[i], " number of overall reservation: ", sumuniqe[i], "cancellation probabilty : ",
                  final_vals_for_unique_values[i])

def check_original_payment_currency(final_vals_for_unique_values, unique_values, sumuniqe):
    print('payment_currency up to 50 %')
    # print all contries with 0.5< in final_vals_for_unique_values and number of people in each country
    for i in range(len(final_vals_for_unique_values)):
        if final_vals_for_unique_values[i] > 0.50:
            print(unique_values[i], " number of overall reservation: ", sumuniqe[i], "cancellation probabilty : ",
                  final_vals_for_unique_values[i])

    print('payment_currency  under 20 %')

    for i in range(len(final_vals_for_unique_values)):
        if final_vals_for_unique_values[i] < 0.2:
            print(unique_values[i], " number of overall reservation: ", sumuniqe[i], "cancellation probabilty : ",
                  final_vals_for_unique_values[i])
def plots_of_features():
    # make  piercing correlation graph for each feature with y (cancellation)
    X, y = preprocess()
    # for each feature make a plot of the feature with y
    for feature in L_features:

        # for each feature find unique values
        unique_values = X[feature].unique()
        # for each unique find the number of time it appears
        X_feature_list = list(X[feature])
        sumuniqe = [X_feature_list.count(uni) for uni in unique_values]
        # for each unique value find the number of cancellation
        unique_values_hash = {unique_values[i]: 0 for i in range(len(unique_values))}
        for row in range(len(y)):
            unique_values_hash[X[feature][row]] += y[row]

        # list of the unique_values_hash.values()
        L_unique_values_hash_values = list(unique_values_hash.values())
        final_vals_for_unique_values = [L_unique_values_hash_values[i] / sumuniqe[i] for i in
                                        range(len(L_unique_values_hash_values))]
        if feature == 'guest_nationality_country_name':
            cheack_guest_nationality_country_name(final_vals_for_unique_values, unique_values, sumuniqe)
        if feature== 'original_payment_currency':
            #get max value and the contry of the max value
            check_original_payment_currency(final_vals_for_unique_values, unique_values, sumuniqe)
        # make a plot of the feature with y
        # plt.scatter(unique_values, L_unique_values_hash_values)
        plt.bar(unique_values, final_vals_for_unique_values, color='maroon',
                width=0.4)
        plt.title(feature)
        plt.xlabel(feature)
        plt.ylabel('cancellation')
        plt.show()


def main():
    important_data()
    plots_of_features()


if __name__ == '__main__':
    main()
