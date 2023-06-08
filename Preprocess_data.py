import pandas as pd

import Currency_convert

CATEGORICAL_FEATURES = ["charge_option", "customer_nationality",
                        "guest_nationality_country_name", "origin_country_code"
                                                          "language", "original_payment_method",
                        "original_payment_type"]


# h_booking_id - delete from train, save from output
# booking_datetime - parse_dates=["booking_datetime"] in read_csv
# checkin_date - parse_dates=["checkin_date"] in read_csv
# checkout_date - delete from train, will be used to calculate stay_duration
# hotel_id - delete for now TODO
# hotel_country_code - categorical
# hotel_live_date - delete for now TODO
# hotel_star_rating - no change, in range [1,5]
# accommadation_type_name - categorical
# charge_option - categorical
# h_customer_id - delete for now TODO
# customer_nationality - categorical, remove "of America" from prefix "United States of America"
# guest_is_not_the_customer - no change, already categorical, in {0,1}
# guest_nationality_country_name - categorical
# no_of_adults - numeric, min: 1, max: TODO
# no_of_children - numeric, min: 0, max: TODO
# no_of_extra_bed - numeric, min: 0, max: TODO
# no_of_room - numeric, min: 1, max: TODO
# origin_country_code - categorical, remove null and what is A1?
# language - categorical
# original_selling_amount - numeric, apply currency_convert, min: TODO, max: TODO
# original_payment_method - categorical
# original_payment_type - categorical
# original_payment_currency - categorical
# is_user_logged_in - no change, already categorical, in {0,1}
# cancellation_policy_code - categorical TODO
# is_first_booking - no change, already categorical, in {0,1}
# request_nonesmoke - null to 0, will be categorical, in {0,1}
# request_latecheckin - null to 0, will be categorical, in {0,1}
# request_highfloor - null to 0, will be categorical, in {0,1}
# request_largebed - null to 0, will be categorical, in {0,1}
# request_twinbeds - null to 0, will be categorical, in {0,1}
# request_airport - null to 0, will be categorical, in {0,1}
# request_earlycheckin - null to 0, will be categorical, in {0,1}
# cancellation_datetime - delete from train, will be y_train, null to 0 the rest will be 1
# hotel_area_code - use hotel_area_code_by_country with hash
# hotel_brand_code - delete for now TODO
# hotel_chain_code - categorical, null to 0 if 0 is not in the list of codes
# hotel_city_code - categorical
#
# ##### new features #####
# stay_duration - (checkout_date - checkin_date).days
# booking_datetime_DayOfYear - df["booking_datetime"].dt.day_of_year
# booking_datetime_year - df["booking_datetime"].dt.year
# checkin_date_DayOfYear - df["checkin_date"].dt.day_of_year
# checkin_date_year - df["checkin_date"].dt.year
# hotel_area_code_by_country - categorical


def preprocess():
    data = pd.read_csv('data\\agoda_cancellation_train.csv')
    X_train = data.drop('cancellation_datetime', axis=1)
    y_train = data['cancellation_datetime']
    y_train = y_train.apply(lambda x: 0 if pd.isnull(x) else 1)

    # booking_datetime - delete, parse_dates=["booking_datetime"] in read_csv, use booking_datetime_DayOfYear
    #                                                                             and booking_datetime_year
    X_train['booking_datetime'] = pd.to_datetime(X_train['checkout_date'])
    X_train["booking_datetime"] = X_train["booking_datetime"].dt.day_of_year
    X_train["booking_datetime"] = X_train["booking_datetime"].dt.year
    X_train = X_train.drop('booking_datetime', axis=1)


    # checkin_date - delete, parse_dates=["checkin_date"] in read_csv, use booking_datetime_DayOfYear
    #                                                                             and booking_datetime_year
    X_train['checkin_date'] = pd.to_datetime(X_train['checkin_date'])
    X_train["checkin_date"] = X_train["checkin_date"].dt.day_of_year
    X_train["checkin_date"] = X_train["checkin_date"].dt.year
    X_train = X_train.drop('checkin_date', axis=1)

    # checkout_date - delete from train, will be used to calculate stay_duration
    # (checkout_date - checkin_date).days
    X_train['checkout_date'] = pd.to_datetime(X_train['checkout_date'])
    X_train["stay_duration"] = (X_train['checkout_date'] - X_train['checkin_date']).dt.days
    X_train = X_train.drop('checkout_date', axis=1)

    # hotel_id - delete for now TODO
    X_train = X_train.drop('hotel_id', axis=1)

    # hotel_country_code - categorical
    X_train['hotel_country_code'] = X_train['hotel_country_code'].dropna()
    X_train = pd.get_dummies(X_train, prefix="hotel_country_code_", columns=['hotel_country_code'])

    # hotel_live_date - delete for now TODO
    X_train = X_train.drop('hotel_live_date', axis=1)

    # hotel_star_rating - no change, in range [1,5]
    mask = X_train['hotel_star_rating'] >= 1
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = X_train['hotel_star_rating'] <= 5
    X_train = X_train[mask]
    y_train = y_train[mask]

    # accommadation_type_name - categorical
    X_train = pd.get_dummies(X_train, prefix="accommadation_type_name_", columns=['accommadation_type_name'])

    # charge_option - categorical
    X_train = pd.get_dummies(X_train, prefix="charge_option_", columns=['charge_option'])

    # h_customer_id - delete for now TODO
    X_train = X_train.drop('h_customer_id', axis=1)

    # customer_nationality - categorical, remove "of America" from prefix "United States of America"
    X_train["customer_nationality"] = X_train["customer_nationality"].apply(lambda country:
                                                                            "United States" if country ==
                                                                            "United States of America" else country)

    X_train = pd.get_dummies(X_train, prefix="customer_nationality_", columns=['customer_nationality'])

    # guest_is_not_the_customer - no change, already categorical, in {0,1}
    mask = X_train['guest_is_not_the_customer'].isin({0, 1})
    X_train = X_train[mask]
    y_train = y_train[mask]

    # guest_nationality_country_name - categorical
    X_train = pd.get_dummies(X_train, prefix="guest_nationality_country_name_",
                             columns=['guest_nationality_country_name'])

    # no_of_adults - numeric int, min: 1, max: TODO
    X_train['no_of_adults'] = X_train['no_of_adults'].astype(int)
    mask = X_train['no_of_adults'] >= 1
    X_train = X_train[mask]
    y_train = y_train[mask]

    # no_of_children - numeric int, min: 0, max: TODO
    X_train['no_of_children'] = X_train['no_of_children'].astype(int)
    mask = X_train['no_of_children'] >= 0
    X_train = X_train[mask]
    y_train = y_train[mask]

    # no_of_rooms - numeric int, min: 1, max: TODO
    X_train['no_of_rooms'] = X_train['no_of_rooms'].astype(int)
    mask = X_train['no_of_rooms'] >= 1
    X_train = X_train[mask]
    y_train = y_train[mask]

    # origin_country_code - categorical, remove null and TODO what is A1?
    X_train['origin_country_code'] = X_train['origin_country_code'].dropna()
    X_train = pd.get_dummies(X_train, prefix="origin_country_code_", columns=['origin_country_code'])

    # language - categorical
    X_train = pd.get_dummies(X_train, prefix="language_", columns=['language'])

    # original_selling_amount - numeric, apply currency_convert, min: TODO, max: TODO
    X_train['original_selling_amount'] = X_train['original_selling_amount'].astype(float)
    X_train['original_selling_amount_in_dollar'] = list(zip(X_train['original_selling_amount'],
                                                            X_train['original_currency_code']))
    X_train['original_selling_amount_in_dollar'] = X_train['original_selling_amount'].apply(lambda amount, currency:
                                                                                  Currency_convert.to_dollar(amount,
                                                                                                             currency))
    # original_payment_method - categorical
    X_train = pd.get_dummies(X_train, prefix="original_payment_method_", columns=['original_payment_method'])

    # original_payment_type - categorical
    X_train = pd.get_dummies(X_train, prefix="original_payment_type_", columns=['original_payment_type'])

    # original_payment_currency - categorical
    X_train = pd.get_dummies(X_train, prefix="original_payment_currency_", columns=['original_payment_currency'])

    # is_user_logged_in - no change, already categorical, in {0,1}
    mask = X_train['is_user_logged_in'].isin({0, 1})
    X_train = X_train[mask]
    y_train = y_train[mask]

    # cancellation_policy_code - categorical TODO
    X_train = pd.get_dummies(X_train, prefix="cancellation_policy_code_", columns=['cancellation_policy_code'])

    # is_first_booking - no change, already categorical, in {0,1}
    mask = X_train['is_first_booking'].isin({0, 1})
    X_train = X_train[mask]
    y_train = y_train[mask]

    # request_* - null to 0, will be categorical, in {0,1}
    for feature in ['request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed',
                    "request_twinbeds", "request_airport", "request_earlycheckin"]:
        X_train[feature] = X_train[feature].fillna(0)
        mask = X_train[feature].isin({0, 1})
        X_train = X_train[mask]
        y_train = y_train[mask]

    # hotel_area_code - use hotel_area_code_by_country with hash
    # hotel_area_code_by_country - categorical
    X_train['original_selling_amount_in_dollar'] = list(zip(X_train['hotel_area_code'],
                                                            X_train['hotel_country_code']))

    X_train['hotel_area_code_by_country'] = X_train['original_selling_amount_in_dollar'].apply(lambda x: hash(x))
    X_train = pd.get_dummies(X_train, prefix="hotel_area_code_by_country_", columns=['hotel_area_code_by_country'])

    # hotel_brand_code - delete for now TODO
    X_train = X_train.drop('hotel_brand_code', axis=1)

    # hotel_chain_code - categorical, null to "No-Chain"
    X_train['hotel_chain_code'] = X_train['hotel_chain_code'].fillna("No-Chain")
    X_train = pd.get_dummies(X_train, prefix="hotel_chain_code_", columns=['hotel_chain_code'])

    # hotel_city_code - categorical
    X_train = pd.get_dummies(X_train, prefix="hotel_city_code_", columns=['hotel_city_code'])

    # h_booking_id - delete from train, save from output
    h_booking_id_save = X_train['h_booking_id']
    X_train = X_train.drop('h_booking_id', axis=1)

    return X_train, y_train, h_booking_id_save


def preprocess2():
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
    X, y, y_index = preprocess()


if __name__ == '__main__':
    main()
