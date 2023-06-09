import numpy as np
import pandas as pd

import Currency_convert
from cancellation_policy_codes import cancellation_cost

CATEGORICAL_FEATURES = ["accommadation_type_name",
                       "charge_option",
                       "customer_nationality",
                       "guest_nationality_country_name",
                       "origin_country_code",
                       "language",
                       "original_payment_method",
                       "original_payment_type",
                       "original_payment_currency",
                       "cancellation_policy_code",
                       "hotel_area_code_by_country",
                       "hotel_chain_code",
                       "hotel_city_code",
                       "hotel_country_code"]

X_train = None

def preprocess_features(X, y_train=None):
    # booking_datetime - delete, parse_dates=["booking_datetime"] in read_csv, use booking_datetime_DayOfYear
    #                                                                             and booking_datetime_year
    X['booking_datetime'] = pd.to_datetime(X['booking_datetime'])
    X["booking_datetime_DayOfYear"] = X["booking_datetime"].dt.dayofyear
    X["booking_datetime_year"] = X["booking_datetime"].dt.year
    X = X.drop('booking_datetime', axis=1)

    # checkin_date - delete, parse_dates=["checkin_date"] in read_csv, use booking_datetime_DayOfYear
    #                                                                             and booking_datetime_year
    X['checkin_date'] = pd.to_datetime(X['checkin_date'])
    X["checkin_date_DayOfYear"] = X["checkin_date"].dt.dayofyear
    X["checkin_date_year"] = X["checkin_date"].dt.year
    X = X.drop('checkin_date', axis=1)

    # checkout_date - delete from train, will be used to calculate stay_duration
    # (checkout_date - checkin_date).days
    X['checkout_date'] = pd.to_datetime(X['checkout_date'])
    days = (X["checkin_date_DayOfYear"] - X["checkout_date"].dt.dayofyear)
    years = X["checkout_date"].dt.year - X['checkin_date_year']
    X["stay_duration"] = (((days % 365) + 365) % 365) + (years * 365) # mod 365

    X = X.drop('checkout_date', axis=1)

    # hotel_id - delete for now TODO
    X = X.drop('hotel_id', axis=1)

    # hotel_live_date - delete for now TODO
    X = X.drop('hotel_live_date', axis=1)

    # hotel_star_rating - no change, in range [1,5]

    if y_train is not None:
        mask = X['hotel_star_rating'] >= 1
        X = X[mask]
        y_train = y_train[mask]
        mask = X['hotel_star_rating'] <= 5
        X = X[mask]
        y_train = y_train[mask]

    # accommadation_type_name - categorical
    X = pd.get_dummies(X, prefix="accommadation_type_name_", columns=['accommadation_type_name'])

    # charge_option - categorical
    X = pd.get_dummies(X, prefix="charge_option_", columns=['charge_option'])

    # h_customer_id - delete for now TODO
    X = X.drop('h_customer_id', axis=1)

    # customer_nationality - categorical, remove "of America" from prefix "United States of America"
    X["customer_nationality"] = X["customer_nationality"].apply(lambda country:
                                                                            "United States" if country ==
                                                                                               "United States of America" else country)

    X = pd.get_dummies(X, prefix="customer_nationality_", columns=['customer_nationality'])

    # guest_is_not_the_customer - no change, already categorical, in {0,1}
    if y_train is not None:
        mask = X['guest_is_not_the_customer'].isin({0, 1})
        X = X[mask]
        y_train = y_train[mask]
    else:
        X['guest_is_not_the_customer'] = \
            X['guest_is_not_the_customer'].apply(lambda x: 0 if x not in {0, 1} else x)

    # guest_nationality_country_name - categorical
    X = pd.get_dummies(X, prefix="guest_nationality_country_name_",
                       columns=['guest_nationality_country_name'])

    # no_of_adults - numeric int, min: 1, max: TODO
    X['no_of_adults'] = X['no_of_adults'].astype(int)
    if y_train is not None:
        mask = X['no_of_adults'] >= 1
        X = X[mask]
        y_train = y_train[mask]
    else:
        X['no_of_adults'] = X['no_of_adults'].apply(lambda x: 1 if x < 1 else x)

    # no_of_children - numeric int, min: 0, max: TODO
    X['no_of_children'] = X['no_of_children'].astype(int)
    if y_train is not None:
        mask = X['no_of_children'] >= 0
        X = X[mask]
        y_train = y_train[mask]
    else:
        X['no_of_children'] = X['no_of_children'].apply(lambda x: 0 if x < 0 else x)

    # no_of_room - numeric int, min: 1, max: TODO
    X['no_of_room'] = X['no_of_room'].astype(int)
    if y_train is not None:
        mask = X['no_of_room'] >= 1
        X = X[mask]
        y_train = y_train[mask]
    else:
        X['no_of_room'] = X['no_of_room'].apply(lambda x: 1 if x < 1 else x)

    # origin_country_code - categorical,remove null and TODO what is A1?
    if y_train is not None:
        mask = X['origin_country_code'].isna()
        X = X[~mask]
        y_train = y_train[~mask]
    else:
        X['origin_country_code'] = X['origin_country_code'].fillna("USA")

    X = pd.get_dummies(X, prefix="origin_country_code_", columns=['origin_country_code'])

    # language - categorical
    X = pd.get_dummies(X, prefix="language_", columns=['language'])

    # original_payment_method - categorical
    X = pd.get_dummies(X, prefix="original_payment_method_", columns=['original_payment_method'])

    # original_payment_type - categorical
    X = pd.get_dummies(X, prefix="original_payment_type_", columns=['original_payment_type'])

    # original_payment_currency - categorical
    X = pd.get_dummies(X, prefix="original_payment_currency_", columns=['original_payment_currency'])

    # is_user_logged_in - no change, already categorical, in {0,1}
    if y_train is not None:
        mask = X['is_user_logged_in'].isin({0, 1})
        X = X[mask]
        y_train = y_train[mask]
    else:
        X['is_user_logged_in'] = X['is_user_logged_in'].apply(lambda x: 0 if x not in {0, 1} else x)

    # is_first_booking - no change, already categorical, in {0,1}
    if y_train is not None:
        mask = X['is_first_booking'].isin({0, 1})
        X = X[mask]
        y_train = y_train[mask]
    else:
        X['is_first_booking'] = X['is_first_booking'].apply(lambda x: 0 if x not in {0, 1} else x)

    # request_* - null to 0, will be categorical, in {0,1}
    for feature in ['request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed',
                    "request_twinbeds", "request_airport", "request_earlycheckin"]:
        X[feature] = X[feature].fillna(0)
        if y_train is not None:
            mask = X[feature].isin({0, 1})
            X = X[mask]
            y_train = y_train[mask]
        else:
            X[feature] = X[feature].apply(lambda x: 0 if x not in {0, 1} else x)

    # hotel_area_code - use hotel_area_code_by_country with hash
    # hotel_area_code_by_country - categorical
    X['hotel_area_code_by_country'] = list(zip(X['hotel_area_code'],
                                               X['hotel_country_code']))

    X['hotel_area_code_by_country'] = X['hotel_area_code_by_country'].apply(lambda x: hash(x))
    X = pd.get_dummies(X, prefix="hotel_area_code_by_country_", columns=['hotel_area_code_by_country'])

    # hotel_brand_code - delete for now TODO
    X = X.drop('hotel_brand_code', axis=1)

    # hotel_chain_code - categorical,null to "No-Chain"
    X['hotel_chain_code'] = X['hotel_chain_code'].fillna("No-Chain")
    X = pd.get_dummies(X, prefix="hotel_chain_code_", columns=['hotel_chain_code'])

    # hotel_city_code - categorical
    X = pd.get_dummies(X, prefix="hotel_city_code_", columns=['hotel_city_code'])

    # hotel_country_code - categorical
    if y_train is not None:
        mask = X["hotel_country_code"].isna()
        X = X[~mask]
        y_train = y_train[~mask]
    else:
        X["hotel_country_code"] = X["hotel_country_code"].fillna('USA')

    X = pd.get_dummies(X, prefix="hotel_country_code_", columns=['hotel_country_code'])

    # h_booking_id -delete from train, save from output
    h_booking_id_save = X['h_booking_id']
    X = X.drop('h_booking_id', axis=1)

    if y_train is not None:
        X = X.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        h_booking_id_save = h_booking_id_save.reset_index(drop=True)
    return X, y_train, h_booking_id_save


def preprocess_for_cost(X, y_train=None):
    X, y_train, h_booking_id_save = preprocess_features(X, y_train)
    if y_train is not None:
        cancellation_days = X['cancellation_datetime'].apply(lambda x: 0 if pd.isnull(x)
                                                                                    else pd.to_datetime(x).dayofyear +
                                                                                         365*pd.to_datetime(x).year)
        days_before_checkin = (X['checkin_date_DayOfYear'] + X['checkin_date_year'] * 365) - \
                              cancellation_days

        func = np.vectorize(cancellation_cost, otypes=[float])
        y_train =  func(X["cancellation_policy_code"], y_train,
                                    days_before_checkin, X['stay_duration'])

        y_train[X['cancellation_datetime'].isna()] = -1

        X['cancellation_datetime'] = X['cancellation_datetime'].apply(lambda x: 0 if pd.isnull(x) else 1)
        X = X.drop('cancellation_policy_code', axis=1)
    return X, y_train, h_booking_id_save

def preprocess_for_cancellation(X, y_train=None):
    if y_train is not None:
        y_train = y_train.apply(lambda x: 0 if pd.isnull(x) else 1)

    # original_selling_amount - numeric, apply currency_convert, min: TODO, max: TODO
    X['original_selling_amount'] = X['original_selling_amount'].astype(float)
    X['original_selling_amount_in_dollar'] = list(zip(X['original_selling_amount'],
                                                            X['original_payment_currency']))
    X['original_selling_amount_in_dollar'] = X['original_selling_amount_in_dollar'].apply(
        lambda amount_currency:
        Currency_convert.to_dollar(amount_currency[0],
                                   amount_currency[1]))

    # cancellation_policy_code - categorical TODO
    # X_train = pd.get_dummies(X_train, prefix="cancellation_policy_code_", columns=['cancellation_policy_code'])
    X = X.drop('cancellation_policy_code', axis=1)
    X, y_train, h_booking_id_save = preprocess_features(X, y_train)
    return X, y_train, h_booking_id_save


def preprocess_test_reindex(X_train, X_test):
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_test


