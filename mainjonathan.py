import csv

import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("data/agoda_cancellation_train.csv")
    # for c in data.columns:
    #     print(c)
    #     data.loc[:, c][np.isnan(data.loc[:, c])] = 0
        # print(c + ": " + str(numpy.unique(data.loc[:, c]).size))
    # accommadation_type_name original_payment_type cancellation_policy_code hotel_area_code customer_nationality
    print(np.unique(data.loc[:, "h_customer_id"]).size)
    # print(np.unique(data.loc[:, "language"]))
    # hotel_country_code accommadation_type_name charge_option
    # customer_nationality guest_nationality_country_name origin_country_code
    # language original_payment_method original_payment_type cancellation_policy_code
    #