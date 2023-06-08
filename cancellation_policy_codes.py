import math


def cancellation_cost(cancellation_policy_codes, nights_count, total_prices, days_before_checkin):
    night_cost = total_prices / nights_count
    parsed_policy_codes = parse_cancellation_policy_codes(cancellation_policy_codes)
    min_index = argmin_days(days_before_checkin, parsed_policy_codes)
    if min_index == math.inf:
        return 0

    days_before_checkin, percentage, nights = parsed_policy_codes[min_index]
    if nights == 0:
        user_cost = total_prices * (percentage / 100)

    else:
        user_cost = night_cost * nights

    return total_prices - user_cost


def argmin_days(days_before_checkin, parsed_policy_codes):
    min_index = 0
    min_valid_days = math.inf
    for i, policy_code in enumerate(parsed_policy_codes):
        days_before_charge, percentage, nights = policy_code
        if days_before_checkin <= days_before_charge < min_valid_days:
            min_valid_days = days_before_charge
            min_index = i
    return min_index


def parse_cancellation_policy_codes(cancellation_policy_codes):
    policy_codes = cancellation_policy_codes.split("_")
    parsed_policy_codes = [parse_single_policy_code(policy_code) for policy_code in policy_codes]
    return parsed_policy_codes


def parse_single_policy_code(policy_code):
    if policy_code == "UNKNOWN":
        return 0, 0, 0
    days_charge = policy_code.split("D")
    percentage = 0
    nights = 0
    if len(days_charge) == 1:
        charge = days_charge[0]
        days_before_charge = 0
    else:
        charge = days_charge[1]
        days_before_charge = int(days_charge[0])

    if "N" in charge:
        nights = int(charge.replace("N", ""))
        percentage = 0

    elif "P" in charge:
        percentage = int(charge.replace("P", ""))
        nights = 0

    return days_before_charge, percentage, nights
