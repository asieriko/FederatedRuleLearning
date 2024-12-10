def average_dict(results):
    sum_dict = {}
    for d in results:
        for key, value in d.items():
            if key in sum_dict:
                sum_dict[key] += value
            else:
                sum_dict[key] = value

    num_dicts = len(results)
    average_result = {key: value / num_dicts for key, value in sum_dict.items()}
    return average_result