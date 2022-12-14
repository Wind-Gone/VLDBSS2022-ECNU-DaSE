import joblib
import numpy as np


def estimate_plan(operator, factors, weights):
    cost = 0.0
    for child in operator.children:
        cost += estimate_plan(child, factors, weights)

    if operator.is_hash_agg():
        # YOUR CODE HERE: hash_agg_cost = input_row_cnt * cpu_fac
        input_row_cnt = operator.est_row_counts()
        cost += input_row_cnt * factors['cpu']
        weights['cpu'] += input_row_cnt

    elif operator.is_hash_join():
        # YOUR CODE HERE:
        # hash_join_cost = (build_hashmap_cost + probe_and_pair_cost)
        #   = (build_row_cnt * cpu_fac) + (output_row_cnt * cpu_fac)
        output_row_cnt = operator.est_row_counts()
        build_side = int(operator.children[1].is_build_side())
        build_row_cnt = operator.children[build_side].est_row_counts()

        cost += (build_row_cnt + output_row_cnt) * factors['cpu']
        weights['cpu'] += (build_row_cnt + output_row_cnt)

    elif operator.is_sort():
        # YOUR CODE HERE: sort_cost = input_row_cnt * log(input_row_cnt) * cpu_fac
        input_row_cnt = operator.est_row_counts()
        cost += input_row_cnt * np.log(input_row_cnt + 1e-10) * factors['cpu']
        weights['cpu'] += input_row_cnt * np.log(input_row_cnt + 1e-10)

    elif operator.is_selection():
        # YOUR CODE HERE: selection_cost = input_row_cnt * cpu_fac
        input_row_cnt = operator.est_row_counts()
        cost += input_row_cnt * factors['cpu']
        weights['cpu'] += input_row_cnt

    elif operator.is_projection():
        # YOUR CODE HERE: projection_cost = input_row_cnt * cpu_fac
        input_row_cnt = operator.est_row_counts()
        cost += input_row_cnt * factors['cpu']
        weights['cpu'] += input_row_cnt

    elif operator.is_table_reader():
        # YOUR CODE HERE: table_reader_cost = input_row_cnt * input_row_size * net_fac
        input_row_cnt = operator.est_row_counts()
        input_row_size = operator.row_size()
        cost += input_row_cnt * input_row_size * factors['net']
        weights['net'] += input_row_cnt * input_row_size

    elif operator.is_table_scan():
        # YOUR CODE HERE: table_scan_cost = row_cnt * row_size * scan_fac
        row_cnt = operator.est_row_counts()
        row_size = operator.row_size()
        cost += row_cnt * row_size * factors['scan']
        weights['scan'] += row_cnt * row_size

    elif operator.is_index_reader():
        # YOUR CODE HERE: index_reader_cost = input_row_cnt * input_row_size * net_fac
        input_row_cnt = operator.est_row_counts()
        input_row_size = operator.row_size()
        cost += input_row_cnt * input_row_size * factors['net']
        weights['net'] += input_row_cnt * input_row_size

    elif operator.is_index_scan():
        # YOUR CODE HERE: index_scan_cost = row_cnt * row_size * scan_fac
        row_cnt = operator.est_row_counts()
        row_size = operator.row_size()
        cost += row_cnt * row_size * factors['scan']
        weights['scan'] += row_cnt * row_size

    elif operator.is_index_lookup():
        # YOUR CODE HERE:
        # index_lookup_cost = net_cost + seek_cost
        #   = (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size) * net_fac +
        #     (build_row_cnt / batch_size) * seek_fac
        build_side = int(operator.children[1].is_build_side())
        build_row_cnt = operator.children[build_side].est_row_counts()
        build_row_size = operator.children[build_side].row_size()
        probe_row_cnt = operator.children[1 - build_side].est_row_counts()
        probe_row_size = operator.children[1 - build_side].row_size()
        batch_size = operator.batch_size()

        cost += (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size) * factors['net']
        weights['net'] += (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size)

        cost += (build_row_cnt / batch_size) * factors['seek']
        weights['seek'] += (build_row_cnt / batch_size)

    else:
        print(operator.id)
        assert (1 == 2)  # unknown operator
    return cost


def load_cost_model():
    cost_model = joblib.load('./data/lab2-linreg.pkl')
    return cost_model


def predict(model, test_plan):
    # evaluation
    est_costs = []
    new_factors = {}
    new_factors['cpu'] = model.coef_.data[0]
    new_factors['scan'] = model.coef_.data[1]
    new_factors['net'] = model.coef_.data[2]
    new_factors['seek'] = model.coef_.data[3]
    w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
    cost = estimate_plan(test_plan.root, new_factors, w)
    est_costs.append(cost)
    return est_costs
