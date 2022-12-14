import json

import numpy
import torch
import xgboost as xgb
from torch import nn

import evaluation_utils as eval_utils
import range_query as rq
import statistics as stats


def min_max_normalize(v, min_v, max_v):
    # The function may be useful when dealing with lower/upper bounds of columns.
    assert max_v > min_v
    return (v - min_v) / (max_v - min_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->
    feature = []
    # YOUR CODE HERE: extract features from query
    parsed_query = range_query.parse_range_query(range_query.query)
    for column in considered_cols:
        if (
                column in parsed_query.column_names() and column in parsed_query.col_left and column in parsed_query.col_right):
            l_bound = parsed_query.col_left[column]
            r_bound = parsed_query.col_right[column]
            feature.append(l_bound)
            feature.append(r_bound)
        else:
            feature.append(table_stats.columns[column].min_val())
            feature.append(table_stats.columns[column].max_val())
    feature.append(stats.AVIEstimator.estimate(parsed_query, table_stats))  # avi_sel
    feature.append(stats.ExpBackoffEstimator.estimate(parsed_query, table_stats))  # ebo_sel
    feature.append(stats.MinSelEstimator.estimate(parsed_query, table_stats))  # min_sel
    return feature


def preprocess_queries(queris, table_stats, columns):
    """
    preprocess_queries turn queries into features and labels, which are used for regression model.
    """
    features, labels = [], []
    for item in queris:
        query, act_rows = item['query'], item['act_rows']
        feature, label = None, None
        # YOUR CODE HERE: transform (query, act_rows) to (feature, label)
        # Some functions like rq.ParsedRangeQuery.parse_range_query and extract_features_from_query may be helpful.
        parsed_range_query = rq.ParsedRangeQuery(query, "tidb", "imdb", [], [])
        feature = extract_features_from_query(parsed_range_query, table_stats, columns)
        label = act_rows
        features.append(feature)
        labels.append(label)
    features = numpy.array(features).astype(numpy.float32)
    labels = numpy.array(labels).astype(numpy.float32)
    features = (features - numpy.min(features)) / (numpy.max(features) - numpy.min(features))
    return torch.from_numpy(features), torch.from_numpy(labels)


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        self.query_data = list(zip(*preprocess_queries(queries, table_stats, columns)))

    def __getitem__(self, index):
        return self.query_data[index]

    def __len__(self):
        return len(self.query_data)


# 搭建MLP回归模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_features=15, out_features=128, bias=True)
        self.hidden2 = nn.Linear(128, 128)
        self.predict = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        output = self.predict(x)
        return output[:, 0]


# 自定义损失函数
class SelfMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow(((x - y) / (x + y)), 2))


def est_mlp(train_data, test_data, table_stats, columns):
    """
    est_mlp uses MLP to produce estimated rows for train_data and test_data
    """
    train_dataset = QueryDataset(train_data, table_stats, columns)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=1)
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure
    mlpreg = MLP()
    optimizer = torch.optim.Adam(mlpreg.parameters(), lr=3e-2)
    loss_func = SelfMSELoss()
    train_loss_all = []
    epochs = 25
    for epoch in range(epochs):
        # Print epoch
        print(f'Starting epoch {epoch + 1}')
        train_loss = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            estimate = mlpreg(b_x)
            loss = loss_func(estimate, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)

    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_est_rows, test_act_rows = [], []
    # YOUR CODE HERE: test procedure
    for i, data in enumerate(test_loader):
        inputs, targets = data
        pre_y = mlpreg(inputs)
        pre_y = pre_y.data.numpy()
        for j in range(len(pre_y)):
            test_est_rows.append(pre_y[j])
    return train_est_rows, train_act_rows, numpy.array(test_est_rows).tolist(), test_act_rows


def est_xgb(train_data, test_data, table_stats, columns):
    """
    est_xgb uses xgboost to produce estimated rows for train_data and test_data
    """
    print("estimate row counts by xgboost")
    train_x, train_y = preprocess_queries(train_data, table_stats, columns)
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure
    params = {
        'booster': 'gbtree',
        'eta': 0.3,
        'max_depth': 10,
        'subsample': 1.0,
        'min_child_weight': 5,
        'colsample_bytree': 0.2,
        'scale_pos_weight': 0.1,
        'eval_metric': 'auc',
        'gamma': 0.4,
        'objective': 'reg:squarederror',
        'lambda': 300
    }
    xg_train = xgb.DMatrix(train_x, label=train_y)
    model = xgb.train(params, xg_train, num_boost_round=2000)
    test_x, test_y = preprocess_queries(test_data, table_stats, columns)
    test_est_rows, test_act_rows = [], []
    # YOUR CODE HERE: test procedure
    test_est_rows = model.predict(xgb.DMatrix(test_x))
    test_act_rows = test_y
    train_act_rows = train_x
    return train_est_rows, train_act_rows, test_est_rows.tolist(), test_act_rows


def cost_error(y_pred, y_true):
    grad = y_pred - y_true.get_label()
    hess = numpy.log(y_pred) - numpy.log(y_true.get_label())
    return grad, hess


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'mlp':
        est_fn = est_mlp
    else:
        est_fn = est_xgb

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

    name = f'{model}_train_{len(train_data)}'
    eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


if __name__ == '__main__':
    stats_json_file = './data/title_stats.json'
    train_json_file = './data/query_train_2000.json'
    test_json_file = './data/query_test_500.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    # eval_model('mlp', train_data, test_data, table_stats, columns)
    eval_model('xgb', train_data, test_data, table_stats, columns)
