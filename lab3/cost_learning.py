import torch
import torch.nn as nn

from plan import Operator

operators = ["Projection", "Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableReader",
             "IndexReader", "IndexLookUp"]


# There are many ways to extract features from plan:
# 1. The simplest way is to extract features from each node and sum them up. For example, we can get
#      a  the number of nodes;
#      a. the number of occurrences of each operator;
#      b. the sum of estRows for each operator.
#    However we lose the tree structure after extracting features.
# 2. The second way is to extract features from each node and concatenate them in the DFS traversal order.
#                  HashJoin_1
#                  /          \
#              IndexJoin_2   TableScan_6
#              /         \
#          IndexScan_3   IndexScan_4
#    For example, we can concatenate the node features of the above plan as follows:
#    [Feat(HashJoin_1)], [Feat(IndexJoin_2)], [Feat(IndexScan_3)], [Feat(IndexScan_4)], [Padding], [Feat(TableScan_6)], [Padding]
#    Notice1: When we traverse all the children in DFS, we insert [Padding] as the end of the children. In this way, we
#    have an one-on-one mapping between the plan tree and the DFS order sequence.
#    Notice2: Since the different plans have the different number of nodes, we need padding to make the lengths of the
#    features of different plans equal.
class PlanFeatureCollector:
    def __init__(self):
        # YOUR CODE HERE: define variables to collect features from plans
        self.feature = []
        # self.est_rows = est_rows
        # self.est_cost = est_cost
        # self.task = task
        # self.acc_obj = acc_obj
        # self.op_info = op_info

    def add_operator(self, op: Operator):
        # YOUR CODE HERE: extract features from op
        op_type = []
        print(op)
        for i, v in enumerate(operators):
            if v in op['id']:
                op_type.append(1)
            else:
                op_type.append(0)
        # loss some info in op id 
        op_info = op['op_info']
        row_size = float(op_info.split(':')[-1])
        self.feature += op_type + [float(op['est_rows']), row_size]

    def walk_operator_tree(self, op: Operator):
        self.add_operator(op)
        if op['children'] == None:
            return self.feature

        for child in op['children']:
            self.walk_operator_tree(child)
        # YOUR CODE HERE: process and return the features
        return self.feature


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, plans, max_operator_num):
        super().__init__()
        self.data = []
        for plan in plans:
            collector = PlanFeatureCollector()
            vec = collector.walk_operator_tree(plan.root)
            # YOUR CODE HERE: maybe you need padding the features if you choose the second way to extract the features.
            padding_size = 80

            if len(vec) >= 80:
                vec = vec[:padding_size]
            while len(vec) < 80:
                vec.append(0)

            features = torch.Tensor(vec)
            exec_time = torch.Tensor([plan.exec_time_in_ms()])
            self.data.append((features, exec_time))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Define your model for cost estimation
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # YOUR CODE HERE
        self.input = nn.Linear(80, 128)
        self.relu1 = nn.ReLU()
        self.hidden1 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.hidden2 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        # YOUR CODE HERE
        x = self.relu1(self.input(x))
        x = self.relu2(self.hidden1(x))
        x = self.relu3(self.hidden2(x))
        return self.out(x)
        pass

    def init_weights(self):
        # YOUR CODE HERE
        pass


class SelfMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return -torch.mean(torch.pow(((x - y) / (x + y)), 3))


def count_operator_num(op: Operator):
    num = 2  # one for the node and another for the end of children
    for child in op.children:
        num += count_operator_num(child)
    return num


def estimate_learning(train_plans, test_plans):
    max_operator_num = 0
    for plan in train_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    for plan in test_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    print(f"max_operator_num:{max_operator_num}")

    train_dataset = PlanDataset(train_plans, max_operator_num)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)

    model = YourModel()
    model.init_weights()

    loss_func = SelfMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.5)

    def loss_fn(est_time, act_time):
        # YOUR CODE HERE: define loss function
        return loss_func(est_time, act_time)
        pass

    # YOUR CODE HERE: complete training loop
    num_epoch = 25
    total_loss = 0
    for epoch in range(num_epoch):
        print(f"epoch {epoch} start")
        for i, data in enumerate(train_loader):
            x, y = data
            output = model(x)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        print(total_loss)
    train_est_times, train_act_times = [], []
    for i, data in enumerate(train_loader):
        # YOUR CODE HERE: evaluate on train data
        # x,y = data
        pass
    torch.save(model.state_dict(), 'lab2-learning.pt')
    test_dataset = PlanDataset(test_plans, max_operator_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    test_est_times, test_act_times = [], []
    for i, data in enumerate(test_loader):
        # YOUR CODE HERE: evaluate on test data
        x, y = data
        # print(x,y)
        pre_y = model(x).tolist()
        for j in pre_y:
            test_est_times += j
        for j in y.tolist():
            test_act_times += j
        pass

    return train_est_times, train_act_times, test_est_times, test_act_times


def load_cost_model():
    cost_model = YourModel()
    cost_model.load_state_dict(torch.load('./data/lab2-learning.pt'))
    return cost_model


def predict(model, test_plan):
    collector = PlanFeatureCollector()
    vec = collector.walk_operator_tree(test_plan)
    # YOUR CODE HERE: maybe you need padding the features if you choose the second way to extract the features.
    padding_size = 80

    if len(vec) >= 80:
        vec = vec[:padding_size]
    while len(vec) < 80:
        vec.append(0)

    features = torch.Tensor(vec)

    return model(features).tolist()[0]
