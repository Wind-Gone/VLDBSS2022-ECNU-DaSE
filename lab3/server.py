import json
from http.server import HTTPServer, BaseHTTPRequestHandler

import card_from_data
import cost_calibration
import cost_learning
import range_query as rq

host = ('localhost', 8888)

cost_model = None
spn = None


class Resquest(BaseHTTPRequestHandler):
    def handle_cardinality_estimate(self, req_data):
        # YOUR CODE HERE: use your model in lab1
        # print("cardinality_estimate post_data: " + str(req_data))
        range_query = rq.ParsedRangeQuery.parse_range_query("select * from imdb.title where " + str(req_data)[2:-1])
        sel = spn.estimate(range_query)
        return {"selectivity": sel, "err_msg": ""}  # return the selectivity

    def handle_cost_estimate(self, req_data):
        print(req_data)
        # YOUR CODE HERE: use your model in lab2
        # print("cost_estimate post_data: " + str(req_data)[2:-1])
        # print(json.loads(str(req_data)[2:-1]))
        cost = cost_learning.predict(cost_model, json.loads(str(req_data)[2:-1]))
        cost = cost
        return {"cost": cost, "err_msg": ""}  # return the cost

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        req_data = self.rfile.read(content_length)
        resp_data = ""
        if self.path == "/cardinality":
            resp_data = self.handle_cardinality_estimate(req_data)
        elif self.path == "/cost":
            resp_data = self.handle_cost_estimate(req_data)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(resp_data).encode())


if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    cost_model = cost_learning.load_cost_model()
    cost_model2 = cost_calibration.load_cost_model()
    spn = card_from_data.SPN.construct_for_imdb_title('./data/title_sample_20000.csv', 100)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
