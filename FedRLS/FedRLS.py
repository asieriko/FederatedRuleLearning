import os,sys
sys.path.append(os.path.abspath(os.path.join("..","ex-fuzzy","ex_fuzzy")))

import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
import ex_fuzzy.eval_tools as eval_tools
import ex_fuzzy.persistence as persistence
import ex_fuzzy.vis_rules as vis_rules

from .fuzzy_functions import comparison, antecedent_comparison
from .ex_fuzzy_manager import parse_rule_base
from .global_agent import GlobalAgent
from .local_agent import LocalAgent


class FedRLS:
    def __init__(self, datasets, model_parameters):
        self.clients = {}
        self.datasets = datasets
        self.server = GlobalAgent(model_parameters)
        self.random_seed = model_parameters["random_seed"]

    # initialize the clients with their data
    # intialize the server
    # present the clients to the server (it doesn't have to know the individual data)

    def init_clients(self, datasets):
        for i in range(len(datasets)):
            la = LocalAgent(datasets[i],name=f"Client {i}",random_seed=self.random_seed)
            self.clients[i] = {"agent": la, "dataset": datasets[i]}
        self.server.set_clients(self.clients)

    def clients_rules(self):
        rules = []
        for client in self.clients.values():
            rules.append(client["agent"].fl_classifier.rule_base)
        return rules

    def local(self):
        self.init_clients(self.datasets)
        return self.server.main()
        # vis_rules.plot_fuzzy_variable(r.antecedents[0])
        # vis_rules.visualize_rulebase(r)