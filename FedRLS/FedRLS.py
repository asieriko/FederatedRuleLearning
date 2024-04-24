from functools import partial
from multiprocessing import Process, Pool
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

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

class LocalAgent:
    pass

    def __init__(self, dataset):
        self.dataset = dataset
        # Import  data
        self.X = dataset[0]
        self.y = dataset[1]

        # Split the data into a training set and a test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                                                random_state=0)

        self.fl_classifier = None
        self.rule_base = None  # Or take them directly from fl_classifier
        self.performance = None
    def set_model(self, model, n_gen, n_pop):
        self.fl_classifier = model
        self.n_gen = n_gen
        self.n_pop = n_pop

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.X = dataset[0]
        self.y = dataset[1]

    def fit(self, mrule_base=None):
        if mrule_base:
            self.fl_classifier.load_master_rule_base(mrule_base)

        # fl_classifier.customized_loss(utils.mcc_loss)
        self.fl_classifier.fit(self.X_train, self.y_train, n_gen=self.n_gen, pop_size=self.n_pop)
        self.rule_base = self.fl_classifier.rule_base
        # str_rules = eval_tools.eval_fuzzy_model(self.fl_classifier, self.X_train, self.y_train, self.X_test, self.y_test,
        #                                         plot_rules=True, print_rules=True, plot_partitions=True,
        #                                         return_rules=True)

        self.performance = self.fl_classifier.performance
        print(self.fl_classifier.performance)
        print(self.fl_classifier.rule_base)

        return self.rule_base
        # antecedents list of  ex_fuzzy.fuzzy_sets.fuzzyVariable
        # consequent_names list of string
        # rule_bases a list of

def run_process(agent, rule_base):
    agent.fit(rule_base)

def init_agent(agent):
    agent['agent'].fit()


class GlobalAgent():

    def __init__(self):
        # model parameters:
        self.n_gen = 30
        self.n_pop = 50

        self.nRules = 15
        self.nAnts = 4
        self.vl = 3
        self.tolerance = 0.001
        self.fz_type_studied = fs.FUZZY_SETS.t1

        self.runner = 1  # 1: single thread, 2+: corresponding multi-thread

    def set_clients(self, clients):
        self.clients = clients
        for client in self.clients.values():
            class_names = np.unique(client['dataset'][1])
            model = GA.BaseFuzzyRulesClassifier(
                nRules=self.nRules,
                nAnts=self.nAnts,
                n_linguist_variables=self.vl, 
                fuzzy_type=self.fz_type_studied, 
                verbose=False,
                tolerance=self.tolerance, 
                class_names=class_names,
                runner=self.runner)
            client['agent'].set_model(model, self.n_gen, self.n_pop)

    def start(self):
        print("Start")
        c = [client for client in self.clients.values()]
        # with Pool(processes=4) as pool:
        #     pool.map(init_agent, c)
        for ci in c:
            init_agent(ci)
        print("Started")
            # for client in self.clients.values():
            #     pool.apply_async(target=run_process, args=(client['agent'],))


    # initialize clients, send model template? partial GA.BaseFuzzyRulesClassifier
    # receive data

    def linguistic_variables(self, antecedents):
        # TODO: Adapt it to other fuzzysets T2, etc
        ant_ret = []
        for ant in antecedents:
            ant_i = []
            domain = ant.linguistic_variables[0].domain
            for lv in ant.linguistic_variables:
                membership = lv.membership_parameters
                ant_i.append(np.hstack([domain[0],membership,domain[1]]))
            ant_ret.append(ant_i)
        return ant_ret

    def rule_antecedents(self, antecedents, lv):
        ant_i = []
        for i,ant in enumerate(antecedents):
                if ant != -1:
                    ant_i.append(lv[i][ant])
                else:
                    ant_i.append(-1)  #TODO: Maybe treat this in some specific way?
        return ant_i

    def compare_rule_bases(self):
        rules = []
        rules_by_class = defaultdict(list)
        for client in self.clients.values():
            client_rule_base = client['agent'].fl_classifier.rule_base
            parse_rule_base(client_rule_base)
            lv = self.linguistic_variables(client_rule_base.antecedents)

            for idx, rb in  enumerate(client_rule_base):
                for rule in rb:
                    rules.append([self.rule_antecedents(rule.antecedents, lv),client_rule_base.consequent_names[idx]])
                    # rules_by_class[client_rule_base.consequent_names[idx]].append(self.rule_antecedents(rule.antecedents, lv))
                    rules_by_class[client_rule_base.consequent_names[idx]].append([self.rule_antecedents(rule.antecedents, lv),client_rule_base.consequent_names[idx]])
                    # TODO: add some metric to aid in the rule selection - rule.score, rule.confidence, rule.accuracy, rule.support
                
        # Find contradictory rules. High simmilarity on the antecedents but different class
        contradictory_rules = []
        for k1 in rules_by_class.keys():
            nrk1 = len(rules_by_class[k1])
            for k2 in rules_by_class.keys():
                if k2 == k1:
                    continue
                nrk2 = len(rules_by_class[k2])
                for i1 in range(nrk1):
                    for i2 in range(nrk2):
                        cj = antecedent_comparison(rules_by_class[k1][i1][0],rules_by_class[k2][i2][0])
                        if cj > 0.7:
                            contradictory_rules.append([k1,i1,k2,i2,cj])                 

        print(contradictory_rules)

        # Test similarity among the rules in the same class
        for k in rules_by_class.keys():
            nrk = len(rules_by_class[k])
            com_mat_k = np.zeros((nrk, nrk))
            for i1 in range(nrk):
                for i2 in range(i1+1,nrk):
                    cj = comparison(rules_by_class[k][i1],rules_by_class[k][i2])
                    com_mat_k[i1][i2] = cj
        
        nr = len(rules)
        com_mat = np.zeros((nr, nr))
        for i1 in range(nr):
            for i2 in range(i1+1,nr):
                cj = comparison(rules[i1],rules[i2])
                com_mat[i1][i2] = cj
            

        print(com_mat)
            # TODO: I believe that we have to homogeinize the lingustic variables to the maximum posible range
            # But, before comparing or after?
            # rules if x is A -> B and if x is A and y is C -> B
            #self How to compoare? What chose? the more restricitve or the lesser
            # client['agent'].rule_base.consequent_names
            # rb = client['agent'].fl_classifier.rule_base
            # rb.rule_bases[0].get_rulebase_matrix()
            # c
            # rb.antecedents[0].linguistic_variables[0].secondMF_lower
            # rb.antecedents[0].linguistic_variables[0].secondMF_upper
            # rb.antecedents[0].linguistic_variables[0].lower_height
        # the resulting selected rules have to have a list with the highest comparison value to each clients rule-base

    def update_clients(self):
        pass

    def main(self):
        self.start()
        self.compare_rule_bases()
        self.update_clients()


class FedRLS:
    def __init__(self, datasets):
        self.clients = {}
        self.server = None
        self.datasets = datasets
        self.server = GlobalAgent()

    # initialize the clients with their data
    # intialize the server
    # present the clients to the server (it doesn't have to know the individual data)

    def init_clients(self, datasets):
        for i in range(len(datasets)):
            la = LocalAgent(datasets[i])
            self.clients[i] = {"agent": la, "dataset": datasets[i]}
        self.server.set_clients(self.clients)

    def local(self):
        self.init_clients(self.datasets)
        self.server.main()
        # vis_rules.plot_fuzzy_variable(r.antecedents[0])
        # vis_rules.visualize_rulebase(r)