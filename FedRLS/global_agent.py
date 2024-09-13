<<<<<<< HEAD
from functools import partial
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
import concurrent.futures
import copy
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
from .ex_fuzzy_manager import parse_rule_base, create_rule_base

def run_process(agent, rule_base):
    agent.fit(rule_base)

def init_agent(agent, fz_type_studied):
    agent.fz_type_studied = fz_type_studied
    agent.fit()

def fit_agent(agent, rule_base):
    agent.fit(rule_base)

def fit_agent_rulebase(rule_base):
    f = partial(fit_agent, rule_base=rule_base)

class GlobalAgent():
    
    def __init__(self):
        # model parameters:
        self.n_gen = 100
        self.n_pop = 100

        self.nRules = 15
        self.nAnts = 4
        self.vl = 3
        self.tolerance = 0.001
        self.fz_type_studied = fs.FUZZY_SETS.t1

        self.runner = 1  # 1: single thread, 2+: corresponding multi-thread

        self.sim_threshold = 0.7
        self.contradictory_factor = 0.7  # threshold to consider two rules with different consequents contradictory

    def set_clients(self, clients):
        self.clients = clients
        for client in self.clients.values():
            client['agent'].set_model(self.nRules, self.nAnts, self.fz_type_studied, self.tolerance,self.runner, self.n_gen,self. n_pop)

    def start(self):
        print("Start")
        agents = [client['agent'] for client in self.clients.values()]

        # with concurrent.futures.ProcessPoolExecutor(max_workers=len(agents)) as executor:
        #     futures = [executor.submit(init_agent, agent) for agent in agents]
        #     concurrent.futures.wait(futures)

        # with ThreadPool (processes=4) as pool:
        #      pool.map(init_agent, agents)

        # for client in self.clients.values():
        #     pool.apply_async(target=run_process, args=(client['agent'],))
        for agent in agents:
            init_agent(agent, self.fz_type_studied)
        print("Started")



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

    def combine_domains(self, d1, d2 = []):
        if not d2:
            return d1
        return [[min(a[0],b[0]),max(a[1],b[1])] for a,b in zip(d1, d2)]

    def domains_from_partitions(self, partitions):
        return [p[0].domain for p in partitions]

    def extract_rule_bases(self):
        rules = []
        rules_by_class = defaultdict(list)
        domains = []
        for client in self.clients.values():
            # For each client is necessary to obtain:
            # the domain, so all of them can be merged
            # the rules with the parametrized antecedents, to compare
            client_rule_base = client['agent'].fl_classifier.rule_base
            rule_base_matrix_params, ndomains, variable_names = parse_rule_base(client_rule_base)
            domains = self.combine_domains(ndomains, domains)
            lv = self.linguistic_variables(client_rule_base.antecedents)

            rules.extend(rule_base_matrix_params)

            for rule in rule_base_matrix_params:
                rules_by_class[rule["class"]].append({
                    "var_shape": rule["var_shape"],
                    "var_idx": rule["var_idx"],
                    "class":rule["class"],
                    "score":rule["score"]})
                # rules_by_class[rule[-1]].append([rule[:-1],rule[-2],rule[-1]])
                # TODO: add some metric to aid in the rule selection - rule.score, rule.confidence, rule.accuracy, rule.support

        df = pd.DataFrame(np.array(domains).T,columns=variable_names)
        partitions = utils.construct_partitions(df, self.fz_type_studied)

        return rules, rules_by_class, partitions


    def find_contradictory_rules(self, rules_by_class):
        # Find contradictory rules. High simmilarity on the antecedents but different class
        # the measure as it is, returns the minimum comparison value among all antecedents
        # it can happen that for rules with one antecedent if another has the same term
        # that the value could be high and not contradictory
        contradictory_rules = []
        classes = list(rules_by_class.keys())
        for k1_idx in range(len(classes)):
            k1 = classes[k1_idx]
            nrk1 = len(rules_by_class[k1])
            for k2_idx in classes[k1_idx+1:]:
                k2 = classes[k2_idx]
                if k2 == k1:
                    continue
                nrk2 = len(rules_by_class[k2])
                for i1 in range(nrk1):
                    for i2 in range(nrk2):
                        cj = antecedent_comparison(rules_by_class[k1][i1]["var_shape"] ,rules_by_class[k2][i2]["var_shape"])
                        if cj > self.contradictory_factor:
                            contradictory_rules.append([k1,i1,k2,i2,cj])  

        return contradictory_rules

    def find_similar_rules(self, rules_by_class):
        # Test similarity among the rules in the same class
        similar_rules = {}
        classes = list(rules_by_class.keys())
        for k in classes:
            nrk = len(rules_by_class[k])
            com_mat_k = np.zeros((nrk, nrk))
            for i1 in range(nrk):
                for i2 in range(i1+1,nrk):
                    cj = comparison([rules_by_class[k][i1]["var_shape"], rules_by_class[k][i1]["class"]],[rules_by_class[k][i2]["var_shape"],rules_by_class[k][i2]["class"]] )
                    com_mat_k[i1][i2] = cj
                    # or instead of the matrix, directly create the list, like above
            similar_rules[k] = np.argwhere(com_mat_k>self.sim_threshold)

        return similar_rules


    def compare_rule_bases(self, rules_by_class):
        contradictory_rules = self.find_contradictory_rules(rules_by_class)               

        print("contradictory_rules")
        print(contradictory_rules)

        # Delete from higher rule index to lower
        # Determine first wich rule to delete (lower score)
        # del rules_by_class[class_i][rule_i] 

        similar_rules = self.find_similar_rules(rules_by_class)
        
        print("similar_rules")
        classes = list(rules_by_class.keys())
        for k in classes:
            if len(similar_rules[k]) > 0:
                # rules_by_class[k][similar_rules[k][0][0]]["score"]  
                print(similar_rules[k])

        return rules_by_class

    def update_clients(self, new_rule_base): # new rule base with selected rules
        # agents = [client['agent'] for client in self.clients.values()]
        for client in self.clients.values():
            client['agent'].update_rule_base(copy.deepcopy(new_rule_base))

    def eval_clients(self):
        for client in self.clients.values():
            client['agent'].eval_test()

    def print_clients(self):
        for client in self.clients.values():
            print(client['agent'].fl_classifier.rule_base)

    def retrain_clients(self):
        for client in self.clients.values():
            client['agent'].fit()


    def main(self):
        print("First Train")
        self.start()
        rules, rules_by_class, partitions = self.extract_rule_bases()
        rules_by_class = self.compare_rule_bases(rules_by_class)
        nrb = create_rule_base(partitions, rules_by_class)
        self.eval_clients()
        print("Update Clients")
        self.update_clients(nrb)
        self.eval_clients()
        print("Retrain")
        self.retrain_clients()
        rules, rules_by_class, partitions = self.extract_rule_bases()
        rules_by_class = self.compare_rule_bases(rules_by_class)
        nrb = create_rule_base(partitions, rules_by_class)
        self.eval_clients()
        print("Update Clients")
        self.update_clients(nrb)
        self.eval_clients()
        # self.print_clients()
=======
from functools import partial
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
import concurrent.futures
import copy
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
from .ex_fuzzy_manager import parse_rule_base, create_rule_base


def train_client(agent, rule_base=None):
    if rule_base:
        agent.fit(rule_base)
    else:
        agent.fit()

def train_agent_rulebase(rule_base):
    f = partial(fit_agent, rule_base=rule_base)

class GlobalAgent():
    
    def __init__(self, model_parameters):
        # model parameters:
        self.n_gen = model_parameters["n_gen"]
        self.n_pop = model_parameters["n_pop"]

        self.nRules = model_parameters["nRules"]
        self.nAnts = model_parameters["nAnts"]
        self.vl = 3 #model_parameters["vl"] # FIXME: Where is it used??
        self.tolerance = model_parameters["tolerance"]
        self.fz_type_studied = model_parameters["fz_type_studied"]

        self.runner = model_parameters["runner"]  # 1: single thread, 2+: corresponding multi-thread

        self.sim_threshold = model_parameters["sim_threshold"]
        self.contradictory_factor = model_parameters["contradictory_factor"]  # threshold to consider two rules with different consequents contradictory

    def set_clients(self, clients):
        self.clients = clients
        agents_mm = np.array([client['agent'].get_max_min() for client in self.clients.values()])
        X_range = np.stack([np.min(agents_mm[:,0],axis=0),np.max(agents_mm[:,1],axis=0)])
        precomputed_partitions = utils.construct_partitions(X_range, self.fz_type_studied)
        for client in self.clients.values():
            client['agent'].set_model(self.nRules, self.nAnts, self.fz_type_studied, self.tolerance,self.runner, self.n_gen,self. n_pop, precomputed_partitions)

    def start(self):
        agents = [client['agent'] for client in self.clients.values()]

        # with concurrent.futures.ProcessPoolExecutor(max_workers=len(agents)) as executor:
        #     futures = [executor.submit(train_client, agent) for agent in agents]
        #     concurrent.futures.wait(futures)

        # with ThreadPool (processes=4) as pool:
        #      pool.map(train_client, agents)

        # for client in self.clients.values():
        #     pool.apply_async(target=train_client, args=(client['agent'],))
        for agent in agents:
            train_client(agent)
            #  agent.fit()  # FIXME: is the same



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

    def combine_domains(self, d1, d2 = []):
        if not d2:
            return d1
        return [[min(a[0],b[0]),max(a[1],b[1])] for a,b in zip(d1, d2)]

    def domains_from_partitions(self, partitions):
        return [p[0].domain for p in partitions]

    def extract_rule_bases(self):
        rules = []
        rules_by_class = defaultdict(list)
        domains = []
        for client in self.clients.values():
            # For each client is necessary to obtain:
            # the domain, so all of them can be merged
            # the rules with the parametrized antecedents, to compare
            client_rule_base = client['agent'].fl_classifier.rule_base
            rule_base_matrix_params, ndomains, variable_names = parse_rule_base(client_rule_base)
            domains = self.combine_domains(ndomains, domains)
            lv = self.linguistic_variables(client_rule_base.antecedents)

            rules.extend(rule_base_matrix_params)

            for rule in rule_base_matrix_params:
                rules_by_class[rule["class"]].append({
                    "var_shape": rule["var_shape"],
                    "var_idx": rule["var_idx"],
                    "class":rule["class"],
                    "score":rule["score"]})
                # rules_by_class[rule[-1]].append([rule[:-1],rule[-2],rule[-1]])
                # TODO: add some metric to aid in the rule selection - rule.score, rule.confidence, rule.accuracy, rule.support

        df = pd.DataFrame(np.array(domains).T,columns=variable_names)
        partitions = utils.construct_partitions(df, self.fz_type_studied)

        return rules, rules_by_class, partitions


    def find_contradictory_rules(self, rules_by_class):
        # Find contradictory rules. High simmilarity on the antecedents but different class
        # the measure as it is, returns the minimum comparison value among all antecedents
        # it can happen that for rules with one antecedent if another has the same term
        # that the value could be high and not contradictory
        contradictory_rules = []
        classes = list(rules_by_class.keys())
        for k1_idx in range(len(classes)):
            k1 = classes[k1_idx]
            nrk1 = len(rules_by_class[k1])
            for k2_idx in classes[k1_idx+1:]:
                k2 = classes[k2_idx]  # FIXME: I got an error here with the digits dataset and random 23
                if k2 == k1:
                    continue
                nrk2 = len(rules_by_class[k2])
                for i1 in range(nrk1):
                    for i2 in range(nrk2):
                        cj = antecedent_comparison(rules_by_class[k1][i1]["var_shape"] ,rules_by_class[k2][i2]["var_shape"])
                        if cj > self.contradictory_factor:
                            contradictory_rules.append([k1,i1,k2,i2,cj])  

        return contradictory_rules

    def find_similar_rules(self, rules_by_class):
        # Test similarity among the rules in the same class
        similar_rules = {}
        classes = list(rules_by_class.keys())
        for k in classes:
            nrk = len(rules_by_class[k])
            com_mat_k = np.zeros((nrk, nrk))
            for i1 in range(nrk):
                for i2 in range(i1+1,nrk):
                    cj = comparison([rules_by_class[k][i1]["var_shape"], rules_by_class[k][i1]["class"]],[rules_by_class[k][i2]["var_shape"],rules_by_class[k][i2]["class"]] )
                    com_mat_k[i1][i2] = cj
                    # or instead of the matrix, directly create the list, like above
            similar_rules[k] = np.argwhere(com_mat_k>self.sim_threshold)

        return similar_rules


    def compare_rule_bases(self, rules_by_class):
        contradictory_rules = self.find_contradictory_rules(rules_by_class)               

        # print("contradictory_rules")
        # print(contradictory_rules)

        # Delete from higher rule index to lower
        # Determine first wich rule to delete (lower score)
        # del rules_by_class[class_i][rule_i] 

        similar_rules = self.find_similar_rules(rules_by_class)
        # TODO: I commented the previos print and following block
        # print("similar_rules")
        # classes = list(rules_by_class.keys())
        # for k in classes:
        #     if len(similar_rules[k]) > 0:
        #         # rules_by_class[k][similar_rules[k][0][0]]["score"]  
        #         print(similar_rules[k])

        return rules_by_class

    def eval_clients(self):
        performances = []
        for client in self.clients.values():
            performances.append(client['agent'].eval_test())
        print(f"fLOG: {np.mean(performances)=}")


    def print_clients(self):
        for client in self.clients.values():
            print(client['agent'].fl_classifier.rule_base)

    def update_clients(self, new_rule_base): # new rule base with selected rules
        # agents = [client['agent'] for client in self.clients.values()]
        for client in self.clients.values():
            client['agent'].update_rule_base(copy.deepcopy(new_rule_base))

    def retrain_clients(self, master_rule_base):
        for client in self.clients.values():
            client['agent'].fit(candidate_rules=master_rule_base)

    def train_clients(self, master_rule_base=None):
        for client in self.clients.values():
            if master_rule_base:
                client['agent'].fit(candidate_rules=master_rule_base)
            else:
                client['agent'].fit()

    def main(self):
        print(f"LOG:,0-Train")
        agents_mm = np.array([client['agent'].get_max_min() for client in self.clients.values()])
        X_range = np.stack([np.min(agents_mm[:,0],axis=0),np.max(agents_mm[:,1],axis=0)])
        precomputed_partitions = utils.construct_partitions(X_range, self.fz_type_studied)
        print(f"LOG:,1stTrain")
        self.start()
        rules, rules_by_class, partitions = self.extract_rule_bases()
        # rules_by_class = self.compare_rule_bases(rules_by_class)
        nrb = create_rule_base(partitions, rules_by_class)
        self.eval_clients()
        print(f"LOG:,1stUpdate")
        self.update_clients(nrb)
        self.eval_clients()
        n_retrains = 3
        for i in range(n_retrains):
            print(f"LOG:,{i}-Retrain clients")
            self.retrain_clients(nrb)
            rules, rules_by_class, partitions = self.extract_rule_bases()
            # rules_by_class = self.compare_rule_bases(rules_by_class)
            nrb = create_rule_base(partitions, rules_by_class)
            self.eval_clients()
            print(f"LOG:,{i}-update clients")
            self.update_clients(nrb)
            self.eval_clients()
            # self.print_clients()
>>>>>>> 04c9cba (add a parameter dictionary with experiment configuration)
