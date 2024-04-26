from functools import partial
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
import concurrent.futures
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

def run_process(agent, rule_base):
    agent.fit(rule_base)

def init_agent(agent):
    agent.fit()

def fit_agent(agent, rule_base):
    agent.fit(rule_base)

def fit_agent_rulebase(rule_base):
    f = partial(fit_agent, rule_base=rule_base)

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

        self.sim_threshold = 0.7
        self.contradictory_factor = 0.7  # threshold to consider two rules with different consequents contradictory

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
        agents = [client['agent'] for client in self.clients.values()]

        # with concurrent.futures.ProcessPoolExecutor(max_workers=len(agents)) as executor:
        #     futures = [executor.submit(init_agent, agent) for agent in agents]
        #     concurrent.futures.wait(futures)

        # with ThreadPool (processes=4) as pool:
        #      pool.map(init_agent, agents)
        for agent in agents:
            init_agent(agent)
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
                rules_by_class[rule[-1]].append([rule[:-1],rule[-1]])
                # TODO: add some metric to aid in the rule selection - rule.score, rule.confidence, rule.accuracy, rule.support

        df = pd.DataFrame(np.array(domains).T,columns=variable_names)
        partitions = utils.construct_partitions(df, self.fz_type_studied)

        return rules, rules_by_class, partitions
                
    def compare_rule_bases(self, rules, rules_by_class):
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
                        if cj > self.contradictory_factor:
                            contradictory_rules.append([k1,i1,k2,i2,cj])                 

        print(contradictory_rules)

        # Test similarity among the rules in the same class
        similar_rules = {}
        for k in rules_by_class.keys():
            nrk = len(rules_by_class[k])
            com_mat_k = np.zeros((nrk, nrk))
            for i1 in range(nrk):
                for i2 in range(i1+1,nrk):
                    cj = comparison(rules_by_class[k][i1],rules_by_class[k][i2])
                    com_mat_k[i1][i2] = cj
                    # or instead of the matrix, directly create the list, like above
            similar_rules[k] = np.argwhere(com_mat_k>self.sim_threshold)
        
        print(similar_rules)

    def update_clients(self, partitions): # new rule base with selected rules
        agents = [client['agent'] for client in self.clients.values()]
        for agent in agents:
            agent.rule_base.antecedets = partitions
            # for rb in agent.rule_bases:  # Not sure if this is needed
            #     rb.antecedets = partitions

    def main(self):
        self.start()
        rules, rules_by_class, partitions = self.extract_rule_bases()
        self.compare_rule_bases(rules, rules_by_class)
        self.update_clients(partitions)