from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.abspath(os.path.join("..","ex-fuzzy","ex_fuzzy")))

import ex_fuzzy.utils as utils
from .fuzzy_functions import comparison, antecedent_comparison
from .ex_fuzzy_manager import parse_rule_base, create_rule_base
from .models import  FedRLSmodel, FRBCmodel


class GlobalAgent():
    
    def __init__(self, frbc_parameters: FRBCmodel, fedrls_parameters: FedRLSmodel):
        # model parameters:
        self.frbc = frbc_parameters

        self.sim_threshold = fedrls_parameters.sim_threshold
        self.contradictory_factor = fedrls_parameters.contradictory_factor  # threshold to consider two rules with different consequents contradictory
        self.max_retrains = fedrls_parameters.max_retrains
        self.adaptative = fedrls_parameters.adaptative

        self.nrb = None

    def set_clients(self, clients):
        self.clients = clients
        agents_mm = np.array([client['agent'].get_max_min() for client in self.clients.values()])
        X_range = np.stack([np.min(agents_mm[:,0],axis=0),np.max(agents_mm[:,1],axis=0)])
        precomputed_partitions = utils.construct_partitions(X_range, self.frbc.fz_type_studied) # maybe make a model variable?
        for client in self.clients.values():
            client['agent'].set_model(self.frbc, self.adaptative, precomputed_partitions)

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
        partitions = utils.construct_partitions(df, self.frbc.fz_type_studied)

        return rules, rules_by_class, partitions  # Partitions are not necessary as they are all the same


    def find_contradictory_rules(self, rules_by_class):
        # Find contradictory rules. High simmilarity on the antecedents but different class
        # the measure as it is, returns the minimum comparison value among all antecedents
        # it can happen that for rules with one antecedent if another has the same term
        # that the value could be high and not contradictory
        # Returns a list of list of contradictory pairs, each of them:
        # [class_rule1,index_rule1,k2,i2,comparison_value]
        contradictory_rules = []
        classes = list(rules_by_class.keys())
        for k1_idx in range(len(classes)):
            k1 = classes[k1_idx]
            nrk1 = len(rules_by_class[k1])
            for k2_idx in range(len(classes[k1_idx:])):
                k2 = classes[k2_idx]  # FIXME: I got an error here with the digits dataset and random 23
                if k2 == k1:
                    continue
                nrk2 = len(rules_by_class[k2])
                for i1 in range(nrk1):
                    for i2 in range(nrk2):
                        cj = antecedent_comparison(rules_by_class[k1][i1]["var_shape"] ,rules_by_class[k2][i2]["var_shape"])
                        if cj > self.contradictory_factor:
                            contradictory_rules.append([k1,i1,k2,i2,cj])
        # TODO: Simplify. There are duplicated
        # [[1, 8, 2, 0, 1], [2, 0, 1, 8, 1]]
        return contradictory_rules

    def find_similar_rules(self, rules_by_class):
        # Test similarity among the rules in the same class
        similar_rules = {}
        triplets = {}
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
            triplets[k] = [(i, j, com_mat_k[i, j]) for i, j in similar_rules[k]]

        return similar_rules, triplets


    def compare_rule_bases(self, rules_by_class):
        contradictory_rules = self.find_contradictory_rules(rules_by_class)               

        if len(contradictory_rules) != 0:
            # TODO: Maybe take into accoutn each rules score
            # if one of them is high and the other low, remove the later
            # print("contradictory_rules")
            # print(contradictory_rules)
            to_delete = defaultdict(set)
            for rule in contradictory_rules:
                to_delete[rule[0]].add(rule[1])
                to_delete[rule[2]].add(rule[3])
            for k,v in to_delete.items():
                for vi in sorted(v,reverse=True):
                    del rules_by_class[k][vi]
            # Delete from higher rule index to lower
            # Determine first wich rule to delete (lower score)
            # rules_by_class[class_i][rule_i] ["score"]
            # del rules_by_class[class_i][rule_i] 


        similar_rules, similar_rules_compvalue = self.find_similar_rules(rules_by_class)
        similar_rules = {k:v.tolist() for k,v in similar_rules.items()}
        # 1st remove equal rules
        to_delete = defaultdict(set)
        for k, v in similar_rules_compvalue.items():
            for ci in v:
                if ci[2] == 1.:
                    # print(k,ci)
                    sc1 = rules_by_class[k][ci[0]]["score"]
                    sc2 = rules_by_class[k][ci[1]]["score"]
                    if sc1 > sc2:
                        to_delete[k].add(ci[1])
                    else:
                        to_delete[k].add(ci[0])
        for k, v in to_delete.items():
            for vi in sorted(v, reverse=True):
                del rules_by_class[k][vi]
        for k, v in similar_rules_compvalue.items():
            for i, ci in reversed(list(enumerate(v))): # ORDER!!
                if ci[0] in to_delete[k] or  ci[1] in to_delete[k]:
                    # print(k,i)
                    del similar_rules_compvalue[k][i]
                    del similar_rules[k][i]
        num_rules = sum([len(x) for x in rules_by_class.values()])

        # select rules based on the score
        if num_rules > self.frbc.nRules:
            classes = list(rules_by_class.keys())

            rules_scores = []
            for k in classes:
                rules_scores.extend([[k,i, y["score"]] for i, y  in enumerate(rules_by_class[k])])
            # select self.nRules with the largest score
            rules_scores_sorted = sorted(rules_scores, key=lambda x: x[-1],reverse=True)
            selected = rules_scores_sorted[:self.frbc.nRules]
            selected_classes = np.unique(np.array(selected)[:, 0])
            # In case a class has not been selected, we remove the worst score's rule
            # and add the best for that class
            if len(selected_classes) != len(classes):
                for k in classes:
                    if k not in selected_classes:
                        for inner_list in rules_scores_sorted:
                            if inner_list[0] == k:
                                selected[-1] = inner_list
                                break

            # print(selected)
            selected_rules_by_class = {}
            for element in selected:
                key, rule_number, _ = element
                if key in rules_by_class:
                    if 0 <= rule_number < len(rules_by_class[key]):
                        if key not in selected_rules_by_class:
                            selected_rules_by_class[key] = []
                        selected_rules_by_class[key].append(rules_by_class[key][rule_number])
        else:
            selected_rules_by_class = rules_by_class


            # similar_rules_compvalue_list = []
            # for key, list_of_lists in similar_rules_compvalue.items():
            #     # Iterate through each inner list
            #     for inner_list in list_of_lists:
            #         # Prepend the key to the inner list
            #         modified_list = [key] + list(inner_list)
            #         # Append the modified list to the result list
            #         similar_rules_compvalue_list.append(modified_list)
            #
            # for k in classes:
            #     # Delete equal rules -> Comp = 1
            #     # The rest comparison are not very fair 2 vs -1 ??
            #     conflincting_rules, counts = np.unique(np.array(similar_rules_compvalue[k])[:,:2],return_counts=True)
            #     scores = [r["score"] for r in rules_by_class[0]]
            #     scores = [scores[y] for y in [int(x) for x in conflincting_rules]]
            #     if len(similar_rules[k]) > 0:
            #         # rules_by_class[k][similar_rules[k][0][0]]["score"]
            #         print(similar_rules[k])

        return selected_rules_by_class

    def eval_clients(self):
        performances = []
        for client in self.clients.values():
            performances.append(client['agent'].eval_test())
        # print(f"LOG: AVG {np.mean([x['final_accuracy']['accuracy'] for x in performances])}")
        return performances

    def print_clients_rulebase(self):
        for client in self.clients.values():
            print(client['agent'].fl_classifier.rule_base)

    def update_clients_rulebase(self, new_rule_base): # new rule base with selected rules
        # agents = [client['agent'] for client in self.clients.values()]
        for client in self.clients.values():
            client['agent'].update_rule_base(copy.deepcopy(new_rule_base))

    def train_clients(self, master_rule_base=None):
        for client in self.clients.values():
            if master_rule_base:
                client['agent'].fit(candidate_rules=master_rule_base)
            else:
                client['agent'].fit()
        # max_workers = 5
        # with ProcessPoolExecutor(max_workers=max_workers) as executor:
        #     for client in self.clients.values():
        #         if master_rule_base:
        #             # executor.submit(client['agent'].fit(candidate_rules=master_rule_base))
        #             executor.submit(client['agent'].fit,candidate_rules=master_rule_base)
        #         else:
        #             # executor.submit(client['agent'].fit())
        #             executor.submit(client['agent'].fit)
        # print('All done!')

    def main(self):
        clients_performances = []
        for i in range(self.max_retrains+1):
            # print(f"LOG:,{i}-Retrain clients")
            # Train the clients, the first time ther is no rule base and it is created on each
            # in subsequent times the rule base from the server is provided
            self.train_clients(self.nrb)  # FIXME: is this needed if the clientes are updated with nrb befor?
            # Extract rules learned from each client
            rules, rules_by_class, partitions = self.extract_rule_bases()
            # Compare and aggregate the rules -> nrb
            rules_by_class = self.compare_rule_bases(rules_by_class)
            consequent_names = self.clients[0]["agent"].fl_classifier.rule_base.consequent_names
            # partitions = self.clients[0]["agent"].fl_classifier.rule_base.antecedents
            self.nrb = create_rule_base(partitions, rules_by_class, consequent_names)
            # Test the clients with their learned rules
            clients_performances.append({'type':'train','epoch':i,'results':self.eval_clients()})
            # print(f"LOG:,{i}-update clients")
            # Update clients rule base with the one from the server
            self.update_clients_rulebase(self.nrb)
            # Test the clients with the global rules
            clients_performances.append({'type':'update','epoch':i,'results':self.eval_clients()})
        # print("End:")
        # print(nrb)
        return rules_by_class, clients_performances