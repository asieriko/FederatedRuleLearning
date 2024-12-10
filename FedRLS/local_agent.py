from functools import partial
from itertools import permutations
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
import ex_fuzzy.eval_rules as eval_rules
import ex_fuzzy.persistence as persistence
import ex_fuzzy.vis_rules as vis_rules

from .fuzzy_functions import comparison, antecedent_comparison
from .ex_fuzzy_manager import parse_rule_base
from .metrics import metrics
from .utils import average_dict
from .models import  FedRLSmodel, FRBCmodel

class LocalAgent:

    def __init__(self, dataset, name="Client"):
        self.name = name
        self.dataset = dataset
        # Import  data
        self.X_train = dataset[0]#.to_numpy()
        self.X_test = dataset[1]#.to_numpy()
        self.y_train = dataset[2]
        self.y_test = dataset[3]

        self.fl_classifier = None
        # Or take them directly from fl_classifier
        self.performance = None
        self.original_activations = None
        self.adaptative_activations = None

    def set_model(self, frbc_parameters: FRBCmodel, adaptative: bool = True, precomputed_partitions = None):
        self.frbc = frbc_parameters

        class_names = np.unique(self.y_train) 
        if precomputed_partitions is None:
            precomputed_partitions = utils.construct_partitions(self.X_train, self.frbc.fz_type_studied)
        model = GA.BaseFuzzyRulesClassifier(
            nRules=self.frbc.nRules,
            nAnts=self.frbc.nAnts,
            fuzzy_type=self.frbc.fz_type_studied,
            verbose=self.frbc.verbose,
            tolerance=self.frbc.tolerance,
            linguistic_variables=precomputed_partitions,
            class_names=class_names,
            runner=self.frbc.runner)
        self.fl_classifier = model
        self.adaptative = adaptative

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.X_train = dataset[0].to_numpy()
        self.X_test = dataset[1].to_numpy()
        self.y_train = dataset[2]
        self.y_test = dataset[3]

    def get_max_min(self):
        return np.stack([self.X_train.min(axis=0),self.X_train.max(axis=0)])

    def fit(self, candidate_rules=None):
        if candidate_rules:
            # self.fl_classifier.load_master_rule_base(candidate_rules)
            self.fl_classifier.fit(self.X_train, self.y_train, candidate_rules=candidate_rules, random_state = self.frbc.random_seed)
        else:
            # fl_classifier.customized_loss(utils.mcc_loss)
            self.fl_classifier.fit(self.X_train, self.y_train, n_gen=self.frbc.n_gen, pop_size=self.frbc.n_pop, random_state = self.frbc.random_seed)
            # str_rules = eval_tools.eval_fuzzy_model(self.fl_classifier, self.X_train, self.y_train, self.X_test, self.y_test,
            #                                         plot_rules=True, print_rules=True, plot_partitions=True,
            #                                         return_rules=True)

        self.performance = self.fl_classifier.performance
        # print(self.fl_classifier.performance)
        # print(self.fl_classifier.rule_base)

        return self.fl_classifier.rule_base
        # antecedents list of  ex_fuzzy.fuzzy_sets.fuzzyVariable
        # consequent_names list of string
        # rule_bases a list of

    def predict(self,X):
        return self.fl_classifier.predict(X)

    def update_rule_base(self, new_rule_base):
        self.fl_classifier.load_master_rule_base(new_rule_base)
    

    def eval_each_rule(self,base_performance):
        # TODO: More complex selection. How to delete more than 1 rule
        # How to deal with non-important rules (label 1)
        activations = []
        for rule_bases in self.fl_classifier.rule_base:
            activations_i = []
            for rule in rule_bases:
                prev_score = rule.score
                rule.score = 0
                performance = self.eval_rule_base(self.fl_classifier.rule_base)["accuracy"]  # There are other metrics also
                rule.score = prev_score
                if performance == base_performance:
                    activations_i.append(1)
                elif performance < base_performance:
                    activations_i.append(2)
                else:
                    activations_i.append(0)
                # activations_i.append(base_performance - performance)  # if < 0 bad, =0 no effect, >0 important rule
            activations.append(activations_i)
        # [len(rb.rules) for rb in self.fl_classifier.rule_base.rule_bases]
        # to copy shape to activations
        return activations

    def eval_each_0_rule(self,base_performance, activations):
        # TODO: More complex selection. How to delete more than 1 rule
        zeros = []
        for i,cl in enumerate(activations):
            for j,act in enumerate(cl):
                if act == 0:
                    zeros.append([i,j])
        
        # remove two rules
        performances = []
        new_activations = []
        old_activations = []
        perms = []
        for perm in permutations(zeros,2):
            perms.append(perm)
            rz1, rz2 = perm
            rule1 = self.fl_classifier.rule_base[rz1[0]][rz1[1]]
            rule2 = self.fl_classifier.rule_base[rz2[0]][rz2[1]]
            prev_score1 = rule1.score
            prev_score2 = rule2.score
            old_activations = [prev_score1, prev_score2]
            rule1.score = 0
            performance1 = self.eval_rule_base(self.fl_classifier.rule_base)["accuracy"]  # There are other metrics also
            rule1.score = prev_score1
            rule2.score = 0
            performance2 = self.eval_rule_base(self.fl_classifier.rule_base)["accuracy"]  # There are other metrics also
            rule1.score = 0
            performance = self.eval_rule_base(self.fl_classifier.rule_base)["accuracy"]  # There are other metrics also
            rule2.score = prev_score2
            rule1.score = prev_score1
            if performance >= performance1 and performance >= performance2:
                performances.append(performance)
                new_activations.append([0,0])
            elif performance1 > performance and performance1 > performance2:
                performances.append(performance1)
                new_activations.append([0,1])
            elif performance2 > performance and performance2 > performance1:
                performances.append(performance2)
                new_activations.append([1,0])
            else: # p1 > p and p2 > p and p1=p2
                if rule1.score > rule2.score:
                    performances.append(performance1)
                    new_activations.append([1, 0])
                else:
                    performances.append(performance2)
                    new_activations.append([0, 1])
        idx = np.argmax(np.array(performances))
        na = new_activations[idx]
        rls = perms[idx]
        return activations

    def adjust_scores(self, activations):
        # FIXME: If two activations are 0 it doesn't mean that both have to be removed, only "the worse"
        # Or study if both have to be removed, triying first one and then the other.
        original_activations = []
        for rule_bases, acts in zip(self.fl_classifier.rule_base, activations):
            rb_activations = []
            for rule, acti in zip(rule_bases, acts):
                rb_activations.append(rule.score)
                if acti == 0:
                    rule.score = 0
            original_activations.append(rb_activations)
        return original_activations

    def adjust_original_scores(self, scores):
        for rule_bases, acts in zip(self.fl_classifier.rule_base, scores):
            for rule, acti in zip(rule_bases, acts):
                rule.score = acti

    def eval_otherdata(self, datasets):
        if  self.adaptative and  self.original_activations: # Just in case the scores are different
            self.adjust_original_scores(self.original_activations)

        results = []

        for dataset in datasets:
            X_test = dataset[1]
            y_test = dataset[3]    
            predicted = self.fl_classifier.predict(X_test)
            performance = metrics(predicted, y_test)
            results.append(performance)
        
        results = average_dict(results)

        if self.adaptative and self.original_activations:  # Just in case the scores are different
            self.adjust_original_scores(self.original_activations)

        return results

    def eval_rule_base(self, rb):
        if sum([len(rx) for rx in rb]) == 0:  # No rules present, In some cases the rule generation process purgest all
            # of them if they are below the tolerance (one case i saw all the scores 0)
            return {"f1": 0, "precision": 0., "recall":0., "jaccard":0., "accuracy":0., "performance":0.}

        str_rules = eval_tools.eval_fuzzy_model(self.fl_classifier, self.X_train, self.y_train, self.X_test, self.y_test,
                                                     plot_rules=False, print_rules=False, plot_partitions=False,
                                                     print_accuracy=False, print_matthew=False, return_rules=False)

        # FIXME: Breaks, Check empty rules?
        evb = eval_rules.evalRuleBase(rb, self.X_train, self.y_train)  # wich X and y?
        evb.add_classification_metrics()

        predicted = self.fl_classifier.predict(self.X_test, out_class_names=True)
        train_performance = metrics(predicted, self.y_test)
        
        train_performance["train_performance"] = self.fl_classifier.performance

        return train_performance #, precision, recall, f1, jaccard, performance

    def eval_test(self) -> dict:
        """
        returns a dict:
            initial_accuracy: accuracy prior to evaluating activations
            activations: importance of each rule
            final_accuracy: accuray after applying activations 
        """
        results = {}
        accuracy = self.eval_rule_base(self.fl_classifier.rule_base)
        results["initial_accuracy"] = accuracy
        initial_accuracy = accuracy
        if self.adaptative:
            self.adaptative_activations = self.eval_each_rule(accuracy["accuracy"])
            # FIXME: If there are two rules to be deleted, first delete (adjust) one and repeat eval and adjust
            # If there has been more than one rule to be deleted, maybe the deletion of all decreases accuracy?
            nzeros = sum(sum( 1 for x in L if x == 0) for L in self.adaptative_activations)
            if nzeros > 1:
                self.adaptative_activations = self.eval_each_0_rule(accuracy["accuracy"], self.adaptative_activations)
            results["activations"] = self.adaptative_activations
            self.original_activations = self.adjust_scores(self.adaptative_activations)
            accuracy = self.eval_rule_base(self.fl_classifier.rule_base)
        else:
            self.adaptative_activations = [[1 for r in rb] for rb in self.fl_classifier.rule_base]
            self.original_activations = self.adaptative_activations
        results["final_accuracy"] = accuracy
        return results

