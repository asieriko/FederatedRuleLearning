<<<<<<< HEAD
from functools import partial
from multiprocessing import Process, Pool
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

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

class LocalAgent:

    def __init__(self, dataset):
        self.dataset = dataset
        # Import  data
        self.X = dataset[0]
        self.y = dataset[1]

        # Split the data into a training set and a test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                                                random_state=0)

        self.fl_classifier = None
        # Or take them directly from fl_classifier
        self.performance = None

    def set_model(self, nRules, nAnts, fz_type_studied, tolerance,runner, n_gen, n_pop):
        self.n_gen = n_gen
        self.n_pop = n_pop

        class_names = np.unique(self.dataset[1])
        precomputed_partitions = utils.construct_partitions(self.X, fz_type_studied)
        model = GA.BaseFuzzyRulesClassifier(
            nRules=nRules,
            nAnts=nAnts,
            fuzzy_type=fz_type_studied, 
            verbose=False,
            tolerance=tolerance, 
            linguistic_variables=precomputed_partitions,
            class_names=class_names,
            runner=runner)
        self.fl_classifier = model

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.X = dataset[0].to_numpy()
        self.y = dataset[1]

    def fit(self, mrule_base=None):
        if mrule_base:
            self.fl_classifier.load_master_rule_base(mrule_base)

        # fl_classifier.customized_loss(utils.mcc_loss)
        self.fl_classifier.fit(self.X_train, self.y_train, n_gen=self.n_gen, pop_size=self.n_pop, random_state = 23)
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
        activations = []
        for rule_bases in self.fl_classifier.rule_base:
            activations_i = []
            for rule in rule_bases:
                prev_score = rule.score
                rule.score = 0
                performance = self.eval_rule_base(self.fl_classifier.rule_base)
                # print(performance)#,precision_recall_fscore_support(predicted, self.y_test,average="weighted"))
                rule.score = prev_score
                if performance == base_performance:
                    activations_i.append(1)
                elif performance < base_performance:
                    activations_i.append(2)
                else:
                    activations_i.append(0)
            activations.append(activations_i)
        # [len(rb.rules) for rb in self.fl_classifier.rule_base.rule_bases]
        # to copy shape to activations
        return activations

    def adjust_scores(self, activations):
        for rule_bases, acts in zip(self.fl_classifier.rule_base, activations):
            for rule, acti in zip(rule_bases, acts):
                if acti == 0:
                    rule.score = 0




    def eval_rule_base(self, rb):
        evb = eval_rules.evalRuleBase(rb, self.X.to_numpy(), self.y)
        evb.add_classification_metrics()
        predicted = self.fl_classifier.predict(self.X_test)
        performance = np.mean(np.equal(predicted, self.y_test))
        return performance

    def eval_test(self):
        performance = self.eval_rule_base(self.fl_classifier.rule_base)
        print("Initial performance: ", performance)#,precision_recall_fscore_support(predicted, self.y_test,average="weighted"))
        activations = self.eval_each_rule(performance)
        print(activations)
        print("Adjust scores")
        self.adjust_scores(activations)
        print("new eval")
        performance = self.eval_rule_base(self.fl_classifier.rule_base)
        print("Final performance: ", performance)

=======
from functools import partial
from multiprocessing import Process, Pool
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

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

class LocalAgent:

    def __init__(self, dataset, name="Client", ramdom_seed = 23):
        self.name = name
        self.dataset = dataset
        # Import  data
        self.X = dataset[0]
        self.y = dataset[1]
        self.ramdom_seed = ramdom_seed

        # Split the data into a training set and a test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                                                random_state=0)

        self.fl_classifier = None
        # Or take them directly from fl_classifier
        self.performance = None

    def set_model(self, nRules, nAnts, fz_type_studied, tolerance,runner, n_gen, n_pop, precomputed_partitions = None):
        self.n_gen = n_gen
        self.n_pop = n_pop



        class_names = np.unique(self.dataset[1])
        if precomputed_partitions is None:
            precomputed_partitions = utils.construct_partitions(self.X, fz_type_studied)
        model = GA.BaseFuzzyRulesClassifier(
            nRules=nRules,
            nAnts=nAnts,
            fuzzy_type=fz_type_studied, 
            verbose=False,
            tolerance=tolerance, 
            linguistic_variables=precomputed_partitions,
            class_names=class_names,
            runner=runner)
        self.fl_classifier = model

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.X = dataset[0].to_numpy()
        self.y = dataset[1]

    def get_max_min(self):
        return np.stack([self.X.min().to_numpy(),self.X.max().to_numpy()])

    def fit(self, candidate_rules=None):
        if candidate_rules:
            # self.fl_classifier.load_master_rule_base(candidate_rules)
            self.fl_classifier.fit(self.X_train, self.y_train, candidate_rules=candidate_rules)
        else:
            # fl_classifier.customized_loss(utils.mcc_loss)
            self.fl_classifier.fit(self.X_train, self.y_train, n_gen=self.n_gen, pop_size=self.n_pop, random_state = self.ramdom_seed)
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
        activations = []
        for rule_bases in self.fl_classifier.rule_base:
            activations_i = []
            for rule in rule_bases:
                prev_score = rule.score
                rule.score = 0
                performance = self.eval_rule_base(self.fl_classifier.rule_base)
                # print(performance)#,precision_recall_fscore_support(predicted, self.y_test,average="weighted"))
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

    def adjust_scores(self, activations):
        # FIXME: If two activations are 0 it doesn't mean that both have to be removed, only "the worse"
        # Or study if both have to be removed, triying first one and then the other.
        for rule_bases, acts in zip(self.fl_classifier.rule_base, activations):
            for rule, acti in zip(rule_bases, acts):
                if acti == 0:
                    rule.score = 0




    def eval_rule_base(self, rb):

        str_rules = eval_tools.eval_fuzzy_model(self.fl_classifier, self.X_train, self.y_train, self.X_test, self.y_test,
                                                     plot_rules=False, print_rules=False, plot_partitions=False,
                                                     print_accuracy=False, print_matthew=False, return_rules=False)

        evb = eval_rules.evalRuleBase(rb, self.X.to_numpy(), self.y)
        evb.add_classification_metrics()
        predicted = self.fl_classifier.predict(self.X_test)
        performance = np.mean(np.equal(predicted, self.y_test))
        return performance

    def eval_test(self):
        print(f"\n -- {self.name} --")
        performance = self.eval_rule_base(self.fl_classifier.rule_base)
        pi = performance
        print("Initial performance: ", performance)#,precision_recall_fscore_support(predicted, self.y_test,average="weighted"))
        activations = self.eval_each_rule(performance)
        print(activations)
        print("Adjust scores")
        self.adjust_scores(activations)
        # FIXME: If there are two rules to be deleted, first delete (adjust) one and repeat eval and adjust
        print("new eval")
        performance = self.eval_rule_base(self.fl_classifier.rule_base)
        print("Final performance: ", performance)
        print(f"LOG:,{self.name},{pi},{performance}")
        print("-- ------ --")
        return performance

>>>>>>> 04c9cba (add a parameter dictionary with experiment configuration)
