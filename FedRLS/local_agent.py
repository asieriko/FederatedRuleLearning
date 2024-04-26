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

