import os,sys

import ex_fuzzy.fuzzy_sets
# sys.path.append(os.path.abspath(os.path.join("..","ex-fuzzy","ex_fuzzy")))
import numpy as np
from ex_fuzzy import *

def argsortterms(terms):
    return [terms.index("Low"),terms.index("Medium"),terms.index("High")] 

def parse_rule_base(rule_base: ex_fuzzy.rules.MasterRuleBase):
    consequent_names = rule_base.get_consequents_names()
    antecedents = rule_base.get_antecedents()
    fuzzy_type = antecedents[0].fuzzy_type()
    domains = [a.domain() for a in antecedents]
    variable_names = [a.name for a in antecedents]
    term_names = [a.linguistic_variable_names() for a in antecedents]  # Names for each linguistic variable, for each antecedent
    terms_sort = [argsortterms(t) for t in term_names]   # the terms can be disordered
    linguistic_variables = [a.get_linguistic_variables() for a in antecedents]  # n_antecedents * 3 (n_var)
    linguistic_variables_params = [[a.membership_parameters for a in lv] for lv in linguistic_variables] # n_antecedents * 3 (n_var) * n_param (4 for t1 trap)
    rule_base_matrix = rule_base.get_rulebase_matrix()
    rule_base_matrix = [rule for rule in rule_base.get_rulebase_matrix() if rule.size != 0]  # There are problems if there are no rules for a consequent
    rule_base_matrix_params = []
    # NOTE: to discard consequents without rules, else the consequents are wrongly assigned
    available_consequents = np.array(consequent_names)[np.argwhere([rule.size for rule in rule_base.get_rulebase_matrix()])]
    # FIXME: Scores not sets after retrain
    # rule_base_scores = rule_base.get_scores()
    # If there is only one rule_base (consequent) in the master_rule_base
    # rules.py 975 can't concatenate the results -> need at least one array to concatenate
    rule_base_scores = []
    for r_base in rule_base:
       rule_base_scores = np.append(rule_base_scores, r_base.get_scores())
    r_idx = 0

    for i_con, r_base in enumerate(rule_base_matrix):
        for rule in r_base:
            rule_params = []
            rule_sorted = [] 
            rule_params_dict = {} 
            for ia, ant in enumerate(rule):
                if ant != -1:
                    rule_params.append(linguistic_variables_params[ia][int(ant)])
                    rule_sorted.append(terms_sort[ia].index(rule[ia]))
                else:
                    rule_params.append([-1])
                    rule_sorted.append(-1)
            rule_params_dict["var_shape"] = rule_params
            rule_params_dict["var_idx"] = rule_sorted
            rule_params_dict["score"] = rule_base_scores[r_idx] 
            rule_params_dict["class"] = available_consequents[i_con][0]  #using argwhere it is converted to numpy array...
            r_idx += 1
            # rule_params.append(consequent_names[i_con])
            # rule_params.append(rule.tolist())
            rule_base_matrix_params.append(rule_params_dict)

    # for i_con, r_base in enumerate(rule_base_matrix):
    #     for rule in r_base:
    #         rule_params = []
    #         rule_sorted = [] 
    #         rule_params_dict = {} 
    #         for ia, ant in enumerate(rule):
    #             if ant != -1:
    #                 rule_params.append(linguistic_variables_params[ia][int(ant)])
    #                 rule_sorted.append(terms_sort[ia].index(rule[ia]))
    #             else:
    #                 rule_params.append([-1])
    #                 rule_sorted.append(-1)
    #         rule_params_dict["var_shape"] = rule_params
    #         rule_params_dict["var_idx"] = rule_sorted
    #         rule_params_dict["score"] = rule_base_scores[r_idx] 
    #         rule_params_dict["class"] = consequent_names[i_con]
    #         r_idx += 1
    #         # rule_params.append(consequent_names[i_con])
    #         # rule_params.append(rule.tolist())
    #         rule_base_matrix_params.append(rule_params_dict)
    return rule_base_matrix_params, domains, variable_names


def create_fuzzy_variable(name:str, domain:list[int], params:list[list], fsnames=["Low","Medium","High"]) -> ex_fuzzy.fuzzy_sets.fuzzyVariable:
    # params [[0,1,2,3],[2,3,4,5],[5,6,7,8]]
    fsets = []
    for p, fsn in zip(params,fsnames):
        fsets.append(fuzzy_sets.FS(fsn, p, domain))
    return fuzzy_sets.fuzzyVariable(name, fsets)

def modify_domain_fuzzy_variable(fv, new_domain:list[int]) -> None:
    for fs in fv:
        fs.domain = new_domain

def create_rule_base(partitions, rules_by_keys, consequent_names) -> ex_fuzzy.rules.MasterRuleBase:
    # NOTE: I add consequent names becasue it can happen that in some cases
    # There is a consequent missing and so the new rule base is created wrongly
    rule_bases = [] 
    for key in consequent_names:
        rule_lst = []
        for rule_i in rules_by_keys[key]:
            # FIXME where to store the original rules with indices
            R = ex_fuzzy.rules.RuleSimple(rule_i["var_idx"] ,key)
            # R.score = rule_i["score"]  # IF we add score. then it does not get update on the client
            rule_lst.append(R)
        RB = ex_fuzzy.rules.RuleBaseT1(antecedents=partitions, rules=rule_lst)
        rule_bases.append(RB)
    master_rule_base = ex_fuzzy.rules.MasterRuleBase(rule_bases)
    return master_rule_base

def fuzzy_variable_to_str(fuzzy_variable: ex_fuzzy.fuzzy_sets.fuzzyVariable) -> list[str]:
    return [str(lv) for lv in fuzzy_variable.linguistic_variables]

def fuzzy_partition_to_str(fuzzy_partition: list[ex_fuzzy.fuzzy_sets.fuzzyVariable]) -> list[list[str]]:
    return [fuzzy_variable_to_str(fp) for fp in fuzzy_partition]