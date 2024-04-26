import os,sys
sys.path.append(os.path.abspath(os.path.join("..","ex-fuzzy","ex_fuzzy")))

from ex_fuzzy import *

def parse_rule_base(rule_base):
    consequent_names = rule_base.get_consequents_names()
    antecedents = rule_base.get_antecedents()
    fuzzy_type = antecedents[0].fuzzy_type()
    domains = [a.domain() for a in antecedents]
    variable_names = [a.name for a in antecedents]
    term_names = [a.linguistic_variable_names() for a in antecedents]  # Names for each linguistic variable, for each antecedent
    linguistic_variables = [a.get_linguistic_variables() for a in antecedents]  # n_antecedents * 3 (n_var)
    linguistic_variables_params = [[a.membership_parameters for a in lv] for lv in linguistic_variables] # n_antecedents * 3 (n_var) * n_param (4 for t1 trap)
    rule_base_matrix = rule_base.get_rulebase_matrix()
    rule_base_matrix_params = []
    for icon, r_base in enumerate(rule_base_matrix):
        for rule in r_base:
            rule_params = []
            for ia, ant in enumerate(rule):
                if ant != -1:
                    rule_params.append(linguistic_variables_params[ia][int(ant)])
                else:
                    rule_params.append([-1])
            rule_params.append(consequent_names[icon])
            rule_base_matrix_params.append(rule_params)
    return rule_base_matrix_params, domains, variable_names


def create_fuzzy_variable(name:str, domain:list[int], params:list[list], fsnames=["Low","Medium","High"]):
    # params [[0,1,2,3],[2,3,4,5],[5,6,7,8]]
    fsets = []
    for p, fsn in zip(params,fsnames):
        fsets.append(fuzzy_sets.FS(fsn,p,domain))
    return fuzzy_sets.fuzzyVariable(name,fsets)

def modify_domain_fuzzy_variable(fv, new_domain:list[int]):
    for fs in fv:
        fs.domain = new_domain


if __name__=="__main__":
    L = fuzzy_sets.FS("Low",[0,1,2,3],[0,10])
    M = fuzzy_sets.FS("Medium",[2,3,4,5],[0,10])
    H = fuzzy_sets.FS("High",[5,6,7,8],[0,10])
    T = fuzzy_sets.fuzzyVariable("Temp",[L,M,H])

    R = rules.RuleSimple([0,1],0)
    RULEMATRIX = rules.list_rules_to_matrix([R,R])
    rules.construct_rule_base(RULEMATRIX,[0,0],antecedents=[T,T],class_names=["0","0"])
    RB = rules.RuleBase(antecedents=[T,T],rules=[R], consequent=0)
