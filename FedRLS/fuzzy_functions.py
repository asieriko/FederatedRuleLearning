import numpy as np

def jaccard(set1, set2):
    # TODO: Other FS, T2...
    universe = [min(set1[0],set2[0]), max(set1[-1],set2[-1])]
    s1 = discretize(universe,set1)
    s2 = discretize(universe,set2)
    intersection = sum(np.minimum(s1,s2))
    union = sum(np.maximum(s1,s2))
    if union == 0.0:
        return 0
    return intersection/union

def discretize(u, fset, n_points=100, shape=[0,1,1,0]):  # shape because the domain limits ...
    return np.interp(np.linspace(u[0],u[1],n_points), fset, shape)


def antecedent_comparison(r1, r2):
    """
    Compares two list of fuzzy sets:
        - If the an antecedent does not appear in either rules, the comparison of these is 1
        - If one of the antecedents is missing, the jaccard is made between the other an the
        set formed by 1, for all of the universe
        - If the two antecedents are fuzzy sets, the jaccarad index is computed between them
    """
    acum = []
    for a1,a2 in zip(r1,r2):
        if len(a1) == len(a2) == 1:
            acum.append(1)
        elif len(a1) == 1:
            acum.append(jaccard([a2[0],a2[0],a2[-1],a2[-1]], a2))
        elif len(a2) == 1:
            acum.append(jaccard(a1, [a1[0],a1[0],a1[-1],a1[-1]]))
        else:
            acum.append(jaccard(a1, a2))

    return min(acum)

def comparison(r1, r2):
    """
    Compares two fuzzy rules as follows:
        - If the consequences are different, the comparison is 0
        - If the an antecedent does not appear in either rules, the comparison of these is 1
        - If one of the antecedents is missing, the jaccard is made between the other an the
        set formed by 1, for all of the universe
        - If the two antecedents are fuzzy sets, the jaccarad index is computed between them
        - Thre return value is the minimun of all the pairwise comparisons
    """
    if r1[-1] != r2[-1]:
        return 0
    comp = antecedent_comparison(r1[:-1][0],r2[:-1][0])
    # I return this because I use min in the aggregation of the antecedents and
    # as the consequences are equal their comparison is 1, so min(comp,1)=comp
    return comp
