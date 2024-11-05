import numpy as np
import sys, os
sys.path.append(os.path.abspath("."))
from pathlib import Path
import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
from FedRLS.metrics import metrics
from datasets.load_dataset import read_keel_dat
from sklearn.model_selection import train_test_split


def non_federated_rules(dataset,data_seed=1,n_gen=30,n_pop=50,nRules=15,nAnts=4,fz_type_studied=fs.FUZZY_SETS.t1,tolerance=0.001,runner=1,random_seed=23,**args):
    # ,**args: to avoid "unexpected keyword for parameters not needed in this function but which are compiled into model_parameters"
        n_gen = n_gen
        n_pop = n_pop

        X = dataset[0]
        y = dataset[2]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=data_seed)

        precomputed_partitions = utils.construct_partitions(X_train, fz_type_studied)
        model = GA.BaseFuzzyRulesClassifier(
            nRules=nRules,
            nAnts=nAnts,
            fuzzy_type=fz_type_studied, 
            verbose=False,
            tolerance=tolerance, 
            linguistic_variables=precomputed_partitions,
            runner=runner)
        fl_classifier = model
        
        fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=n_pop, random_state = random_seed)

        predicted = fl_classifier.predict(X_test, out_class_names=True)
        test_performance = metrics(predicted, y_test)
        
        test_performance["train_performance"] = fl_classifier.performance
        # print(fl_classifier.rule_base)
        print(f"LOG:,Exfuzzy,{test_performance}")

        return fl_classifier.rule_base, test_performance

def main(datasetName, model_parameters, exp_name="Experiment", n_rep=5):
    path = Path(f"results/{datasetName}")
    Path.mkdir(path, parents=True, exist_ok=True)
    dataset = read_keel_dat(datasetName)

    for i in range(n_rep):  # FIXME: Change random_seed each rep
        rules_cent, acc_cent = non_federated_rules(dataset, data_seed=i, **model_parameters)
        with open(Path("results") / f"{datasetName}_summary_all_exfuzzy.csv","a") as results_file:
            results_file.write(f"{datasetName},{model_parameters['n_pop']},{model_parameters['n_gen']},{model_parameters['nRules']},{model_parameters['nAnts']},{i},{acc_cent['accuracy']:.4f},{acc_cent['f1']:.4f},{acc_cent['jaccard']:.4f},{acc_cent['recall']:.4f},{acc_cent['train_performance'][0]:.4f}\n")

if __name__ == "__main__":
    print(sys.argv)
    n_gens = [30, 50, 100, 200]
    n_pops = [50, 100, 200]
    n_Ruless = [10, 15, 20, 30]
    n_Antss = [4, 7, 10, 15]
    n_rep = 5  # FOLDS

    if len(sys.argv) > 1:
        datasetName = sys.argv[1]
    else:
        raise Exception("No dataset provided")

    for n_gen in n_gens:
        for n_pop in n_pops:
            for n_Rules in n_Ruless:
                for n_Ants in n_Antss:
                    model_parameters = {
                        "n_gen":n_gen,
                        "n_pop":n_pop,
                        "nRules":n_Rules,
                        "nAnts":n_Ants,
                        "fz_type_studied":fs.FUZZY_SETS.t1,
                        "tolerance":0.001,
                        "runner":1,
                        "random_seed":23,
                    }
                    main(datasetName, model_parameters, exp_name=f"test_exfuzzy-{datasetName}",n_rep=n_rep)
                