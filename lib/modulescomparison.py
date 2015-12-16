import numpy as np
import pandas as pd

# for speed, the module comparison functions are implemented in Cython
import pyximport; pyximport.install()
import ebcubed
import jaccard

import json
from modulecontainers import Modules

from util import JSONExtendedEncoder

import os

def harmonic_mean(X):
    X = np.array(X)
    if np.any(X <= 0):
        return 0
    else:
        return len(X)/(np.sum(1/np.array(X)))

class ModulesComparison():
    def __init__(self, modulesA, modulesB, G):
        self.modulesA = modulesA
        self.modulesB = modulesB
        self.G = G

        self.membershipsA = self.modulesA.cal_membership(self.G).astype(np.uint8)
        self.membershipsB = self.modulesB.cal_membership(self.G).astype(np.uint8)

        if len(self.modulesB) > 0 and len(self.modulesA) > 0:
            self.jaccards = np.nan_to_num(jaccard.cal_similaritymatrix_jaccard(self.membershipsA.T.as_matrix(), self.membershipsB.T.as_matrix()))
        else:
            self.jaccards = np.zeros((1,1))

    def score(self, baseline):
        if self.membershipsA.shape[1] == 0 or self.membershipsB.shape[1] == 0:
            recoveries = relevances = recalls = precisions = np.zeros(1)
        else:
            recoveries = self.jaccards.max(1)
            relevances = self.jaccards.max(0)

            recalls, precisions = ebcubed.cal_ebcubed(self.membershipsA.as_matrix(), self.membershipsB.as_matrix(), self.jaccards.T.astype(np.float64))

        scores = {
            "recoveries":recoveries,
            "recovery":recoveries.mean(),

            "relevances":relevances,
            "relevance":relevances.mean(),

            "recalls":recalls,
            "recall":recalls.mean(),

            "precisions":precisions,
            "precision":precisions.mean()
        }

        # normalize every score with the baseline and calculate the harmonic mean
        scores["F1norm_rprr"] = harmonic_mean([(scores[scorename]/baseline[scorename]) for scorename in ["recovery", "relevance", "recall", "precision"]])
        return scores

import multiprocessing as mp

class Modeval:
    def __init__(self, settings_name):
        self.settings_name = settings_name
        self.scoring_folder = "../results/modeval/"

    def run(self, settings, pool):
        baselines = pd.read_table("../results/modeval/baselines.tsv", index_col=[0, 1])

        jobs = []
        manager = mp.Manager()
        scores = manager.dict()

        params_pool = []

        i = 0
        for setting in settings:
            modules = Modules(json.load(open("../" + setting["output_folder"] + "modules.json")))
            runinfo = json.load(open("../" + setting["output_folder"] + "runinfo.json"))
            
            method = json.load(open("../" + setting["method_location"]))
            dataset = json.load(open("../" + setting["dataset_location"]))

            for knownmodules_name, knownmodules_location in dataset["knownmodules"].items():
                baseline = baselines.ix[dataset["datasetname"], knownmodules_name].to_dict()

                params_pool.append((i, modules, knownmodules_name, knownmodules_location, method, dataset, runinfo, baseline, scores))

                i+=1

        pool.starmap(modevalworker, params_pool)

        self.scores = pd.DataFrame(list(scores.values()))
        self.scores = self.scores[[column for column in self.scores if column not in ["recoveries", "relevances", "recalls", "precisions"]]]
        self.scores_full = list(scores.values())

    def save(self):
        self.scores.to_csv(self.scoring_folder + self.settings_name + ".tsv", sep="\t")
        json.dump(self.scores_full, open(self.scoring_folder + self.settings_name + ".json", "w"), cls=JSONExtendedEncoder)

    def load(self):
        self.scores = pd.read_table(self.scoring_folder + self.settings_name + ".tsv")

def modevalworker(settingid, modules, knownmodules_name, knownmodules_location, method, dataset, runinfo, baseline, scores):
    knownmodules = Modules(json.load(open("../" + knownmodules_location)))
    allgenes = {g for module in knownmodules for g in module}
    
    filteredmodules = modules.filter_retaingenes(allgenes).filter_size(5)
    
    comp = ModulesComparison(filteredmodules, knownmodules, allgenes)
    settingscores = comp.score(baseline)
    
    settingscores.update(method["params"])
    settingscores.update(dataset["params"])
    settingscores["knownmodules"] = knownmodules_name
    settingscores["runningtime"] = runinfo["runningtime"]
    
    scores[settingid] = settingscores