import pandas as pd
import numpy as np
class Module(set):
    def __init__(self, genes=None):
        if genes is not None:
            set.__init__(self, genes)

    def filter_retaingenes(self, retaingenes):
        return(Module(set(self) & set(retaingenes)))

    def to_json(self):
        return "[" + ", ".join(sorted(list(self)))

class Modules(list):
    def __init__(self, modules=None):
        if modules is not None:
            # coerce arguments to Module
            list.__init__(self, [Module(module) for module in modules])

    def filter_retaingenes(self, retaingenes):
        return Modules([module.filter_retaingenes(retaingenes) for module in self])

    def filter_size(self, minsize):
        return Modules([module for module in self if len(module) >= minsize])

    def cal_membership(self, G=None):
        if G is None:
            G = sorted(list(set([g for module in self for g in module])))

        membership = []
        for module in self:
            membership.append([g in module for g in G])

        if len(self) > 0:
            return pd.DataFrame(np.array(membership, ndmin=2), columns=G, index=["M" + str(i) for i in range(len(self))]).T
        else:
            return pd.DataFrame(np.zeros((0, len(G))), columns=G).T