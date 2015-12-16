import pandas as pd
import numpy as np

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.pandas2ri
rpy2.robjects.pandas2ri.activate()
from rpy2.robjects.packages import importr

###### Distance/similarity functions #######

def cal_triangular(E, func):
    """
    Calculates a given function for every gene pair in E
    """
    Emat = E.T.as_matrix()
    correlations = np.identity(Emat.shape[0])
    todo = Emat.shape[0] * Emat.shape[0] / 2 - Emat.shape[0]
    for i, x in enumerate(Emat):
        row = []
        start = datetime.now()
        sys.stdout.write(str(i) + "/"+str(Emat.shape[0]))
        for y in Emat[i+1:]:
            sys.stdout.write(">")
            row.append(func(x, y))

        correlations[i, 1+i:] = row
        correlations[1+i:, i] = row

        todo -= len(row)
        if len(row) > 20:
            sys.stdout.write(
                "\nETA: " + str(todo * (datetime.now() - start)/len(row))+ 
                "\nPer: " + str((datetime.now() - start)/len(row)))

        sys.stdout.write("\n")
    return correlations

def simdist(E, simdist_function, similarity=True, **kwargs):
    """

    """

    choices = {
        "pearson_correlation":[True, lambda E: np.corrcoef(E.T)],
        "pearson_correlation_absolute":[True, lambda E: np.abs(np.corrcoef(E.T))],
        "spearman_rank_correlation":[True, lambda E: scipy.stats.spearmanr(E)[0]],
        "spearman_rank_correlation_absolute":[True, lambda E: np.abs(scipy.stats.spearmanr(E)[0])],
        "kendall_tau_correlation":[True, lambda E: cal_triangular(E, lambda x,y: scipy.stats.kendalltau(x, y)[0])],
        "kendall_tau_correlation_absolute":[True, lambda E: cal_triangular(E, lambda x,y: np.abs(scipy.stats.kendalltau(x, y)[0]))],
        "percentage_bend_correlation":[True, percentage_bend_correlation],
        "biweight_midcorrelation":[True, biweight_midcorrelation],
        "distance_correlation":[True, distance_correlation],

        "euclidean":[False, lambda E: sklearn.metrics.euclidean_distances(sklearn.preprocessing.scale(E).T)],
        "manhattan":[False, lambda E: scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(sklearn.preprocessing.scale(E).T, "cityblock"))],

        "maximal_information_coefficient": [True, maximal_information_coefficient],
        "aracne":[True, aracne],
        "clr":[True, clr],
        "knnmi":[True, knnmi],
        "pymi":[True, pymi],

        "hoeffdings_d":[True, hoeffdings_d],

        "topological_overlap_measure":[True, topological_overlap_measure],
        "topological_overlap_measure_absolute":[True, topological_overlap_measure_absolute],

        "minet_empirical":[True, minet_empirical],
        "minet_shrink":[True, minet_shrink],
        "minet_mm":[True, minet_mm]
    }

    measure_similarity, func = choices[simdist_function]

    if simdist_function == "given":
        if "similarities" in kwargs:
            simdist_matrix = kwargs["similarities"]
            measure_similarity = True
        elif "distances" in kwargs:
            simdist_matrix = kwargs["distances"]
            measure_similarity = False
        else:
            raise ValueError("simdist_function is given, but neither similarities nor distances in arguments")
    elif simdist_function + "_location" in kwargs:
        print("Loading similarity matrix")
        simdist_matrix = pd.read_csv(kwargs[simdist_function + "_location"], sep="\t", index_col=0)

        assert simdist_matrix.shape[0] == len(E.columns)
    else:
        simdist_matrix = func(E)
        simdist_matrix = pd.DataFrame(simdist_matrix, columns=E.columns, index=E.columns)


    if (measure_similarity and similarity) or (not measure_similarity and not similarity):
        return simdist_matrix
    else:
        return (-simdist_matrix) + simdist_matrix.max().max()




def knnmi(E):
    print("Warning: will use all available cores! See code")
    ro.packages.importr("parmigene")
    # v doesn't work, but why?
    ro.r("Sys.setenv(OMP_NUM_THREADS=\"1\")") # this algorithm uses all available cores by default, but cannot see unavailable cores (eg. on the cluster), the only way to prevent this is to set this environment variable
    similarities = np.array(ro.r["knnmi.all"](ro.Matrix(E.as_matrix().T)))

    return similarities

def topological_overlap_measure(E):
    ro.packages.importr("WGCNA")
    rtom = ro.r["TOMsimilarityFromExpr"]

    similarities = np.array(rtom(E, networkType="signed"))

    return similarities

def topological_overlap_measure_absolute(E):
    ro.packages.importr("WGCNA")
    rtom = ro.r["TOMsimilarityFromExpr"]

    similarities = np.array(rtom(E, networkType="unsigned"))

    return similarities

def hoeffdings_d(E):
    ro.packages.importr("Hmisc")
    similarities = np.array(ro.r["hoeffd"](ro.Matrix(E.as_matrix())).rx2("D"))

    return similarities
def pymi(E):
    sys.path.append(os.environ["PERSOFTWARELOCATION"] + "/pymi/")

    from mutual_info import mutual_information_2d

    correlations = cal_triangular(E, mutual_information_2d)

    return correlations

def aracne(E):
    with TemporaryDirectory() as tmpdir:
        expression_location = tmpdir + "/E.csv"
        aracne_folder = os.environ["PERSOFTWARELOCATION"] + "/ARACNE/ARACNE"
        aracne_location = aracne_folder + "/aracne2"
        output_location = tmpdir + "/output.csv"

        E.T.to_csv(expression_location, sep="\t")
        command = "{aracne_location} -i {expression_location} -o {output_location} -H {aracne_folder}".format(**locals())
        sp.call(command, shell=True)

        with open(output_location) as infile:
            similarities = pd.DataFrame(np.identity(len(E.columns)), index=E.columns, columns=E.columns)
            for line in infile:
                if line.startswith(">"):
                    continue;
                else:
                    line = line.strip().split("\t")
                    g1 = line.pop(0)
                    
                    for g2, mi in zip(line[0::2], line[1::2]):
                        similarities.ix[g1, g2] = float(mi)
    return similarities

def clr(E):
    with TemporaryDirectory() as tmpdir:
        E.to_csv(tmpdir + "/E.csv", sep="\t")

        clr_folder = os.environ["PERSOFTWARELOCATION"] + "/CLRv1.2.2/Code/"

        matlab_command = """
        addpath '""" + clr_folder + """';
        data = transpose(dlmread('""" + tmpdir + """/E.csv','\\t', 1, 1));

        A = clr(data);

        dlmwrite('""" + tmpdir + """/fullmatrix.csv', A, '\\t');
        exit;
        """

        matlab_command = matlab_command.replace("\n", "")

        command = "matlab -nodesktop -nosplash -nojvm -nodisplay -r \"" + matlab_command + "\""
        print(command)
        sp.call(command, shell=True)

        # postprocess context-likelihood ratio matrix, remove non-regulators

        similarities = pd.read_csv(tmpdir + "/fullmatrix.csv", sep="\t", header=None, index_col=None).as_matrix()
    return similarities

def minet(E, estimator="mi.empirical", disc="globalequalwidth"):
    importr("minet")
    return pd.DataFrame(np.array(ro.r["build.mim"](scale(E), estimator=estimator, disc=disc)), index=E.columns, columns=E.columns)
    #return pd.DataFrame(np.array(ro.r["minet"](scale(E), estimator=estimator, disc=disc)), index=E.columns, columns=E.columns)

def minet_empirical(E, **kwargs):
    return minet(E, "mi.empirical", "globalequalwidth")

def minet_shrink(E, **kwargs):
    return minet(E, "mi.shrink", "globalequalwidth")

def minet_mm(E, **kwargs):
    return minet(E, "mi.mm", "globalequalwidth")

def biweight_midcorrelation(E):
    ro.packages.importr("WGCNA")
    correlations = np.array(ro.r["bicor"](E))

    return correlations

def percentage_bend_correlation(E):
    ro.packages.importr("asbio")
    rrbp = ro.r["r.pb"]

    correlations = cal_triangular(E, lambda x,y: rrbp(x,y).rx2("r.bp")[0])

    return correlations

def maximal_information_coefficient(E, pool=None):
    # The R version of MIC works veryyyy slow, it takes weeks to calculate the similarity matrix on a 2000genes x 2000conditions expression matrix on 24 CPU cores
    # The python version, which under the hood uses C, goes somewhat faster (~60%), although the aforementioned calculation still takes around 2 days.
    # These two versions give exactly the same results

    # ro.packages.importr("minerva")

    # rresults = ro.r["mine"](E, n_cores = ncores)
    # similarities = np.array(rresults.rx2("MIC"))

    Emat = E.as_matrix().T

    if pool is None:
        mine = minepy.MINE()

        similarities = np.identity(Emat.shape[0])
        for i,j in combinations(np.arange(Emat.shape[0]), 2):
            mine.compute_score(Emat[i], Emat[j])
            similarities[i,j] = similarities[j,i] = mine.mic()
    else:
        Emat = E.as_matrix().T
        args = [[x, Emat, i+1] for i, x in enumerate(Emat)]
        mics = pool.map(_cal_mic_row_star, args)

        similarities = np.identity(E.shape[1])
        for i in range(len(mics)):
            similarities[i, 1+i:] = mics[i]
            similarities[1+i:, i] = mics[i]

    return similarities

def _cal_mic_row_star(args):
    return _cal_mic_row(*args)
def _cal_mic_row(x, Y, i):
    mine = minepy.MINE()

    start = datetime.now()
    similarities_row = []
    for y in Y[i:]:
        mine.compute_score(x, y)
        similarities_row.append(mine.mic())

    if len(Y[i:]) > 0:
        print(len(Y[i:]))
        print((datetime.now() - start).total_seconds() / len(Y[i:]))

    return similarities_row

def distance_correlation(E, pool=None):
    if pool == None:
        As = [_cal_A(row) for row in E.as_matrix().T]
        dvars = np.array([_cal_dvar(A) for A in As])

        dcovs = np.diag(dvars)
        for i,j in combinations(np.arange(len(As)), 2):
            dcovs[i,j] = dcovs[j,i] = _cal_dcov(As[i], As[j])

        similarities = dcovs / np.sqrt(dvars[None,:] * dvars[:,None])
    else:
        print(pool)
        print("start")
        Emat = E.as_matrix().T
        args = [[x, Emat, i+1] for i, x in enumerate(Emat)]
        dcors = pool.map(_cal_dcor_row_star, args)

        print("done")

        similarities = np.identity(E.shape[1])
        for i in range(len(dcors)):
            similarities[i, 1+i:] = dcors[i]
            similarities[1+i:, i] = dcors[i]

    return similarities

def _cal_dcor(x, y):
    A = _cal_A(x)
    Avar = _cal_dvar(A)
    B = _cal_A(y)
    Bvar = _cal_dvar(B)

    return _cal_dcov(A, B)/(np.sqrt(Avar * Bvar))

def _cal_dcor_row_star(args):
    return _cal_dcor_row(*args)
def _cal_dcor_row(x, Y, i):
    A = _cal_A(x)
    Avar = _cal_dvar(A)

    start = datetime.now()
    dcors = []
    for y in Y[i:]:
        B = _cal_A(y)
        Bvar = _cal_dvar(B)

        dcors.append(_cal_dcov(A, B)/(np.sqrt(Avar * Bvar)))

    if len(Y[i:]) > 0:
        print(len(Y[i:]))
        print((datetime.now() - start).total_seconds() / len(Y[i:]))

    return dcors

def _cal_A(x):
    d = np.abs(x[:, None] - x)
    return d - d.mean(0) - d.mean(1)[:,None] + d.mean()
def _cal_dvar(A):
    return np.sqrt((A**2).sum() / len(A)**2)
def _cal_dcov(A,B):
    return np.sqrt((A * B).sum() / len(A)**2)