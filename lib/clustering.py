import sklearn.cluster
from collections import defaultdict
from modulecontainers import Module, Modules

import pandas as pd
import numpy as np

from simdist import simdist

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.pandas2ri
rpy2.robjects.pandas2ri.activate()
from rpy2.robjects.packages import importr

from util import TemporaryDirectory

def standardize(X):
    return (X - X.mean())/(X.std())

## clustering functions
def flame(E, knn=10,threshold=-1,  threshold2=-3.0, steps=500):
    with TemporaryDirectory() as tmpdir:
        with open(tmpdir + "/E.csv", "w") as outfile:
            outfile.write(str(E.shape[1]) + " " + str(E.shape[0]) + "\n")
            standardize(E).T.to_csv(outfile, index=False, header=False, sep=" ")

        binary = os.environ["PERSOFTWARELOCATION"] + "/flame/sample"
        command = "{binary} {tmpdir}/E.csv {knn} {threshold2} {steps} {threshold}".format(**locals())

        process = sp.Popen(command, shell=True, stdout=sp.PIPE)
        out, err = process.communicate()

        modules = []
        for row in out.decode().split("\n"):
            if row.startswith("Cluster") and "outliers" not in row:
                gids = row[row.index(":")+1:].split(",")
                if gids[0] != "":
                    module = Module([E.columns[int(gid)] for gid in gids])
                    modules.append(module)

    return modules

def kmedoids(E, number=100, simdist_function="pearson_correlation"):
    importr("cluster")

    distances = simdist(E, simdist_function, similarity=False)
    rresults = ro.r["pam"](distances, diss=True, k=number)
    modules = convert_labels2modules(list(rresults.rx2("clustering")), E.columns)
    return modules

def som(E, dim=6, dims=None, topo="rectangular", rlen=100, alpha=[0.05, 0.01], radius=None):
    importr("kohonen")

    if dims is None:
        dims = [dim, dim]

    if radius is None:
        rresults = ro.r["som"](standardize(E).T, ro.r["somgrid"](dims[0], dims[1], "rectangular"), rlen=rlen, alpha=alpha)
    else:
        rresults = ro.r["som"](standardize(E).T, ro.r["somgrid"](dims[0], dims[1], "rectangular"), rlen=rlen, alpha=alpha, radius=radius)

    modules = convert_labels2modules(list(rresults.rx2("unit.classif")), E.columns)

    return modules

def kmeans(E, k=100, max_iter=300, n_init=10, seed=None):
    kmeans = sklearn.cluster.KMeans(n_clusters=int(k), max_iter=int(max_iter), n_init=int(n_init), random_state=seed)
    kmeans.fit(standardize(E).T)
    modules = convert_labels2modules(kmeans.labels_, E.columns)
    return modules

def cmeans(E, k=100, m="auto", cutoff=0.5, cluster_all=True):
    importr("Mfuzz")
    importr("Biobase")
    Exprs = ro.r["ExpressionSet"](ro.r["as.matrix"](standardize(E).T))
    if m == "auto":
        m = ro.r["mestimate"](Exprs)

    rresults = ro.r["mfuzz"](Exprs, k, m)

    membership = np.array(rresults.rx2("membership"))

    modules = []
    for membership_cluster in membership.T:
        genes = E.columns[membership_cluster >= cutoff]
        modules.append(Module(genes))

    return modules

def spectral_similarity(E, k=100, seed=None, simdist_function="pearson_correlation"):
    similarities = simdist(E, simdist_function)
    spectral = sklearn.cluster.SpectralClustering(n_clusters=int(k), affinity="precomputed", random_state = seed)
    spectral.fit(similarities+1)
        
    return convert_labels2modules(spectral.labels_, E.columns)

def affinity(E, preference_fraction=0.5, simdist_function="pearson_correlation", damping=0.5, max_iter=200):
    similarities = simdist(E, simdist_function)

    similarities_max, similarities_min = similarities.as_matrix().max(), similarities.as_matrix().min()
    preference = (similarities_max - similarities_min) * preference_fraction

    ro.packages.importr("apcluster")

    rresults = ro.r["apcluster"](s=ro.Matrix(similarities.as_matrix()), p=preference)
    labels = np.array(ro.r["labels"](rresults, "enum"))

    modules = convert_labels2modules(labels, E.columns)

    return modules

def spectral_knn(E, k=100, knn=50, seed=None):
    spectral = sklearn.cluster.SpectralClustering(n_clusters=int(k), n_neighbors = int(knn), affinity="nearest_neighbors", random_state = seed)
    spectral.fit(standardize(E).T)
        
    return convert_labels2modules(spectral.labels_, E.columns)

def wgcna(E, power=6, mergeCutHeight=0.15, minModuleSize=20, deepSplit=2, detectCutHeight=0.995, TOMDenom="min", reassignThreshold=1e-6):
    importr("WGCNA")

    ro.r("allowWGCNAThreads()")

    rblockwiseModules = ro.r["blockwiseModules"]
    rresults = rblockwiseModules(
        E, 
        power=power, 
        mergeCutHeight=mergeCutHeight, 
        minModuleSize=minModuleSize, 
        deepSplit=deepSplit, 
        detectCutHeight=detectCutHeight, 
        numericLabels=True, 
        TOMDenom=TOMDenom, 
        reassignThreshold=reassignThreshold,
        minCoreKME=minCoreKME,
        minCoreKMESize=minCoreKMESize,
        minKMEtoStay=minKMEtoStay
    )

    modules = convert_labels2modules(list(rresults.rx2("colors")), E.columns, ignore_label=0)

    return modules

def agglom(E, k=100, linkage="complete", simdist_function="pearson_correlation"):
    importr("cluster")
    ro.globalenv["distances"] =  simdist(E, simdist_function, similarity=False)
    ro.r("hclust_results = hclust(as.dist(distances), method='{linkage}')".format(**locals()))
    rresults = ro.r("labels = cutree(hclust_results, k={k})".format(**locals()))
    modules = convert_labels2modules(list(rresults), E.columns)
    return modules

def hybrid(E, k=100):
    importr("hybridHclust")

    ro.globalenv["E"] = standardize(E).T
    ro.r("hclust_results = hybridHclust(E)")
    rresults = ro.r("cutree(hclust_results, k={k})".format(**locals()))

    modules = convert_labels2modules(list(rresults), E.columns)

    return modules

def divisive(E, k=100):
    importr("cluster")

    ro.globalenv["E"] = E
    ro.r("diana_results = diana(as.dist(1-cor(E)),diss=TRUE)".format(**locals()))
    rresults = ro.r("cutree(diana_results, k={k})".format(**locals()))

    modules = convert_labels2modules(list(rresults), E.columns)

    return modules

def sota(E, maxCycles=1000, maxEpochs=1000, distance="euclidean", wcell=0.01, pcell=0.005, scell=0.001, delta=1e-04, neighb_level=0, alpha=0.95, unrest_growth=False):
    importr("clValid")

    distances = simdist(standardize(E), "euclidean", False)
    maxDiversity = np.percentile(distances.as_matrix().flatten(), maxDiversity_percentile)

    rresults = ro.r["sota"](standardize(E).T, maxCycles, maxEpochs, distance, wcell, pcell, scell, delta, neighb_level, maxDiversity, unrest_growth)

    modules = convert_labels2modules(list(rresults.rx2("clust")), E.columns)

    return modules

def dclust(E, rho=0.5, delta=0.5, simdist_function="pearson_correlation"):
    ro.packages.importr("densityClust")

    distances = simdist(E, simdist_function, False)
    rresults =  ro.r["densityClust"](ro.r["as.dist"](distances))
    rresults = ro.r["findClusters"](rresults, rho=rho, delta=delta)

    modules = convert_labels2modules(list(rresults.rx2("clusters")), E.columns)

    return modules

def click(E, homogeneity=0.5):
    tmpdir = tempfile.mkdtemp()

    try:
        with open(tmpdir + "/clickInput.orig", "w") as outfile:
            outfile.write("{nG} {nC}\n".format(nG = len(E.columns), nC=len(E.index)))
            
            E.T.to_csv(outfile, sep="\t", header=False)
            
        with open(tmpdir + "/clickParams.txt", "w") as outfile:
            outfile.write("""
DATA_TYPE
FP 
INPUT_FILES_PREFIX
{tmpdir}/clickInput 
OUTPUT_FILE_PREFIX
{tmpdir}/clickOutput 
SIMILARITY_TYPE
CORRELATION 
HOMOGENEITY
{homogeneity}
            """.format(tmpdir=tmpdir, homogeneity=homogeneity))

        click_location = os.environ["PERSOFTWARELOCATION"] + "/Expander/click.exe"

        command = "{click_location} {tmpdir}/clickParams.txt".format(**locals())

        sp.call(command, shell=True)

        labels = pd.read_csv(tmpdir + "/clickOutput.res.sol", sep="\t", index_col=0, header=None, squeeze=True)
    finally:
        shutil.rmtree(tmpdir)

    modules = convert_labels2modules(labels.tolist(), labels.index.tolist(), 0)
    return modules

def dbscan(E, eps=0.2, MinPts=5):
    importr("fpc")

    ro.globalenv["E"] = E
    rresults = ro.r("dbscan(as.dist(1-cor(E)), method='dist', eps={eps}, MinPts={MinPts})".format(**locals()))
    modules = convert_labels2modules(list(rresults.rx2("cluster")), E.columns, 0)

    return modules

def meanshift(E, bandwidth=None, cluster_all=True):
    if bandwidth is None or bandwidth == "auto":
        meanshift = sklearn.cluster.MeanShift(cluster_all=cluster_all)
    else:
        meanshift = sklearn.cluster.MeanShift(bandwidth=bandwidth, cluster_all=cluster_all)

    meanshift.fit(standardize(E).T)
    meanshift.labels_

    modules = convert_labels2modules(meanshift.labels_, E.columns)

    return modules

def clues(E, disMethod="1-corr", n0=300, alpha=0.05, eps=1e-4, itmax=20, strengthMethod="sil", strengthIni=-1, **kwargs):
    ro.packages.importr("clues")

    rresults = ro.r["clues"](
        ro.Matrix(standardize(E).T), 
        disMethod=disMethod, 
        n0=n0,
        alpha=alpha,
        eps=eps,
        itmax=itmax,
        strengthMethod=strengthMethod,
        strengthIni=strengthIni,
        quiet=False
    )

    modules = convert_labels2modules(list(rresults.rx2("mem")), E.columns)

    return modules

## Decomposition
def ica_fdr(E, k=200, qvalcutoff=1e-3, seed=None):
    source = _ica_fastica(E, k, seed)
    modules = _ica_fdr(E, source, qvalcutoff)

    return modules

def ica_zscores(E, k=200, stdcutoff=3):
    source = _ica_fastica(E, k)
    modules = _ica_zscores(E, source, stdcutoff)

    return modules

def ica_percentage(E, k=200, perccutoff=0.075):
    source = _ica_fastica(E, k)
    modules = _ica_perccutoff(E, source, perccutoff)

    return modules

def ipca(E, k=200, qvalcutoff=1e-3, **kwargs):
    source = _ipca(E, k)
    modules = _ica_fdrtool(E, source, qvalcutoff)

    return modules

def pca(E, k=200, stdcutoff=3, **kwargs):
    source = _pca_weights(E, k)
    modules = _ica_zscores(E, source, stdcutoff)

    return modules

def _ipca(E, k):
    ro.packages.importr("mixOmics")

    ipca = ro.r["ipca"]
    rresults = ipca(E, ncomp=k)

    source = np.array(rresults.rx2("loadings"))

    return source

def _ica_fastica(E, k, seed=None):

    ica = sklearn.decomposition.FastICA(n_components=k, random_state=seed)
    source = ica.fit_transform(standardize(E).T)

    return source

def _ica_zscores(E, source, stdcutoff):
    modules = []
    for source_row in source.T:
        genes = E.columns[source_row < -source_row.std() * stdcutoff].tolist() + E.columns[source_row > +source_row.std() * stdcutoff].tolist()

        modules.append(Module(genes))
    return modules

def _ica_fdrtool(E, source, qvalcutoff):
    importr("fdrtool")
    rfdrtool = ro.r["fdrtool"]

    modules = []

    print("qvalcutoff: " + str(qvalcutoff))

    for source_row in source.T:
        rresults = rfdrtool(ro.FloatVector(source_row), plot=False, cutoff_method="fndr", verbose=False)
        qvals = np.array(rresults.rx2("qval"))

        genes = E.columns[qvals < qvalcutoff]

        modules.append(Module(genes))
    return modules

def _ica_perccutoff(E, source, perccutoff):
    modules = []

    for source_row in source.T:
        sortedgenes = E.columns[source_row.argsort()]
        genes = sortedgenes[:int(round(len(E.columns) * perccutoff))]
        modules.append(Module(genes))
        
        genes = sortedgenes[-int(round(len(E.columns) * perccutoff)):]
        modules.append(Module(genes))
    return modules

def _pca(E, k):
    pca = sklearn.decomposition.PCA(n_components=int(k))

    source = pca.fit_transform(standardize(E))

    return pca.components_.T

## utility functions
def convert_labels2modules(labels, G, ignore_label=None):
    modules = defaultdict(Module)
    for label, gene in zip(labels, G):
        if label != ignore_label:
            modules[label].add(gene)
    return list(modules.values())