import numpy as np
import cooler
import click
from tqdm import tqdm
import pandas as pd
import numba
from numba import jit
import sys
from sklearn.neighbors import KDTree
import functools

eps = np.finfo(float).eps

cached={}
counts={'hitcache':0,'misscache':0}

def cacheDelta(func):
    @functools.wraps(func)
    def lazzyDelta(**kwargs):
        tad = (kwargs['left'],kwargs['right'])
        if kwargs['mask'] is not None:
            mask = tuple(dict.fromkeys(kwargs['mask']))
        else:
            mask=()
        if tad in cached and mask in cached[tad]:
            counts['hitcache']+=1
            return cached[tad][mask]
        val=func(**kwargs)
        if tad not in cached:
            counts['misscache']+=1
            cached[tad]={}
        cached[tad][mask]=val
        return val


    return lazzyDelta

@jit(nopython=True, parallel=True,error_model='python')
def DeltaNB(diags, left, right,w, minRatio=1.1):

    scores = np.zeros(w)
    N = 0
    for diag in numba.prange(1, w):
        crossIF1 = diags[max([0,left - diag]):left, diag]
        crossIF2 = diags[right -diag:right, diag]
        crossIF = np.concatenate((crossIF1, crossIF2))
        withinIF = diags[left:right - diag, diag]
        n1 = len(withinIF)
        n2 = len(crossIF)
        score = np.sign(np.nanmean(withinIF)-np.nanmean(crossIF))*2-1#np.count_nonzero(ratio >= minRatio)
        scale = n1*n2
        score = score / scale * (n1 + n2)
        N += (n1 + n2)
        scores[diag]= score
    return scores[1:],N

@cacheDelta
def Delta(data, offset, left, right, minRatio=1.1, mask=None):
    s = data.shape[0]
    if mask is not None:
        for i in range(len(mask)):
            l,r = mask[i]
            data[max([0, l - offset]):r - offset + 1, max([0, l - offset]):r - offset + 1] = np.nan
            data[max([0, l - offset - 4]):l - offset + 5,
            max([0, r - offset - 4]):r - offset + 5] = np.nan  # mask dot corner

    left = left - offset
    right = right - offset
    w = right - left
    diags = np.zeros((s, w + 2)) * np.nan

    for j in range(min([w + 2, s])):
        diagj = np.diagonal(data, j)
        diags[:len(diagj), j] = diagj
    scores,N=DeltaNB(diags, left, right, w, minRatio)
    return np.nansum(np.asarray(scores)) / (N+eps)


def merge(data, TADs, distance, minRatio=1.1):
    TADs = np.asarray(TADs)
    mergedTADs = []
    posTree = KDTree(TADs, leaf_size=30, metric='chebyshev')
    NNindexes, NNdists = posTree.query_radius(TADs, r=distance, return_distance=True)

    for i in range(len(NNindexes)):
        if len(NNindexes[i]) > 1:
            bestScore = -np.inf
            bestIdx = -1
            for j in range(len(TADs[NNindexes[i]])):
                l, r = TADs[NNindexes[i]][j]
                offset = max([0, 2 * l - r - 1])
                end = 2 * r - l + 1
                s = Delta(data=data[offset:end, offset:end].toarray(), offset=offset, left=l, right=r, minRatio=minRatio,mask=None)
                if s > bestScore:
                    bestScore = s
                    bestIdx = j
            mergedTADs.append(list(TADs[NNindexes[i][bestIdx]]))

        else:
            mergedTADs.append(list(TADs[NNindexes[i][0]]))
    _mergedTADs = []
    for l, r in mergedTADs:
        _mergedTADs.append((l, r))
    mergedTADs = list(set(_mergedTADs))
    if len(TADs) > len(mergedTADs):
        mergedTADs = merge(data, mergedTADs, distance, minRatio)
    return mergedTADs



def dp(data, lefts, rights, minDelta=0.3, minRatio=1.1, resol=5000, maxTAD=3000000, minTAD=30000, distance=50000):
    maxTAD = maxTAD / resol
    minTAD = minTAD / resol
    distance = distance / resol
    s = data.shape[0]

    boundaries = sorted(list(set(lefts + rights)))
    lefts = set(lefts)
    rights = set(rights)
    if boundaries[0] not in lefts:
        lefts.add(0)
        boundaries.insert(0, 0)
    if boundaries[-1] not in rights:
        rights.add(s - 1)
        boundaries.append(s - 1)
    n = len(boundaries)
    S = np.zeros((n, n))
    pairS = np.zeros((n, n))
    T = {}
    L = {}

    for i in range(len(boundaries)):
        if boundaries[i] not in T:
            T[boundaries[i]] = {}
            L[boundaries[i]] = {}
    K = np.zeros((n, n), dtype=int) - 1
    # Initialization
    for i in range(n - 1):
        if boundaries[i] not in T:
            T[boundaries[i]] = {}
            L[boundaries[i]] = {}
        if boundaries[i + 1] not in T[boundaries[i]]:
            T[boundaries[i]][boundaries[i + 1]] = []
            L[boundaries[i]][boundaries[i + 1]] = 0
        if boundaries[i] in lefts and boundaries[i + 1] in rights and maxTAD > boundaries[i + 1] - boundaries[
            i] > minTAD:
            offset = max([0, 2 * boundaries[i] - boundaries[i + 1] - 1])
            end = 2 * boundaries[i + 1] - boundaries[i] + 1
            s = Delta(data=data[offset:end, offset:end].toarray(), offset=offset, left=boundaries[i], right=boundaries[i + 1], mask=None,minRatio=minRatio)
            if s > minDelta:
                S[i, i + 1] = s
                pairS[i, i + 1] = s
                T[boundaries[i]][boundaries[i + 1]].append((boundaries[i], boundaries[i + 1]))
                L[boundaries[i]][boundaries[i + 1]] = 1

    # Foward pass
    for diag in tqdm(range(2, n)):
        for i in range(n):
            j = i + diag
            if j >= n:
                break

            bestScore = -np.inf
            bestTAD = []
            bestK = -1
            bestPairS = -1
            _pairS = 0
            _level = 0
            bestLevel  = 0

            for k in range(i + 1, j):
                s = S[i, k] + S[k, j]
                nestTad = T[boundaries[i]][boundaries[k]] + T[boundaries[k]][boundaries[j]]
                _level = max([L[boundaries[i]][boundaries[k]], L[boundaries[k]][boundaries[j]]])

                if boundaries[i] in lefts and boundaries[j] in rights and maxTAD > boundaries[j] - boundaries[
                            i] > minTAD:
                    offset = max([0, 2 * boundaries[i] - boundaries[j] - 1])
                    end = 2 * boundaries[j] - boundaries[i] + 1

                    s_delta = Delta(data=data[offset:end, offset:end].toarray(), offset=offset, left=boundaries[i], right=boundaries[j],
                                    minRatio=minRatio, mask=nestTad)
                    if s_delta > minDelta:
                        largestMetaTADL = -np.inf
                        for _l, _r in nestTad:
                            if _r - _l > largestMetaTADL:
                                largestMetaTADL = _r - _l
                        if largestMetaTADL / (boundaries[j] - boundaries[i]) < 10:#0.9:
                            nestTad = [(boundaries[i], boundaries[j])]
                            s = s + np.nanmax([s_delta, 0])
                            _pairS = np.nanmax([s_delta, 0])
                            _level += 1

                if s > bestScore:
                    bestScore = s
                    bestTAD = nestTad
                    bestK = k
                    bestLevel = _level
                    bestPairS = _pairS
            S[i, j] = bestScore
            K[i, j] = bestK
            pairS[i, j] = bestPairS
            T[boundaries[i]][boundaries[j]] = bestTAD
            L[boundaries[i]][boundaries[j]] = bestLevel

    # Backtracking
    finalTADs = []
    TADlevels = {}
    unvisted = [[0, n - 1]]
    while len(unvisted) > 0:
        i, j = unvisted[0]
        unvisted.remove([i, j])
        if len(T[boundaries[i]][boundaries[j]]) == 1 and T[boundaries[i]][boundaries[j]][0][0] == boundaries[i] and T[boundaries[i]][boundaries[j]][0][1] == boundaries[j]:
            TADlevels[(boundaries[i], boundaries[j])] = L[boundaries[i]][boundaries[j]]
        finalTADs = finalTADs + T[boundaries[i]][boundaries[j]]
        k = K[i, j]
        if k != -1:
            unvisted.append([i, k])
            unvisted.append([k, j])

    finalTADs = list(set(finalTADs))

    finalLefts = []
    finalRights = []
    unused = {'left': [], 'right': []}
    for l, r in finalTADs:
        finalLefts.append(l)
        finalRights.append(r)
    finalLefts = set(finalLefts)
    finalRights = set(finalRights)
    unused['left'] = lefts - finalLefts
    unused['right'] = rights - finalRights
    print(unused)


    finalTADs = merge(data, finalTADs, distance, minRatio)
    scores = []
    levels = []
    for l, r in finalTADs:
        # print(l, r, 'score=', pairS[boundaries.index(l), boundaries.index(r)])
        scores.append(pairS[boundaries.index(l), boundaries.index(r)])
        levels.append(TADlevels[(l,r)])

    return finalTADs, S[0, n - 1], scores,levels


@click.command()
@click.option('--resol', default=10000,type=int, help='resolution [10000]')
@click.option('--mintad', default=50000, type=int, help='min TAD size [50000]')
@click.option('--maxtad', default=3000000, type=int, help='max TAD size [3000000]')
@click.option('--distance', default=50000, type=int, help='max distance for merging two TADs [50000]')
@click.option('--ratio', default=1.2, type=float, help='min ratio for comparision [1.2]')
@click.option('--delta', default=0.2, type=float, help='min score for forming a possible pair [0.2]')
@click.option('--alpha', default=0.05, type=float, help='alpha-significant in FDR [0.05]')
@click.option('--chr', default=None, help='comma separated chromosomes')
@click.argument('coolfile', type=str, default=None, required=True)
@click.argument('boundaryfile', type=str, default=None, required=True)
@click.argument('prefix', type=str, default=None, required=True)
def fastassembly(delta, ratio, resol, maxtad, mintad, distance, boundaryfile, coolfile, prefix, alpha, chr):
    try:
        c = cooler.Cooler(coolfile+'::/resolutions/'+str(resol))
    except:
        c = cooler.Cooler(coolfile)
    if 'bin-size' in c.info and c.info['bin-size']!=resol:
        print('contact map at '+str(resol)+' does not exist!\nGood bye!')
        sys.exit(0)
    print("analysis at ",resol, 'resolution')
    boundaryfile = pd.read_csv(boundaryfile, sep='\t', header=None)
    boundaryfile[[1,2]]//=resol
    results = {'chrom':[],'start':[],'end':[],'score':[],'level':[]}
    if chr is None:
        chr = list(set(boundaryfile[0]))
    else:
        chr = chr.split(',')
    for _chr in chr:
        lefts = list(boundaryfile[(boundaryfile[0]==_chr)&(boundaryfile[4]==1)][1].to_numpy())
        rights = list(boundaryfile[(boundaryfile[0] == _chr) & (boundaryfile[6] == 1)][1].to_numpy())
        mat = c.matrix(balance=True,sparse=True).fetch(_chr).tocsr()

        # y=500
        # mat=mat[:y,:y]
        # lefts =np.asarray(lefts)
        # rights =np.asarray(rights)
        #
        # lefts = list(lefts[lefts < y])
        # rights = list(rights[rights < y])

        _TADs, _score, _scores,levels = dp(mat, lefts, rights, delta, ratio, resol, maxtad, mintad, distance)
        for i in range(len(_TADs)):
            l,r=_TADs[i]
            results['chrom'].append(_chr)
            results['start'].append(l*resol)
            results['end'].append((r+1)*resol)
            results['score'].append(_scores[i])
            results['level'].append(levels[i])
    pd.DataFrame.from_dict(results).to_csv(prefix+'.bed', index=False, header=False, float_format='%.4f',sep='\t')

    print('hitcache,',counts['hitcache'])
    print('misschace,',counts['misscache'])

