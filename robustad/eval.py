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
        crossIF=crossIF[~np.isnan(crossIF)] + eps
        withinIF = diags[left:right - diag, diag] + eps
        withinIF=withinIF[~np.isnan(withinIF)]

        ratio = np.outer(withinIF, 1 / crossIF)
        n1, n2 = ratio.shape
        win = np.count_nonzero(ratio >= minRatio)
        loss = np.count_nonzero(ratio <= 1 / minRatio)
        score = win - loss
        scale = n1*n2
        # scale=win + loss + eps
        score = score / scale * (n1 + n2)
        N += (n1 + n2)
        scores[diag]= score
    return scores[1:],N

@cacheDelta
def Delta(data, offset, left, right, minRatio=1.1, mask=None):
    data = data.copy()
    # print(left,right,'.....')

    s = data.shape[0]
    if mask is not None:
        for i in range(len(mask)):
            l,r = mask[i]
            data[np.max(np.asarray([0, l - offset])):r - offset + 1, np.max(np.asarray([0, l - offset])):r - offset + 1] = np.nan
            data[np.max(np.asarray([0, l - offset - 4])):l - offset + 5,
            np.max(np.asarray([0, r - offset - 4])):r - offset + 5] = np.nan  # mask dot corner

    left = left - offset
    right = right - offset
    w = right - left
    diags = np.zeros((s, w + 2)) * np.nan

    for j in range(np.min(np.asarray([w + 2, s]))):
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
                offset = np.max([0, 2 * l - r - 1])
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







@click.command()
@click.option('--resol', default=10000,type=int, help='resolution [10000]')
@click.option('--ratio', default=1, type=float, help='min ratio for comparision [1.]')
@click.argument('coolfile', type=str, default=None, required=True)
@click.argument('tadfile', type=str, default=None, required=True)
@click.argument('output', type=str, default=None, required=True)
def eval(resol, ratio, tadfile, coolfile,output):
    try:
        c = cooler.Cooler(coolfile+'::/resolutions/'+str(resol))
    except:
        c = cooler.Cooler(coolfile)
    if 'bin-size' in c.info and c.info['bin-size']!=resol:
        print('contact map at '+str(resol)+' does not exist!\nGood bye!')
        sys.exit(0)
    print("analysis at ",resol, 'resolution')
    tadfile = pd.read_csv(tadfile, sep='\t', header=None)
    tadfile[[1,2]]//=resol
    results = {'chrom':[],'start':[],'end':[],'score':[],'level':[]}

    chr = list(set(tadfile[0]))

    results=[]
    for _chr in chr:
        cmap = c.matrix(balance=True, sparse=False).fetch(_chr)#.tocsr()
        cmap[np.diag_indices(cmap.shape[0], 2)] = np.nan
        chrdata = tadfile[tadfile[0]==_chr][[0,1,2]].reset_index(drop=True)
        tads = []
        for i in range(len(chrdata)):
            tads.append((chrdata[1][i],chrdata[2][i]))
        deltas = []
        for tad in tads:
            nestTADs=[]
            for tad2 in tads:
                if tad2[0]>=tad[0] and tad2[1]<=tad[1]:
                    if tad2[0]==tad[0] and tad2[1] == tad[1]:
                        pass
                    else:
                        nestTADs.append(tad2)
            l, r = tad[0],tad[1]
            offset = np.max([0, 2 * l - r - 1])
            end = 2 * r - l + 1
            if len(nestTADs) > -1:
                _delta = Delta(data=cmap[offset:end, offset:end], offset=offset, left=l, right=r, minRatio=ratio,mask=nestTADs)
            else:
                _delta = -1
            deltas.append(_delta)
        results.append(np.concatenate([chrdata.to_numpy(),np.asarray(deltas)[:,None]],axis=1))
    results=np.concatenate(results,axis=0)
    pd.DataFrame.from_dict(results).to_csv(output, index=False, header=False, float_format='%.4f', sep='\t')
    #     lefts = list(boundaryfile[(boundaryfile[0]==_chr)&(boundaryfile[4]==1)][1].to_numpy())
    #     rights = list(boundaryfile[(boundaryfile[0] == _chr) & (boundaryfile[6] == 1)][1].to_numpy())
    #     mat = c.matrix(balance=True,sparse=True).fetch(_chr).tocsr()
    #
    #     # y=500
    #     # mat=mat[:y,:y]
    #     # lefts =np.asarray(lefts)
    #     # rights =np.asarray(rights)
    #     #
    #     # lefts = list(lefts[lefts < y])
    #     # rights = list(rights[rights < y])
    #
    #     _TADs, _score, _scores,levels = dp(mat, lefts, rights, delta, ratio, resol, maxtad, mintad, distance)
    #     for i in range(len(_TADs)):
    #         l,r=_TADs[i]
    #         results['chrom'].append(_chr)
    #         results['start'].append(l*resol)
    #         results['end'].append((r+1)*resol)
    #         results['score'].append(_scores[i])
    #         results['level'].append(levels[i])
    # pd.DataFrame.from_dict(results).to_csv(prefix+'.bed', index=False, header=False, float_format='%.4f',sep='\t')
    #
    # print('hitcache,',counts['hitcache'])
    # print('misschace,',counts['misscache'])
    #
