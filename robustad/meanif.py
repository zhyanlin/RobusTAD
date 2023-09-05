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
from robustad.util import distanceNormalization_by_mean

from matplotlib import pylab as plt

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
        crossIF1 = diags[max(0, left - diag):left, diag] + eps
        crossIF1 = crossIF1[~np.isnan(crossIF1)]
        crossIF2 = diags[right - diag:right, diag] + eps
        crossIF2 = crossIF2[~np.isnan(crossIF2)]
        crossIF = np.concatenate((crossIF1, crossIF2))
        withinIF = diags[left:right - diag, diag] + eps
        withinIF = withinIF[~np.isnan(withinIF)]
        # if len(withinIF) < 2 or len(crossIF) < 2:
        #     continue
        ratio = np.outer(withinIF, 1 / crossIF)
        n1, n2 = ratio.shape
        win = np.sum(ratio > minRatio)
        loss = np.sum(ratio < 1 / minRatio)
        score = win - loss
        scale = np.sum(ratio>-1)#
        # scale=win + loss + eps
        score = score / scale * (n1 + n2)
        N += (n1 + n2)
        scores[diag]= score
    return scores[1:],N

@cacheDelta
def Delta(data, offset, left, right, mask=None):
    s = data.shape[0]
    if mask is not None:
        for i in range(len(mask)):
            l,r = mask[i]
            data[np.max(np.asarray([0, l - offset])):r - offset + 1, np.max(np.asarray([0, l - offset])):r - offset + 1] = np.nan
            data[np.max(np.asarray([0, l - offset - 4])):l - offset + 5,
            np.max(np.asarray([0, r - offset - 4])):r - offset + 5] = np.nan  # mask dot corner

    left = left - offset
    right = right - offset
    # if np.nanmean(data[left:right,left:right])<0:
    #     print(data[left:right,left:right])

    meanif=np.nanmean(data[left:right,left:right])
    if np.isnan(meanif):
        return 0
    return meanif


@click.command()
@click.option('--resol', default=10000,type=int, help='resolution [10000]')
@click.argument('coolfile', type=str, default=None, required=True)
@click.argument('tadfile', type=str, default=None, required=True)
@click.argument('output', type=str, default=None, required=True)
def meanif(resol, tadfile, coolfile,output):
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

    chr = list(set(tadfile[0]))

    results=[]
    for _chr in chr:
        cmap = c.matrix(balance=True, sparse=False).fetch(_chr)
        cmap=distanceNormalization_by_mean(cmap)
        # plt.figure()
        # plt.imshow(np.log(cmap[:2000,:2000]))
        # plt.show()
        chrdata = tadfile[tadfile[0]==_chr][[0,1,2]].reset_index(drop=True)
        tads = []
        for i in range(len(chrdata)):
            tads.append((chrdata[1][i],chrdata[2][i]))
        deltas = []
        for tad in tads:
            # nestTADs=[]
            # for tad2 in tads:
            #     if tad2[0]>=tad[0] and tad2[1]<=tad[1]:
            #         if tad2[0]==tad[0] and tad2[1] == tad[1]:
            #             pass
            #         else:
            #             nestTADs.append(tad2)
            l, r = tad[0],tad[1]
            offset = np.max([0, 2 * l - r - 1])
            end = 2 * r - l + 1
            # if len(nestTADs)==0:
            _delta = Delta(data=cmap[offset:end, offset:end].copy(), offset=offset, left=l, right=r,mask=None)
            # else:
            #     _delta = -1
            deltas.append(_delta)
        results.append(np.concatenate([chrdata.to_numpy(),np.asarray(deltas)[:,None]],axis=1))
    results=np.concatenate(results,axis=0)
    pd.DataFrame.from_dict(results).to_csv(output, index=False, header=False, float_format='%.4f', sep='\t')

