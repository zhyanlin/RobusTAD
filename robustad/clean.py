import numpy as np
import cooler
import click
from tqdm import tqdm
import pandas as pd
from importlib_resources import files
import numba
from numba import jit
import sys
from sklearn.neighbors import KDTree
import functools
from matplotlib import  pylab as plt
from scipy.stats import pearsonr as pcc
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('wr',["w", "r"], N=256)

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

        # n1 = len(withinIF)
        # n2 = len(crossIF)
        # if n1>0 and n2>0:
        #     win=timesOfAgeB(withinIF,crossIF)
        #     loss = timesOfAgeB(crossIF,withinIF)
        # else:
        #     win=1
        #     loss=1


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

eps = np.finfo(float).eps
looptpl = np.loadtxt(str(files('robustad').joinpath('data/looptpl.txt')))
def filterTAD(data, offset, left, right, minRatio=1.1):
    scores = []
    cscore=0
    idx = []
    for i in range(-5, 6):
        _left = left + i
        _right = right + i
        if _left-offset<0 or _right-offset>=data.shape[0]:
            continue
        score = Delta(data=data, offset=offset, left=_left, right=_right, minRatio=minRatio,mask=None)
        if i==0:
            cscore = score
        scores.append(score)
        idx.append(i)
    return scores, idx,cscore


def popOneL0TAD(tads):
    for tad in tads:
        nestTADs = []
        for tad2 in tads:
            if tad2[0] >= tad[0] and tad2[1] <= tad[1]:
                if tad2[0] == tad[0] and tad2[1] == tad[1]:
                    pass
                else:
                    nestTADs.append(tad2)

        if len(nestTADs) == 0:
            tads.remove(tad)
            return tad, tads



def o2oe(o,l,r,diagExp):
    d=o*0
    for i in range(o.shape[0]):
        for j in range(o.shape[1]):
            d[i,j] = diagExp[abs(r-l + j-i)]
    return o/d



def cleanTAD(tads=None,data=None):
    goodtads = []
    badtads = []

    diagExp = []
    for i in range(data.shape[0]):
        diagExp.append(np.nanmean(np.diag(data, i)))
    diagExp = np.asarray(diagExp)+eps

    data[np.diag_indices(data.shape[0], 2)] = np.nan

    while len(tads) > 0:
        # print(len(tads))
        tad, tads = popOneL0TAD(tads)
        l, r = tad

        start = max([0, 2 * l - r - 1 - 30])
        end = 2 * r - l + 1 + 30
        # print('start,end',start,end)
        m =data[start:end, start:end].copy()
        # if r-l>100:
        #     for diagoff in range(1,50):
        #         m[:-diagoff, diagoff:][np.diag_indices(m.shape[0] - diagoff, 2)] = np.nan
        s, idx,cscore = filterTAD(m, start, l, r)
        peak = np.abs(idx[np.argmax(s)])
        rank = np.argwhere(np.argsort(-np.asarray(s)) == 5).flatten()[0]
        # print(l,r)
        corner = o2oe(data[l - 8:l + 9, r - 8:r + 9], l - 8, r - 8, diagExp)
        if corner.shape == looptpl.shape and np.sum(~np.isnan(corner)) > 100:
            cornerpcc = pcc(corner[~np.isnan(corner)], looptpl[~np.isnan(corner)])[0]
        else:
            cornerpcc = -1
        offset = 0
        if (peak <= 4 and rank <= 5) or r - l < 20 or cornerpcc > 0.3:
            data[np.max(np.asarray([0, l - offset])):r - offset + 1,
            np.max(np.asarray([0, l - offset])):r - offset + 1] = np.nan
            data[np.max(np.asarray([0, l - offset - 4])):l - offset + 5,
            np.max(np.asarray([0, r - offset - 4])):r - offset + 5] = np.nan  # mask dot corner
            goodtads.append((l, r))
            # print('cscore',cscore,l,r)
        else:
            badtads.append((l, r))

    return goodtads,badtads


def addback(boundaries,dpTads,data,resol=5000):
    result =[]
    diagExp = []
    badnum=0

    for i in range(data.shape[0]):
        diagExp.append(np.nanmean(np.diag(data, i)))
    diagExp = np.asarray(diagExp)+eps

    left = list(boundaries[boundaries[4]==1][1]//resol)
    right = list(boundaries[boundaries[6] == 1][1] // resol)
    maxij=max([max(left),max(right)])
    label=np.zeros(maxij+1)
    for i in range(len(left)):
        label[left[i]] = 1
    for i in range(len(right)):
        if label[right[i]] == 1:
            label[right[i]] = 3
        else:
            label[right[i]]=2
    leftidx = np.argwhere((label==1) | (label==3)).flatten()
    for l in leftidx:
        for r in range(l+6,min([maxij,l+5000000//resol])):
            if label[r] >= 2:
                corner = o2oe(data[l - 8:l + 9, r - 8:r + 9], l - 8, r - 8, diagExp)
                if np.sum(~np.isnan(corner)) > 100:
                    cornerpcc = pcc(corner[~np.isnan(corner)], looptpl[~np.isnan(corner)])[0]
                else:
                    cornerpcc = -1
                if cornerpcc>0.3:
                    start = max([0,l-50])

                    nestTADs = []
                    for tad2 in dpTads:
                        if tad2[0] >= l and tad2[1] <= r:
                            if tad2[0] == l and tad2[1] == r:
                                pass
                            else:
                                nestTADs.append(tad2)

                    score = Delta(data=data[start:2*r-l,start:2*r-l].copy(), offset=start, left=l, right=r, minRatio=1, mask=nestTADs)
                    # print('score',l,r,score)
                    if score > 0.:
                        result.append((l,r))
                else:
                    badnum+=1
    # print('add back',len(result))
    # print('bad',badnum)
    return result


def summary(HqTads,LqTads,data,chr,resol,minRatio=1):
    nestedScore=[]
    score = []
    Hq = []
    chrs = []
    startbp=[]
    endbp=[]
    level=[]
    # compute nestedScore , i.e. TADScore without masking subTADs
    for l,r in HqTads:
        start = max([0, 2 * l - r - 1])
        end = 2 * r - l + 1
        _nestedScore = Delta(data=data[start:end, start:end].copy(), offset=start, left=l, right=r,minRatio=minRatio, mask=None)
        nestedScore.append(_nestedScore)
        chrs.append(chr)
        Hq.append(1)
        startbp.append(l*resol)
        endbp.append(r*resol)
    for l, r in LqTads:
        start = max([0, 2 * l - r - 1])
        end = 2 * r - l + 1
        _nestedScore = Delta(data=data[start:end, start:end].copy(), offset=start, left=l, right=r, minRatio=minRatio,mask=None)
        nestedScore.append(_nestedScore)
        chrs.append(chr)
        Hq.append(0)
        startbp.append(l * resol)
        endbp.append(r * resol)
    # compute TADScore, i.e. TADScore by masking subTADs
    for tad in HqTads:
        nestTADs=[]
        for tad2 in HqTads:
            if tad2[0]>=tad[0] and tad2[1]<=tad[1]:
                if tad2[0]==tad[0] and tad2[1] == tad[1]:
                    pass
                else:
                    nestTADs.append(tad2)
        l, r = tad[0],tad[1]
        start = np.max([0, 2 * l - r - 1])
        end = 2 * r - l + 1
        _score = Delta(data=data[start:end, start:end].copy(), offset=start, left=l, right=r, minRatio=minRatio,mask=nestTADs)
        score.append(_score)
    for tad in LqTads:
        nestTADs = []
        for tad2 in HqTads:
            if tad2[0] >= tad[0] and tad2[1] <= tad[1]:
                if tad2[0] == tad[0] and tad2[1] == tad[1]:
                    pass
                else:
                    nestTADs.append(tad2)
        l, r = tad[0], tad[1]
        start = np.max([0, 2 * l - r - 1])
        end = 2 * r - l + 1
        _score = Delta(data=data[start:end, start:end].copy(), offset=start, left=l, right=r, minRatio=minRatio,
                       mask=nestTADs)
        score.append(_score)
    # compute TAD level
    levelPerBp = np.zeros(data.shape[0])-1
    tad2level={}
    tads = HqTads.copy()
    while len(tads) > 0:
        tad, tads = popOneL0TAD(tads)
        levelPerBp[tad[0]:tad[1]+1]+=1
        tad2level[(tad[0],tad[1])]=np.max(levelPerBp[tad[0]:tad[1]+1])
    tads = LqTads.copy()
    while len(tads) > 0:
        tad, tads = popOneL0TAD(tads)
        levelPerBp[tad[0]:tad[1] + 1] += 1
        tad2level[(tad[0], tad[1])] = np.max(levelPerBp[tad[0]:tad[1] + 1])
    for tad in HqTads:
        level.append(tad2level[(tad[0],tad[1])])
    for tad in LqTads:
        level.append(tad2level[(tad[0], tad[1])])
    return pd.DataFrame.from_dict({'chrom':chrs,'startbp':startbp,'endbp':endbp,'score':score,'nestedScore':nestedScore,'Hq':Hq,'level':level})

@click.command()
@click.option('--resol',type=int,default=5000,help='resol [5000]')
@click.argument('tads', type=str, default=None, required=True)
@click.argument('boundaries', type=str, default=None, required=True)
@click.argument('cool', type=str, default=None, required=True)
def clean(tads,boundaries,cool,resol):
    '''
    Filtering TADs [optional step].
    '''
    goodf = tads.replace('.bed','') + '_cleaned.bed'
    tads=pd.read_csv(tads,header=None,sep='\t')
    boundaries=pd.read_csv(boundaries,header=None,sep='\t')


    pdresults=[]
    if cooler.fileops.is_cooler(cool):
        c = cooler.Cooler(cool)
        if c.info['bin-size']!=resol:
            print('contact map at '+str(resol)+' resolution does not exist!\nGood bye!')
            sys.exit(0)
    else:
        try:
            c = cooler.Cooler(cool + '::/resolutions/' + str(resol))
        except:
            print('contact map at '+str(resol)+' resolution does not exist!\nGood bye!')
            sys.exit(0)

    for chr in tqdm(set(tads[0])):
        tadchr=tads[tads[0]==chr].reset_index(drop=True)
        boundarychr = boundaries[boundaries[0]==chr].reset_index(drop=True)
        ctads=[]
        for i in range(len(tadchr)):
            ctads.append((tadchr[1][i]//resol,tadchr[2][i]//resol))
        mat = c.matrix(balance=True,sparse=False).fetch(chr)
        addTADs = addback(boundarychr,ctads, mat,resol=resol)

        goodtad,badtad = cleanTAD(ctads,mat)
        del mat
        for l,r in addTADs:
            if (l,r) not in goodtad:
                goodtad.append((l,r))
        mat = c.matrix(balance=True, sparse=False).fetch(chr)
        pdresults.append(summary(goodtad, badtad, mat, chr, resol, minRatio=1))
        del mat

    result=pd.concat(pdresults)
    result[(result['Hq']==1) & (result['score']>0)].to_csv(goodf,sep='\t',index=False,header=False)
    # result.to_csv(allf, sep='\t', index=False, header=False)







