import numpy as np
import cooler
import click
from tqdm import tqdm
import pandas as pd
import numba
from numba import jit
eps = np.finfo(float).eps
from matplotlib import pylab as plt
import torch.nn as nn
import torch
import pandas as pd
from einops import rearrange
from torch.nn import functional as F
from matplotlib import pylab as plt
import numpy as np
import cooler
from sklearn.neighbors import KDTree
from matplotlib import pylab as plt
from scipy import stats


from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('wr',["w", "r"], N=256)


def delta(data, offset, left, right, minRatio=1.1, mask=None, weighted=True):

    data = data.copy()
    s = data.shape[0]
    if mask:
        for l, r in mask:
            data[np.max([0,l - offset]):r - offset+1, np.max([0,l - offset]):r - offset+1] = np.nan
            data[np.max([0,l - offset-4]):l - offset+5, np.max([0,r - offset-4]):r - offset+5] = np.nan # mask dot corner
    
    left = left - offset
    right = right - offset
    w = right - left
    diags = np.zeros((s, w + 2))*np.nan

    for j in range(np.min([w+2,s])):
        diagj =np.diagonal(data, j)
        diags[:len(diagj), j] = diagj
    # for i in range(s):
    #     for j in range(w + 2):
    #         if i + j < s:
    #             diags[i, j] = data[i, i + j]

    scores = []
    mus = []
    variances = []
    N=0
    for diag in range(1, w):
        crossIF1 = diags[np.max([0,left - diag]):left, diag] + eps
        crossIF1=crossIF1[~np.isnan(crossIF1)]
        crossIF2 = diags[right -diag:right, diag] + eps
        crossIF2=crossIF2[~np.isnan(crossIF2)]
        crossIF = np.concatenate([crossIF1, crossIF2])
        withinIF = diags[left:right - diag, diag] + eps 
        withinIF=withinIF[~np.isnan(withinIF)]


        if len(withinIF)<2 or len(crossIF)<2:
            continue
        ratio = np.outer(withinIF, 1 / crossIF)

        ratio = ratio[~np.isnan(ratio).any(axis=1), :]
        n1,n2 = ratio.shape
        win=np.sum(ratio > minRatio)
        loss=np.sum(ratio < 1 / minRatio)
        score =  win-loss 
        scale = win+loss+eps
        score = score /scale*(n1+n2)
        N+=(n1+n2)

        scores.append(score)

    return np.nansum(scores)/N


# def delta(data, offset, left, right, minRatio=1.1, mask=None,weighted=True):
#     data = data.copy()
#     s = data.shape[0]
#     if mask:
#         for l,r in mask:
#             data[l-offset:r-offset,l-offset:r-offset] = np.nan
#     left = left-offset
#     right = right-offset
#     w = right-left
#     diags = np.zeros((s, w+1))

#     for i in range(s):
#         for j in range(w+1):
#             if i+j < s:
#                 diags[i, j] = data[i, i+j]

#     scores=[]
#     for diag in range(1, w ):
#         crossIF1 = diags[left - diag:left, diag] + eps
#         crossIF2 = diags[right+1:right + diag +1, diag] + eps
#         # crossIF = np.concatenate([crossIF1,crossIF2])
#         withinIF = diags[left:right-diag, diag] + eps
#         # ratio 1
#         ratio1 = np.outer(withinIF, 1 / crossIF1)
#         ratio1 =ratio1[~np.isnan(ratio1).any(axis=1),:]
#         score1 = (ratio1>minRatio)-1*(ratio1<1/minRatio)
#         score1 = np.sum(score1)
#         # ratio 2
#         ratio2 = np.outer(withinIF, 1 / crossIF2)
#         ratio2 =ratio2[~np.isnan(ratio2).any(axis=1),:]
#         score2 = (ratio2>minRatio)-1*(ratio2<1/minRatio)
#         score2 = np.sum(score2)
#         if weighted:
#             scale1 = (ratio1>minRatio)+1*(ratio1<1/minRatio)
#             scale1[scale1==0] = 1
#             scale1 = np.sum(scale1)
#             score1 = score1/scale1
            
#             scale2 = (ratio2>minRatio)+1*(ratio2<1/minRatio)
#             scale2[scale2==0] = 1
#             scale2 = np.sum(scale2)
#             score2 = score2/scale2
            
#             score = np.mean([score1,score2])
#         scores.append(score)
#     return np.mean(scores)

def pval(data, offset, left, right, minRatio=1.1, mask=None, weighted=True,perms=100):

    data = data.copy()
    s = data.shape[0]
    if mask:
        for l, r in mask:
            data[np.max([0,l - offset]):r - offset+1, np.max([0,l - offset]):r - offset+1] = np.nan
            data[np.max([0,l - offset-4]):l - offset+5, np.max([0,r - offset-4]):r - offset+5] = np.nan # mask dot corner
    
    left = left - offset
    right = right - offset
    w = right - left
    oriDiags = np.zeros((s, w + 2))*np.nan

    for i in range(s):
        for j in range(w + 2):
            if i + j < s:
                oriDiags[i, j] = data[i, i + j]
    scores=[]
    for perm in range(perms+1):
        diags=oriDiags.copy()
        if perm>0:
            for i in range(0,diags.shape[-1]):
                _tmp=diags[~np.isnan(diags[:,i]),i]
                np.random.shuffle(_tmp)
                diags[~np.isnan(diags[:,i]),i]=_tmp
        _scores = []
        for diag in range(1, w):
            crossIF1 = diags[np.max([0,left - diag]):left, diag] + eps
            crossIF1=crossIF1[~np.isnan(crossIF1)]
            crossIF2 = diags[right -diag:right, diag] + eps
            crossIF2=crossIF2[~np.isnan(crossIF2)]
            crossIF = np.concatenate([crossIF1, crossIF2])
            withinIF = diags[left:right - diag, diag] + eps 
            withinIF=withinIF[~np.isnan(withinIF)]
            if len(withinIF)<2 or len(crossIF)<2:
                continue
            ratio = np.outer(withinIF, 1 / crossIF)
            # print(ratio.shape)
            ratio = ratio[~np.isnan(ratio).any(axis=1), :]
            # print(' -->',ratio.shape)
            n1,n2 = ratio.shape
            score = (ratio > minRatio) - 1 * (ratio < 1 / minRatio)
            score = np.sum(score)
            if weighted:
                scale = np.sum((ratio > minRatio) + (ratio < 1 / minRatio))
                if scale==0:
                    continue
                score = score /scale#/(n1+n2+1)#(n1*n2)
            _scores.append(score)
        scores.append(np.nanmean(_scores))
    return 1-np.sum(scores[0]>scores[1:])/perms

def filtering(data, TADs, minRatio=1.1, mask=None, weighted=True):
    newTADs = []
    for l,r in TADs:
        TADsize = r-l
        i = np.max([0,l-20])
        j = np.min([i+40,data.shape[0]])
        scores=[]
        offset = np.max([0,i - TADsize - 1])
        end = j+TADsize + 1
        mask=[]
        for nl,nr in TADs:
            if nl==l and nr==r:
                continue
            if nl>=offset and nr<end:
                mask.append([nl,nr])
        for _l in range(i,j+1):
            _r = np.min([data.shape[0],_l+TADsize])
            s=delta(data[offset:end,offset:end], offset, _l, _r, mask=mask,minRatio=minRatio,weighted=True)
            scores.append(s)
        print(l,r,scores,scores[l-i],np.sum(scores[l-i]>scores)/len(scores))
        if np.sum(scores[l-i]>scores)/len(scores)>0.2:
            newTADs.append((l,r))
        
    return newTADs

def addUnusedTADB(data,TADs,lefts,rights,boundaries,unused,minRatio=1.1,minDelta=0.3):
    unusedleft=list(unused['left'])
    unusedright=list(unused['right'])
    for l in unusedleft:
        for i in range(boundaries.index(l)+1,len(boundaries)):
            r=boundaries[i]
            if r not in rights:
                continue
            offset = np.max([0,2*l - r - 1])
            end = 2*r - l + 1
            s=delta(data[offset:end,offset:end], offset, l, r,minRatio)
            if s>minDelta:
                TADs.append((l,r))
                break

    for r in unusedright:
        for i in range(boundaries.index(r)-1,-1,-1):
            l=boundaries[i]
            if l not in lefts:
                continue
            offset = np.max([0,2*l - r - 1])
            end = 2*r - l + 1
            s=delta(data[offset:end,offset:end], offset, l, r,minRatio)
            if s>minDelta:
                TADs.append((l,r))
                break
    return TADs


def merge(data,TADs,distance,minRatio=1.1):
    TADs=np.asarray(TADs)
    mergedTADs = []
    posTree = KDTree(TADs, leaf_size=30, metric='chebyshev')
    NNindexes, NNdists = posTree.query_radius(TADs, r=distance, return_distance=True)


    for i in range(len(NNindexes)):
        if len(NNindexes[i])>1:
            bestScore=-np.inf
            bestIdx = -1
            for j in range(len(TADs[NNindexes[i]])):
                l,r=TADs[NNindexes[i]][j]
                offset = np.max([0,2*l - r - 1])
                end = 2*r - l + 1
                s=delta(data[offset:end,offset:end], offset, l, r,minRatio=minRatio)
                if s>bestScore:
                    bestScore = s
                    bestIdx = j
            mergedTADs.append(list(TADs[NNindexes[i][bestIdx]]))

        else:
            mergedTADs.append(list(TADs[NNindexes[i][0]]))
    _mergedTADs = []
    for l,r in mergedTADs:
        _mergedTADs.append((l,r))
    mergedTADs = list(set(_mergedTADs))
    if len(TADs)>len(mergedTADs):
        mergedTADs=merge(data,mergedTADs,distance,minRatio)
    return mergedTADs


# @jit(nopython=False, parallel=True,error_model='python')
def assembly(data, lefts, rights,minDelta=0.3,minRatio=1.1,resol=5000,maxTAD=3000000,minTAD=30000,distance=50000):
    maxTAD =maxTAD/resol
    minTAD = minTAD/resol
    distance= distance/resol
    s = data.shape[0]

    boundaries = sorted(list(set(lefts+rights)))
    lefts = set(lefts)
    rights = set(rights)
    if boundaries[0] not in lefts:
        lefts.add(0)
        boundaries.insert(0,0)
    if boundaries[-1] not in rights:
        rights.add(s-1)
        boundaries.append(s-1)
    n = len(boundaries)
    S = np.zeros((n,n))
    pairS = np.zeros((n,n))
    T = {}
    K = np.zeros((n,n),dtype=int)-1
    # Initialization   
    for i in range(n-1):
        if boundaries[i] not in T:
            T[boundaries[i]] = {}
        if boundaries[i+1] not in T[boundaries[i]]:
            T[boundaries[i]][boundaries[i+1]] = []
        if boundaries[i] in lefts and boundaries[i+1] in rights and boundaries[i+1]-boundaries[i]<maxTAD and boundaries[i+1]-boundaries[i]>minTAD:
            offset = np.max([0,2*boundaries[i] - boundaries[i+1] - 1])
            end = 2*boundaries[i+1] - boundaries[i] + 1
            s=delta(data[offset:end,offset:end], offset, boundaries[i], boundaries[i+1],minRatio=minRatio)
            if s > minDelta:
                S[i,i+1] = s
                pairS[i,i+1]=s
                T[boundaries[i]][boundaries[i+1]].append((boundaries[i],boundaries[i+1]))
                


    # Foward pass
    for diag in tqdm(range(2,n)):
        for i in range(n):
            j = i+diag
            if j>=n:
                break

            bestScore = -np.inf
            bestTAD= []
            bestK = -1
            bestPairS = -1
            _pairS = 0

            for k in numba.prange(i+1,j):
                s = S[i,k] + S[k,j]

                # print(i,k,boundaries[i],boundaries[k])
                # print(k,j,boundaries[k],boundaries[j])
                nestTad = T[boundaries[i]][boundaries[k]] + T[boundaries[k]][boundaries[j]]
                # print(nestTad)
                if boundaries[i] in lefts and boundaries[j] in rights and boundaries[j]-boundaries[i]<maxTAD and boundaries[j]-boundaries[i]>minTAD:
                    offset = np.max([0,2*boundaries[i] - boundaries[j] - 1])
                    end = 2*boundaries[j] - boundaries[i] + 1
                    s_delta=delta(data[offset:end,offset:end], offset, boundaries[i], boundaries[j], minRatio=minRatio, mask=nestTad,weighted=True)
                    if s_delta > minDelta:
                        largestMetaTADL=-np.inf
                        for _l,_r in nestTad:
                            if _r-_l>largestMetaTADL:
                                largestMetaTADL=_r-_l
                        if largestMetaTADL/(boundaries[j]-boundaries[i])<0.9:
                            nestTad=[(boundaries[i],boundaries[j])]
                            s = s+np.nanmax([s_delta,0])
                            _pairS=np.nanmax([s_delta,0])
           
                if s> bestScore:
                    bestScore=s
                    bestTAD = nestTad
                    bestK = k
                    bestPairS = _pairS
            S[i,j] = bestScore
            K[i,j] = bestK
            pairS[i,j] = bestPairS
            if boundaries[i] not in T:
                T[boundaries[i]] = {}
            T[boundaries[i]][boundaries[j]] = bestTAD
    # print(S)
    # print(boundaries,lefts,rights)
            
    # Backtracking
    finalTADs = []
    unvisted =[[0,n-1]]
    while len(unvisted)>0:
        i,j = unvisted[0]
        unvisted.remove([i,j])
        finalTADs=finalTADs+T[boundaries[i]][boundaries[j]]
        k = K[i,j]
        if k!=-1:
            unvisted.append([i,k])
            unvisted.append([k,j])

    finalTADs = list(set(finalTADs))

    finalLefts=[]
    finalRights = []
    unused={'left':[],'right':[]}
    for l,r in finalTADs:
        finalLefts.append(l)
        finalRights.append(r)
    finalLefts=set(finalLefts)
    finalRights =set(finalRights)
    unused['left'] = lefts-finalLefts
    unused['right'] = rights-finalRights
    print(unused)
#     # print(boundaries)
#     finalTADs=addUnusedTADB(data,finalTADs,lefts,rights,boundaries,unused,minRatio,minDelta)
#     finalTADs = list(set(finalTADs))

#     finalLefts=[]
#     finalRights = []
#     unused={'left':[],'right':[]}
#     for l,r in finalTADs:
#         finalLefts.append(l)
#         finalRights.append(r)
#     finalLefts=set(finalLefts)
#     finalRights =set(finalRights)
#     unused['left'] = lefts-finalLefts
#     unused['right'] = rights-finalRights
#     print(unused)

    finalTADs=merge(data,finalTADs,distance,minRatio)
    scores=[]
    for l,r in finalTADs:
        print(l,r, 'score=',pairS[boundaries.index(l),boundaries.index(r)])
        scores.append(pairS[boundaries.index(l),boundaries.index(r)])
    ######################################################################## filtering by p-value
    # maxScore=sorted(scores)[len(scores)//5+1]
    # newfinalTADs=[]
    # newscores=[]
    # for i in range(len(finalTADs)):
    #     if scores[i]>=maxScore:
    #         newfinalTADs.append(finalTADs[i])
    #         newscores.append(scores[i])
    #     else:
    #         l,r = finalTADs[i]
    #         nestTad=[]
    #         for _l,_r in finalTADs:
    #             if _l>=l and _r<=r and _r-_l<r-l:
    #                 nestTad.append([_l,_r])
    #         offset = np.max([0,2*l - r - 1])
    #         end = 2*r - l + 1
    #         _pval=pval(data[offset:end,offset:end], offset, l, r, minRatio=minRatio, mask=nestTad)
    #         print(l,r,_pval)
    #         if _pval<0.05:
    #             newfinalTADs.append(finalTADs[i])
    #             newscores.append(scores[i])
    # finalTADs=newfinalTADs
    # scores = newscores
    return finalTADs,S[0,n-1],scores



###############################################################################################
TADB = pd.read_csv('chr20_alpha0.01.bedpe.bed',header=None,sep='\t')
TADB=TADB[TADB[0]=='chr20']
lefts=TADB[TADB[4]==1][1]//5000
rights=TADB[TADB[6]==1][1]//5000
c=cooler.Cooler('4DNFIXP4QG5B_Rao2014_GM12878.mcool::/resolutions/5000')
mat=c.matrix(balance=True).fetch('chr20')
y=3000
left = list(lefts[lefts<y])
right = list(rights[rights<y])
m = mat[:y,:y]

TADs3,score,scores=assembly(m,left,right,minDelta=0,minRatio=1.1,maxTAD=1500000)

