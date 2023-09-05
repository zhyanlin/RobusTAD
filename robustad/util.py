import numpy as np
from numba import jit
import numba as nb
from tqdm import tqdm
import click

eps = np.finfo(float).eps

@nb.njit
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@nb.njit
def nansum(array, axis=1):
    return np_apply_along_axis(np.nansum, axis, array)



@jit(nopython=True, parallel=True)
def l1Error(x,y):
    idx = np.arange(1, x.shape[0])
    loss = np.sum(np.abs((y[-1] - y[0]) / (x[-1] - x[0]) * (x[idx] - x[0]) + y[0] - y[idx]))
    return loss



@jit(nopython=True)
def getTurningPoints(x, y,w=1):
    ''' fits TAD boundary score curve by capturing major trend via finding turning points greedly

    Args:
        x: 1D, index of loci
        y: 1D, score at loci

    Return:
        turningPoints: index of turning points
    '''
    # w = np.max([w,1e-10])
    if w<1e-10:
        w=1e-10
    turningPoints = []
    p_start_idx = 0
    p_last_idx = x.shape[0] - 1
    turningPoints.append(p_start_idx)
    p_next_idx = p_start_idx + 1
    score = 1
    while p_next_idx < p_last_idx:
        seglen = np.power(np.power(x[p_next_idx] - x[p_start_idx], 2) + np.power(y[p_next_idx] - y[p_start_idx], 2), 0.5)
        new_score = w*seglen - l1Error(x[p_start_idx:p_next_idx + 1], y[p_start_idx:p_next_idx + 1])
        if new_score < score:
            turningPoints.append(p_next_idx - 1)
            p_start_idx = p_next_idx - 1
            score = 0
        else:
            p_next_idx += 1
            score = new_score
    turningPoints.append(p_last_idx)
    return np.asarray(turningPoints)


def refineTurningPoints(x,y, pointsIdx,win=2):
    '''
    refine turning points find by the gready approach. The goal is to refine turning points
    as local exteme locus around it. By doing it, we can capture the boundary score trend with
    called boundaries
    '''
    length = x.shape[0]
    peak = []
    trough = []
    rest = [pointsIdx[0], pointsIdx[-1]]
    for i in range(1, pointsIdx.shape[0] - 1):
        if y[pointsIdx[i]] > y[pointsIdx[i - 1]] and y[pointsIdx[i]] >= y[pointsIdx[i + 1]]:
            peak.append(pointsIdx[i])
        elif y[pointsIdx[i]] < y[pointsIdx[i - 1]] and y[pointsIdx[i]] <= y[pointsIdx[i + 1]]:
            trough.append(pointsIdx[i])
        else:
            rest.append(pointsIdx[i])
    new_peak = []
    new_trough = []
    for pointIdx in peak:
        new_peak.append(np.max([pointIdx - win, 0]) + np.argmax(y[np.max([pointIdx - win, 0]):np.min([pointIdx + win+1, length - 1])]))
    for pointIdx in trough:
        new_trough.append(np.max([pointIdx - win, 0]) + np.argmin(y[np.max([pointIdx - win, 0]):np.min([pointIdx + win+1, length - 1])]))
    return sorted(new_peak + new_trough + rest)


def curveFitting(x,y,win=2,w=1):
    '''
    fitting curve by a greedy approach to capture the trend with a piece wise line
    Args:
        x: list of xi
        y: list of yi
    return:
        x,fitted_y
    '''
    turningPointIdx = getTurningPoints(x,y,w)
    turningPointIdx = refineTurningPoints(x, y, turningPointIdx,win)
    fitted_y = []
    for i in range(1,len(turningPointIdx)):
        start = turningPointIdx[i-1]
        end = turningPointIdx[i]
        preds = (y[end] - y[start]) / (x[end] - x[start]) * (x[start:end] - x[start]) + y[start]
        for j in range(preds.shape[0]):
            fitted_y.append(preds[j])
    fitted_y.append(y[turningPointIdx[-1]])
    return np.asarray(x),np.asarray(fitted_y)






def fdr(target,decoy,alpha=0.05):
    '''
    perform FDR control to select true positive target samples at a user specific alpha. alpha=0.05 means 100 samples containing 5 decoy datapoints
    :param target: list of float scores from target dataset; the higher score, the better
    :param decoy: list of float scores from decoy dataset
    :param alpha: FDR alpha level, default 0.05
    :return:
        cutoff: minimum true positive scores
    '''
    eps = np.finfo(float).eps
    target = np.asarray(target)
    targetLabel = target * 0 + True
    decoy = np.asarray(decoy)
    decoyLabel = decoy * 0 + False
    val = np.concatenate([target, decoy])
    label = np.concatenate([targetLabel, decoyLabel])
    reverseArgSort = np.argsort(val)[::-1]
    val = val[reverseArgSort]
    label = label[reverseArgSort]

    numOfTarget = 0
    numOfDecoy = 0
    while numOfTarget + numOfDecoy == 0 or numOfDecoy / (numOfTarget+eps) <= alpha:
        idx = numOfTarget + numOfDecoy
        if idx>=len(val):
            break
        cutoff = val[idx]
        if label[idx]:
            numOfTarget += 1
        else:
            numOfDecoy += 1
    if numOfDecoy / (numOfTarget+eps) > alpha:
        cutoff -= eps
    return max([cutoff,0.15])


# @jit(nopython=True, parallel=True)
def distanceNormalization_by_mean(mat):
    mat=np.triu(mat,k=0)
    for i in range(1,mat.shape[0]):
        mat[:-i,i:][np.diag_indices(mat.shape[0]-i, ndim=2)]/=np.nanmean(mat[:-i,i:][np.diag_indices(mat.shape[0]-i, ndim=2)])
    # make the matrix symetric
    mat+=mat.transpose()
    mat[np.diag_indices(mat.shape[0], ndim=2)]/=np.nanmean(mat[np.diag_indices(mat.shape[0], ndim=2)])
    return mat




import pandas as pd
import sys
import pickle
import tqdm
@click.command()
@click.option('--resol', default=5000,type=int, help='resolution [5000]')
@click.option('--output',type=str,required=True, help='output')
def createdatabase(resol,output):
    database = {}
    for filename in tqdm.tqdm(sys.stdin):
        filename=filename.strip()
        database[filename] = {}
        f=pd.read_csv(filename,header=None,sep='\t')
        for chrom in set(f[0]):
            database[filename][chrom]=f[f[0]==chrom][[3,4,5,6]].to_numpy()
    with open(output, 'wb') as handle:
        pickle.dump({'data':database,'resol':5000}, handle, protocol=pickle.HIGHEST_PROTOCOL)