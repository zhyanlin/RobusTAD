import numpy as np
import numba
from numba import jit
from util import fdr
from scipy.signal import find_peaks
eps = np.finfo(float).eps
import scipy

@jit(nopython=True, parallel=True,error_model='python')
def rightBoundaryScores(data, minW, maxW, minRatio, zScore=True, weighted=True):
    s = data.shape[0]
    if minW>s//2-2:
        return np.zeros(s)
    if maxW>s//2-2:
        maxW=s//2-2
    diags = np.zeros((s, maxW+1))

    for i in numba.prange(s):
        for j in range(maxW+1):
            if i+j < s:
                diags[i, j] = data[i, i+j]
    # leftBoundaryScore = np.zeros(s)-np.inf
    rightBoundaryScore = np.zeros(s)-np.inf
    for i in numba.prange(maxW+2-1, s-maxW-2):
        rightScores = np.zeros(maxW+1)
        for diag in range(1, maxW + 1):
            crossIF = diags[i - diag + 1:i + 1, diag] + eps
            withinIF = diags[i - maxW:i - diag + 1, diag] + eps
            ratio = np.outer(withinIF, 1 / crossIF)
            score = (ratio>minRatio)-1*(ratio<1/minRatio)
            score = np.sum(score, axis=1)
            cumsum = np.cumsum(np.flip(score))

            if weighted:
                scale = (ratio>minRatio)+1*(ratio<1/minRatio)
                scale = np.sum(scale,axis=1)
                scale = np.cumsum(np.flip(scale))
                scale[scale==0] = 1
                rightScores[-cumsum.shape[0]:] += cumsum/scale
            else:
                rightScores[-cumsum.shape[0]:] += cumsum
        w = np.arange(minW, maxW + 1)
        if weighted:
            rightScores[minW:maxW + 1] /= w
        else:
            maxiPossible = (w ** 3 + 3 * (w ** 2) + 2 * w) / 6
            rightScores[minW:maxW + 1] /= (maxiPossible ** 0.5)

        rightBoundaryScore[i + 1] = np.mean(rightScores[minW:maxW+1])

    mean = np.mean(rightBoundaryScore[rightBoundaryScore > -np.inf])
    rightBoundaryScore[rightBoundaryScore == -np.inf] = mean
    if zScore:
        rightBoundaryScore = (rightBoundaryScore-np.mean(rightBoundaryScore))/np.std(rightBoundaryScore)
    return rightBoundaryScore



def leftBoundaryScores(data, minW, maxW, minRatio,zScore=True,weighted=True):
    data = np.flip(data)
    leftBoundaryScore = np.flip(rightBoundaryScores(data,minW,maxW,minRatio,zScore,weighted))
    return leftBoundaryScore


def leftBoundaryCaller(mat, minW, maxW, minRatio=1.1, weighted=True, num_decoys=1, alpha=0.05,distance=5):
    zScore=False
    scores = leftBoundaryScores(mat, minW, maxW, minRatio, weighted=weighted, zScore=zScore)
    targetPeaks, targetPeakProperties = find_peaks(scores, height=0, distance=distance,plateau_size=0)
    targetPeakHeights=targetPeakProperties['peak_heights']
    heightCutoffs=[]
    for i in range(num_decoys):
        decoyMat = mat.copy()
        for diagIdx in range(decoyMat.shape[0]):
            np.random.shuffle(np.diag(decoyMat, diagIdx))
        decoyMat[np.tril_indices(decoyMat.shape[0])] = 0
        decoyMat = decoyMat + decoyMat.transpose()
        decoyScores = leftBoundaryScores(decoyMat, minW, maxW, minRatio, weighted=weighted, zScore=zScore)
        del decoyMat
        _, decoyPeakProperties = find_peaks(decoyScores, height=0, distance=distance)
        decoyPeakHeights=decoyPeakProperties['peak_heights']
        heightCutoff = fdr(targetPeakHeights, decoyPeakHeights, alpha=alpha)
        heightCutoffs.append(heightCutoff)
    heightCutoff=np.mean(heightCutoffs)
    peaks = np.zeros(scores.shape)
    for j in range(len(targetPeaks)):
        if targetPeakProperties['peak_heights'][j] >= heightCutoff:
            peaks[targetPeaks[j]] = 1
    return scores, peaks


def rightBoundaryCaller(mat, minW, maxW, minRatio=1.1, weighted=True, num_decoys=1, alpha=0.05,distance=5):
    zScore=False
    scores = rightBoundaryScores(mat, minW, maxW, minRatio, weighted=weighted, zScore=zScore)
    targetPeaks, targetPeakProperties = find_peaks(scores, height=0, distance=distance,plateau_size=0)
    targetPeakHeights=targetPeakProperties['peak_heights']
    heightCutoffs=[]
    for i in range(num_decoys):
        decoyMat = mat.copy()
        for diagIdx in range(decoyMat.shape[0]):
            np.random.shuffle(np.diag(decoyMat, diagIdx))
        decoyMat[np.tril_indices(decoyMat.shape[0])] = 0
        decoyMat = decoyMat + decoyMat.transpose()
        decoyScores = rightBoundaryScores(decoyMat, minW, maxW, minRatio, weighted=weighted, zScore=zScore)
        del decoyMat
        _, decoyPeakProperties = find_peaks(decoyScores, height=0, distance=distance)
        decoyPeakHeights=decoyPeakProperties['peak_heights']
        heightCutoff = fdr(targetPeakHeights, decoyPeakHeights, alpha=alpha)
        heightCutoffs.append(heightCutoff)
    heightCutoff=np.mean(heightCutoffs)
    peaks = np.zeros(scores.shape)
    for j in range(len(targetPeaks)):
        if targetPeakProperties['peak_heights'][j] >= heightCutoff:
            peaks[targetPeaks[j]] = 1
    return scores, peaks
