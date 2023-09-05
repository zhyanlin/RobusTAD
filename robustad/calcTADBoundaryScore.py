import numpy as np
import numba
from numba import jit
from robustad.util import fdr,nansum
from scipy.signal import find_peaks
eps = np.finfo(float).eps
from tqdm import tqdm

@jit(nopython=True, parallel=True,error_model='python')
def rightBoundaryScoresNB(diags,s, minW, maxW, minRatio=1.1):
    rightBoundaryScore = np.zeros(s)-np.inf
    for i in numba.prange(maxW+2-1, s-maxW-2):
        rightScores = np.zeros(maxW+1)
        for diag in range(1, maxW + 1):
            crossIF = diags[i - diag + 1:i + 1, diag] + eps
            withinIF = diags[i - maxW:i - diag + 1, diag] + eps
            ratio = np.outer(withinIF, 1 / crossIF)

            win = ratio>minRatio
            loss = ratio<1/minRatio
            # loss = ratio <minRatio

            score = win-1*loss
            score = nansum(score, axis=1)
            cumsum = np.nancumsum(np.flip(score))

            scale = ratio>-1#win+loss
            scale = scale*0+1
            scale = nansum(scale,axis=1)#+eps
            # print(scale)
            scale = np.nancumsum(np.flip(scale))
            # scale[scale==0] = 1
            rightScores[-cumsum.shape[0]:] += cumsum/scale

        w = np.arange(minW, maxW + 1)
        rightScores[minW:maxW + 1] /= w
        rightBoundaryScore[i + 1] = np.max(rightScores[minW:maxW+1])
        #rightBoundaryScore[i + 1] = np.mean(rightScores[minW:maxW + 1])

    mean = np.mean(rightBoundaryScore[rightBoundaryScore > -np.inf])
    rightBoundaryScore[rightBoundaryScore == -np.inf] = mean
    return rightBoundaryScore


def rightBoundaryScores(data, minW, maxW, minRatio=1.1,decoy=False):
    s = data.shape[0]
    diags = np.zeros((s+maxW*2, maxW + 1)) * np.nan
    for j in range(min([maxW+1,s])):
        diagj =data.diagonal(j)
        if decoy:
            np.random.shuffle(diagj)
        diags[maxW:len(diagj)+maxW, j] = diagj
    rightBoundaryScore=rightBoundaryScoresNB(diags,s+maxW*2, minW, maxW, minRatio)
    return rightBoundaryScore[maxW:-maxW]

def leftBoundaryScores(data, minW, maxW, minRatio=1.1,decoy=False):
    data = np.flip(data)
    leftBoundaryScore = np.flip(rightBoundaryScores(data,minW,maxW,minRatio,decoy))
    return leftBoundaryScore


def leftBoundaryCaller(mat, minW, maxW, minRatio=1.1, num_decoys=1, alpha=0.05,distance=5):
    # print('distance',distance)

    scores = leftBoundaryScores(mat, minW, maxW, minRatio)
    targetPeaks, targetPeakProperties = find_peaks(scores, height=0, distance=distance,plateau_size=0)
    targetPeakHeights=targetPeakProperties['peak_heights']
    heightCutoffs=[]
    for i in range(num_decoys):
        decoyScores = leftBoundaryScores(mat, minW, maxW, minRatio,decoy=True)
        _, decoyPeakProperties = find_peaks(decoyScores, height=0, distance=distance)
        decoyPeakHeights=decoyPeakProperties['peak_heights']
        heightCutoff = fdr(targetPeakHeights, decoyPeakHeights, alpha=alpha)
        heightCutoffs.append(heightCutoff)
    heightCutoff=np.mean(heightCutoffs)
    peaks = np.zeros(scores.shape)


    for j in range(len(targetPeaks)):
        if targetPeakProperties['peak_heights'][j] >= heightCutoff:
            # if np.sum(scores[targetPeaks[j]]>scores[targetPeaks[j]-distance:targetPeaks[j]+distance])+2> distance*2: # this helps a little bit to remove FP, but not nessary
                peaks[targetPeaks[j]] = 1


    return scores, peaks


def rightBoundaryCaller(mat, minW, maxW, minRatio=1.1, num_decoys=1, alpha=0.05,distance=5):

    scores = rightBoundaryScores(mat, minW, maxW, minRatio)
    targetPeaks, targetPeakProperties = find_peaks(scores, height=0, distance=distance,plateau_size=0)
    targetPeakHeights=targetPeakProperties['peak_heights']
    heightCutoffs=[]
    for i in range(num_decoys):
        decoyScores = rightBoundaryScores(mat, minW, maxW, minRatio,decoy=True)
        _, decoyPeakProperties = find_peaks(decoyScores, height=0, distance=distance)
        decoyPeakHeights=decoyPeakProperties['peak_heights']
        heightCutoff = fdr(targetPeakHeights, decoyPeakHeights, alpha=alpha)
        heightCutoffs.append(heightCutoff)
    heightCutoff=np.mean(heightCutoffs)
    peaks = np.zeros(scores.shape)
    for j in range(len(targetPeaks)):
        if targetPeakProperties['peak_heights'][j] >= heightCutoff:
            # if np.sum(scores[targetPeaks[j]]>scores[targetPeaks[j]-distance:targetPeaks[j]+distance])+2> distance*2: # this helps a little bit to remove FP, but not nessary
                peaks[targetPeaks[j]] = 1
    # from matplotlib import pylab as plt
    # plt.figure()
    # plt.plot(scores, label='target')
    # plt.plot(decoyScores,label='decoy')
    # plt.legend()
    # plt.show()


    return scores, peaks
