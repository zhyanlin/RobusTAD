import numpy as np
import cooler,random
import click
from tqdm import tqdm
import pandas as pd
import pickle
from robustad.calcTADBoundaryScore import leftBoundaryCaller,rightBoundaryCaller
boundaryCaller={'left':leftBoundaryCaller,'right':rightBoundaryCaller}

from scipy.signal import find_peaks


def fetchLMCC(chrom, target, database, btype, win=5):
    LMCCs = {}
    if btype == 'right':
        scoreCol = -2
        tadCol = -1
    else:
        scoreCol = 0
        tadCol = 1
    for i in target:
        LMCCs[i] = []
        for sample in database:
            if chrom in database[sample]:
                data = database[sample][chrom]
                if data[i - win:i + win + 1, tadCol].sum() == 1:
                    LMCCs[i].append((sample, data[i - win:i + win + 1, scoreCol]))
    return LMCCs




@click.command()
@click.option('--mind', type=int, help='min distance allowed between neighboring boundaries in kb [minW]')
@click.option('--alpha',default=0.05, type=float,help='alpha-significant in FDR [0.05]')
@click.option('--minW', default=50,type=int, help='min window size in kb [50]')
@click.option('--maxW', default=600, type=int,help='max window size in kb [600]')
@click.option('--chr',default=None,help='comma separated chromosomes')
@click.argument('coolfile',type=str,default = None,required=True)
@click.option('--ratio',default=1.2,type=float,help ='minRatio for comparision [1.2]')
@click.argument('prefix',type=str,default = None, required=True)
@click.option('--ds',type=int,default = -1, required=False, help="development only; please leave as default.")
@click.option('--resol',type=int,default=5000,help='resol [5000]')
def lmcc(coolfile,prefix,alpha,chr,ratio,resol,minw,maxw,ds,mind):
    '''Detect left and right boundaries from Hi-C contact map with LMCC approaches'''

    if resol==5000:
        # with open('/home/yanlin/workspace/PhD/robusTAD/src/robustadv2/experiment/rerunSep1/hg38_5kb_50_600.pkl', 'rb') as handle:#
        with open('/home/yanlin/workspace/PhD/nextProject/2022_revisit/hg38Dtabase_5kb.pkl', 'rb') as handle:
            database = pickle.load(handle)
                
    elif resol == 10000:
        with open('/home/yanlin/workspace/PhD/nextProject/2022_revisit/hg38Dtabase_10kb.pkl', 'rb') as handle:
            database = pickle.load(handle)
    
    if ds>-1:
        samples=[]
        for key in database['data']:
            samples.append(key)
        random.Random(42).shuffle(samples)
        samples=samples[:ds]
        tmpdatabase = {}
        for key in samples:
            tmpdatabase[key] = database['data'][key]
        database['data']=tmpdatabase


    # resol = database['resol']
    minw *= 1000
    maxw *= 1000
    # ratio = database['ratio']
    if not mind:
        mind = minw
    else:
        mind = mind*1000

    offset ={'left':1 * resol, 'right':-1 * resol}

    c = cooler.Cooler(coolfile + '::/resolutions/' + str(resol))
    if chr is None:
        chromnames=c.chromnames
    else:
        chromnames=chr.split(',')



    results={}

    print('initial call with robusTAD')
    for chr in tqdm(chromnames):
        mat = c.matrix(balance=True,sparse=True).fetch(chr).tocsr()
        if mat.nnz<1000:
            continue

        bins = np.asarray(c.bins()[['chrom', 'start', 'end']].fetch(chr))
        for key in boundaryCaller:
            scores, calledPeaks = boundaryCaller[key](mat, int(minw / resol), int(maxw / resol), ratio, alpha=alpha,distance=mind//resol)
            scores=scores.reshape(-1,1)
            calledPeaks=calledPeaks.reshape(-1,1)
            _bins = bins.copy()
            _bins[:,1]+=offset[key]
            _bins[:,2]+=offset[key]
            result=np.hstack([_bins,scores,calledPeaks])
            if key not in results:
                results[key]=result
            else:
                results[key]=np.vstack([results[key],result])

    left=pd.DataFrame(results['left'])
    right = pd.DataFrame(results['right'])
    # left.merge(right, on=[0, 1, 2])[[0,1,2,'3_x','3_y']].to_csv(prefix+'.bed', index=False, header=False, float_format='%.4f',sep='\t')
    # left.merge(right, on=[0, 1, 2]).to_csv(prefix+'.bed', index=False, header=False, float_format='%.4f',sep='\t')
    result = left.merge(right, on=[0, 1, 2])

    win = 5
    newResults = []
    print('refine with LMCC')
    for chrom in tqdm(set(result[0])):
        chrdata = result[result[0] == chrom].to_numpy()
        # update right boundary call
        lmccs = fetchLMCC(chrom, np.argwhere(chrdata[:, -1] == 1).flatten(), database['data'], 'right', win=win)
        for key in lmccs:
            scores = chrdata[key - win:key + win + 1, -2]
            for i in range(len(lmccs[key])):
                scores = scores + lmccs[key][i][-1]
            idx, score = find_peaks(scores, height=-1)

            if len(idx) > 0:
                peak = idx[np.argmax(score['peak_heights'])]
                peak = peak - win + key
            else:
                peak = key
            chrdata[key, -1] = 0
            chrdata[peak, -1] = 1

        # update left boundary call
        lmccs = fetchLMCC(chrom, np.argwhere(chrdata[:, -3] == 1).flatten(), database['data'], 'left', win=win)
        for key in lmccs:
            scores = chrdata[key - win:key + win + 1, -4]
            for i in range(len(lmccs[key])):
                scores = scores + lmccs[key][i][-1]
            idx, score = find_peaks(scores, height=-1)

            if len(idx) > 0:
                peak = idx[np.argmax(score['peak_heights'])]
                peak = peak - win + key
            else:
                peak = key
            chrdata[key, -3] = 0
            chrdata[peak, -3] = 1


        newResults.append(chrdata)

    np.savetxt(prefix + '.bed', np.concatenate(newResults), fmt='%s', delimiter='\t')


if __name__ =='__main__':
    lmcc()