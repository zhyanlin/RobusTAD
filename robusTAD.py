import numpy as np
import cooler
import click
from tqdm import tqdm
import pandas as pd
from scipy.sparse import csr_matrix
from calcTADBoundaryScore import leftBoundaryCaller,rightBoundaryCaller
boundaryCaller={'left':leftBoundaryCaller,'right':rightBoundaryCaller}



@click.command()
@click.option('--resol', default=5000, help='resolution')
@click.option('--minibatch', default=False, type=bool, help='low')
@click.option('--minW', default=50,type=int, help='min window size [kb]')
@click.option('--maxW', default=250, type=int,help='max window size [kb]')
@click.option('--ratio',default=1.1,type=float,help ='minRatio for comparision')
@click.option('--alpha',default=0.05, type=float,help='alpha-significant in FDR')
@click.option('--chr',default=None,help='comma separated chromosomes')
@click.option('--tsv',default=False,type=bool,help='tsv input: chrom locusA locusB IF [False]')
@click.argument('hicfile',type=str,default = None,required=True)
@click.argument('prefix',type=str,default = None, required=True)
def robusTAD(hicfile,prefix,resol,minw,maxw,alpha,ratio,minibatch,chr,tsv):
    offset ={'left':1 * resol, 'right':-1 * resol}
    minw=minw*1000
    maxw=maxw*1000
    regions = []
    if tsv:
        c = pd.read_csv(hicfile,sep='\t',header=None)
        c[[1,2]]//=resol
        if chr is None:
            regions = list(set(c[0]))
        else:
            regions=chr.split(',')
    else:
        c = cooler.Cooler(hicfile + '::/resolutions/' + str(resol))
        if chr is None:
            chromnames=c.chromnames
        else:
            chromnames=chr.split(',')

        for chr in chromnames:
            bin = c.bins()[['chrom', 'start', 'end']].fetch(chr)
            bin = np.asarray(bin)
            N = bin.shape[0]
            if minibatch:
                maxN=25000
            else:
                maxN=int(1e9)
            for i in range(0, N, maxN):
                region = chr + ':' + str(bin[i, 1]) + '-' + str(bin[np.min([i + maxN, N]) - 1, 2])
                regions.append(region)

    results={}
    
    for region in tqdm(regions):
        if tsv:
            dfChr = c[c[0]==region].reset_index(drop=True)
            N = max([np.max(dfChr[1]),np.max(dfChr[2])])+1
            mat=csr_matrix((dfChr[3], (dfChr[1], dfChr[2])), shape=(N,N),dtype=float)
            chrf = np.asarray([region for i in range(N)])
            startf = np.asarray([i*resol for i in range(N)],dtype=int)
            endf = np.asarray([i*resol+resol for i in range(N)],dtype=int)
            bins = pd.DataFrame(data=np.array([chrf,startf,endf]).transpose()).astype({1: 'int32',2: 'int32'})
            bins=np.asarray(bins)

        else:
            mat = c.matrix(balance=True,sparse=True).fetch(region)
            bins = np.asarray(c.bins()[['chrom', 'start', 'end']].fetch(region))
        if mat.nnz==0:
            continue
        mat = mat.todense()
        mat[np.isnan(mat)] = 0


        for key in boundaryCaller:
            scores, calledPeaks = boundaryCaller[key](mat, int(minw / resol), int(maxw / resol), ratio, weighted=True, alpha=alpha)
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
    left.merge(right, on=[0, 1, 2]).to_csv(prefix+'.bed', index=False, header=False, float_format='%.4f',sep='\t')


if __name__ =='__main__':
    robusTAD()
