import numpy as np
import cooler
import click
import sys
from tqdm import tqdm
import pandas as pd
from robustad.calcTADBoundaryScore import leftBoundaryCaller,rightBoundaryCaller
boundaryCaller={'left':leftBoundaryCaller,'right':rightBoundaryCaller}


@click.command()
@click.option('--resol', default=5000, help='resolution [5000]')
@click.option('--mind', type=int, help='min distance allowed between neighboring boundaries in kb [minW]')
@click.option('--minW', default=50,type=int, help='min window size in kb [50]')
@click.option('--maxW', default=600, type=int,help='max window size in kb [600]')
@click.option('--ratio',default=1.2,type=float,help ='minRatio for comparision [1.2]')
@click.option('--alpha',default=0.05, type=float,help='alpha-significant in FDR [0.05]')
@click.option('--chr',default=None,help='comma separated chromosomes')
@click.argument('coolfile',type=str,default = None,required=True)
@click.argument('prefix',type=str,default = None, required=True)
def boundary(coolfile,prefix,resol,minw,maxw,alpha,ratio,chr,mind):
    '''Individual sample only boundary annotation'''

    offset ={'left':1 * resol, 'right':-1 * resol}

    minw=minw*1000
    maxw=maxw*1000
    if not mind:
        mind = minw
    else:
        mind = mind*1000

    if cooler.fileops.is_cooler(coolfile):
        c = cooler.Cooler(coolfile)
        if c.info['bin-size']!=resol:
            print('contact map at '+str(resol)+' does not exist!\nGood bye!')
            sys.exit(0)
    else:
        try:
            c = cooler.Cooler(coolfile + '::/resolutions/' + str(resol))
        except:
            print('contact map at '+str(resol)+' does not exist!\nGood bye!')
            sys.exit(0)

    print("analysis at ",resol, 'resolution')

    if chr is None:
        chromnames=c.chromnames
    else:
        chromnames=chr.split(',')
    # print(chromnames)



    results={}

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
    left.merge(right, on=[0, 1, 2]).to_csv(prefix+'.bed', index=False, header=False, float_format='%.4f',sep='\t')


if __name__ =='__main__':
    boundary()