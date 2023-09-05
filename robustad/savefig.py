import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cooler,sys
import numpy as np
import click
import pandas as pd
from  matplotlib.lines import Line2D
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('wr',["w", "r"], N=256)


def diffScore(a, b):
    scores = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i][1] < b[j][0] or a[i][0] > b[j][1]:
                scores[i, j] = 1
            ai = set(np.arange(a[i][0], a[i][1] + 1))
            bj = set(np.arange(b[j][0], b[j][1] + 1))
            scores[i, j] = 1 - len(ai.intersection(bj)) / len(ai.union(bj))
    return np.concatenate([np.min(scores, axis=1), np.min(scores, axis=0)]).mean()


def diffScores(A, B):
    distance = np.zeros(len(A))
    for i in range(len(A)):
        if len(A[i]) + len(B[i]) == 0:
            distance[i] = 0
        elif len(A[i]) + len(B[i]) > 0 and len(A[i]) * len(B[i]) == 0:
            distance[i] = 1
        else:
            distance[i] = diffScore(A[i], B[i])
    return distance

# def diffCal(f1,f2):
#     A = np.zeros((1 + np.max([np.max(f1[[1, 2]].to_numpy()), np.max(f2[[1, 2]].to_numpy())]),
#                   1 + np.max([np.max(f1[4]), np.max(f2[4])])))
#     B = np.zeros((1 + np.max([np.max(f1[[1, 2]].to_numpy()), np.max(f2[[1, 2]].to_numpy())]),
#                   1 + np.max([np.max(f1[4]), np.max(f2[4])])))
#     for l, r, level in f1[[1, 2, 4]].to_numpy():
#         A[l:r, level] = 1
#     for l, r, level in f2[[1, 2, 4]].to_numpy():
#         B[l:r, level] = 1
#     return np.sum(np.abs(A - B), axis=1)

def diffCal(A,B):
    maxij=np.max([np.max(A[[1,2]]),np.max(B[[1,2]])])+1
    Atads=[[] for i in range(maxij)]
    Btads=[[] for i in range(maxij)]
    for i, j in A[[1,2]].to_numpy():
        for k in range(i,j+1):
            Atads[k].append((i,j))
    for i, j in B[[1,2]].to_numpy():
        for k in range(i,j+1):
            Btads[k].append((i,j))
    return diffScores(Atads,Btads)


@click.command()
@click.option('--resol', default=10000,type=int, help='resolution')
@click.option('--posonly', default=False,type=bool, help='only first three col')
@click.argument('tool', type=str, default=None, required=True)
@click.argument('coolfile', type=str, default=None, required=True)
@click.argument('tadfile', type=str, default=None, required=True)
@click.argument('tool2', type=str, default=None, required=True)
@click.argument('coolfile2', type=str, default=None, required=True)
@click.argument('tadfile2', type=str, default=None, required=True)
@click.argument('region', type=str, default=None, required=True)
def savefig(region,resol,coolfile,tadfile,posonly,coolfile2,tadfile2,tool,tool2):
    try:
        c = cooler.Cooler(coolfile+'::/resolutions/'+str(resol))
    except:
        c = cooler.Cooler(coolfile)
    if 'bin-size' in c.info and c.info['bin-size']!=resol:
        print('contact map at '+str(resol)+' does not exist!\nGood bye!')
        sys.exit(0)

    try:
        c2 = cooler.Cooler(coolfile2+'::/resolutions/'+str(resol))
    except:
        c2 = cooler.Cooler(coolfile2)
    if 'bin-size' in c2.info and c2.info['bin-size']!=resol:
        print('contact map at '+str(resol)+' does not exist!\nGood bye!')
        sys.exit(0)

    if ':' in region:
        chr,_bp=region.split(':')
        start=int(_bp.split('-')[0])
        end = int(_bp.split('-')[1])
    else:
        chr=region
        start=0
        end=np.inf
    mat = c.matrix(balance=True,sparse=False).fetch(region)
    mat2 = c2.matrix(balance=True, sparse=False).fetch(region)
    tadfile=pd.read_csv(tadfile,sep='\t',header=None)
    tadfile=tadfile[(tadfile[0]==chr) & (tadfile[1]>=start) & (tadfile[2]<end)].reset_index(drop=True)

    tadfile2=pd.read_csv(tadfile2,sep='\t',header=None)
    tadfile2=tadfile2[(tadfile2[0]==chr) & (tadfile2[1]>=start) & (tadfile2[2]<end)].reset_index(drop=True)
    if posonly:
        scores = tadfile[1].to_numpy()*0+1
        scores2 = tadfile2[1].to_numpy() * 0 + 1
        tadfile[4] = 1
        tadfile2[4] = 1
    else:
        scores=tadfile[3].to_numpy()
        scores2 = tadfile2[3].to_numpy()
    tadfile[[1,2]] = (tadfile[[1,2]]-start)//resol
    tadfile2[[1,2]] = (tadfile2[[1, 2]] - start) // resol

    TADs = tadfile[[1, 2]].to_numpy()
    TADs2 = tadfile2[[1, 2]].to_numpy()

    diffScore=diffCal(tadfile,tadfile2)


    fig = plt.figure()

    ax=plt.gca()


    mat = np.triu(mat,1)+np.tril(mat2,-1)
    size = mat.shape[0]
    ax.imshow(mat,cmap=cmap,vmax=np.nanmean(np.diag(mat,5)))
    recs=[]
    for i,j in TADs:
        recs.append(Line2D([i,j,j],[i,i,j],color='black',alpha=0.5))
        ax.add_patch(recs[-1])

    recs2 = []
    for i, j in TADs2:
        recs2.append(Line2D([i, i, j],[i, j, j],  color='green',alpha=0.5))
        ax.add_patch(recs2[-1])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.text(size-90, 20, tool, fontsize=12)
    plt.text(10, size-15, tool2, fontsize=12)
    plt.savefig(tool+"_"+tool2+"_"+region+".png", bbox_inches='tight')


    # plt.show()
