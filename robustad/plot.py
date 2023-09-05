import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cooler,sys
import numpy as np
import click
import pandas as pd
from robustad.util import distanceNormalization_by_mean
from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('wr',["w", "r"], N=256)

@click.command()
@click.option('--resol', default=10000,type=int, help='resolution')
@click.option('--posonly', default=False,type=bool, help='only first three col')
@click.argument('coolfile', type=str, default=None, required=True)
@click.argument('tadfile', type=str, default=None, required=True)
@click.argument('region', type=str, default=None, required=True)
def plot(region,resol,coolfile,tadfile,posonly):
    try:
        c = cooler.Cooler(coolfile+'::/resolutions/'+str(resol))
    except:
        c = cooler.Cooler(coolfile)
    if 'bin-size' in c.info and c.info['bin-size']!=resol:
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
    # mat = distanceNormalization_by_mean(mat)
    tadfile=pd.read_csv(tadfile,sep='\t',header=None)
    tadfile=tadfile[(tadfile[0]==chr) & (tadfile[1]>=start) & (tadfile[2]<end)].reset_index(drop=True)
    if posonly:
        scores = tadfile[1].to_numpy()*0+1
    else:
        scores=tadfile[3].to_numpy()
    TADs = (tadfile[[1,2]].to_numpy()-start)//resol

    print(scores)
    fig = plt.figure()
    ax = fig.add_subplot()

    plt.subplots_adjust(bottom=0.25)
    slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    tad_slider = Slider(
        ax=slider,
        label='TAD strength',
        valmin=np.min(scores),
        valmax=np.max(scores),
        valinit=np.median(scores),
    )

    ax.imshow(mat,cmap=cmap,vmax=np.nanmean(np.diag(mat,50)))
    recs=[]
    for i,j in TADs:

        recs.append(patches.Rectangle((i, j),j-i+1,i-j-1,fill=False))
        ax.add_patch(recs[-1])


    def update(val):
        for i in range(len(recs)):
            if scores[i]>=tad_slider.val:
                recs[i].set_visible(True)
            else:
                recs[i].set_visible(False)
        fig.canvas.draw_idle()

    tad_slider.on_changed(update)
    plt.show()
