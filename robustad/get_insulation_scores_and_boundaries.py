from cooltools.insulation import calculate_insulation_score
from cooltools.insulation import find_boundaries
import cooler
import bioframe
import click
import sys
import pandas as pd
import numpy as np

import logging
import warnings
# from util import quantile_normalize

logging.basicConfig(level=logging.INFO)



def insulationRatio(clr, window_bp, btype='both',chunksize_bp=20000000,chromosomes=None,verbose=True):
    """ Calculates insulation ratio score from a Hi-C contact map.
    
    Args:
        clr (cooler.Cooler): A cooler with balanced Hi-C data.
        btype: boundary type, left, right or both. default: both- calculates insulation ratio for
            both type of boundary.
    """

    if chromosomes is None:
        chromosomes = clr.chromnames

    bin_size = clr.info["bin-size"]

    window_bin = window_bp // bin_size
    bad_win_size = window_bp % bin_size != 0
    if bad_win_size:
        raise Exception(
            "The window sizes {} has to be a multiple of the bin size {}".format(
                window_bp, bin_size)
            )
    chunksize_bin=chunksize_bp // bin_size

    mask_inter = np.triu(np.ones((window_bin, window_bin)))
    np.fill_diagonal(mask_inter, 0)
    mask_inter[0, -1] = 0
    mask_intra = np.tril(np.ones((window_bin, window_bin)))
    np.fill_diagonal(mask_intra, 0)
    mask_intra[-1, 0] = 0


    insr_chrom_tables = []
    for chrom in chromosomes:
        if verbose:
            logging.info("Processing {}".format(chrom))

        chrom_bins = clr.bins().fetch(chrom)
        insr_chrom = chrom_bins[["chrom", "start", "end"]].copy()
        insr_chrom["is_bad_bin"] = chrom_bins["weight"].isnull()
        mat=clr.matrix(balance=True,sparse=True).fetch(chrom)
        mat=mat.todense()
        mat[np.isnan(mat)] = 0
        # reduce distance effect
        dim = mat.shape[0]
        target = np.linspace(0, dim-1, dim)
        for i in range(1, dim):
            diag = np.asmatrix(np.diagonal(mat, offset=i))
#             diag=np.append(diag,[[0]*i],axis=1)
            # diag = quantile_normalize(diag, axis=0, target=target).transpose()
            # diag = diag/np.mean(diag)#/(np.std(diag)+1e-10)
            diag = (diag-np.mean(diag))/(np.std(diag)+1e-12)
            mat[:-i, i:][np.diag_indices(dim - i)[0], np.diag_indices(dim - i)[1]] = diag#[:,:dim-i]
            mat[i:, :-i][np.diag_indices(dim - i)[0], np.diag_indices(dim - i)[1]] = diag#[:,:dim-i]

        mat -= np.min(mat)


        insr_left=[]
        insr_right=[]
        for i in range(mat.shape[0]):
            if btype=='both' or btype=='left':
                if i>window_bin and i+window_bin<mat.shape[0]:
                    ratio = np.nanmean(
                        mat[i + 1:i + 1 + window_bin, i + 1:i + 1 + window_bin] * mask_inter) * 1.0 \
                    / np.nanmean(mat[i - window_bin + 1:i + 1, i + 1:i + 1 + window_bin] * mask_intra)
                else:
                    ratio = np.nan
                insr_left.append(ratio)
            if btype=='both' or btyp=='right':
                if i>=window_bin and i+window_bin<mat.shape[0]:
                    ratio= np.nanmean(
                        mat[i - window_bin:i, i - window_bin:i] * mask_inter) * 1.0 \
                    / np.nanmean(mat[i - window_bin:i, i:i + window_bin] * mask_intra)
                else:
                    ratio=np.nan
                insr_right.append(ratio)

        if btype=='both' or btype=='left':
            insr_chrom["left_insulation_ratio_{}".format(window_bp)] = insr_left
        if btype=='both' or btype=='right':
            insr_chrom["right_insulation_ratio_{}".format(window_bp)] = insr_right


        insr_chrom_tables.append(insr_chrom)
        del mat

    insr_table = pd.concat(insr_chrom_tables)

    return insr_table

@click.command()
@click.argument('mcoolfile')
@click.argument('filename')
@click.option('--resolution', default=-1)
@click.option('--window', default=250000)
@click.option('--bweak', default=0)
@click.option('--bstrong', default=0.5)
@click.option('--cutoff', default=2)
@click.option('--pixels_frac', default=0.66)
def main(mcoolfile, filename, resolution, window, bweak, bstrong, cutoff, pixels_frac):
    cooler_path = str(mcoolfile) + '::/resolutions/' + str(resolution)
    c = cooler.Cooler(cooler_path)
    # calculate insulation score
    ins_table = calculate_insulation_score(c, window)

    # Find boundaries
    ins_table = find_boundaries(ins_table, pixels_frac, cutoff)


    # Classify the boundaries as strong and week
    # strong boundaries >= 0.5,  weak < 0.5
    boun_table = ins_table[ins_table[f'boundary_strength_{window}'] > bweak].copy()
    boun_table['boundary_classification'] = np.where(boun_table[f'boundary_strength_{window}'] >= bstrong, 'Strong', 'Weak')
    
    
    insr_table=insulationRatio(c, window, btype='both',chunksize_bp=20000000,chromosomes=None)

    boun_table=pd.merge(boun_table,insr_table,how='left',left_on=['chrom','start','end'],right_on=['chrom','start','end'])
    columns = ['chrom', 'start', 'end', 'boundary_classification', f'boundary_strength_{window}',f'left_insulation_ratio_{window}',f'right_insulation_ratio_{window}']
    boun_table=boun_table[(boun_table[f'left_insulation_ratio_{window}']>1) & (boun_table[f'right_insulation_ratio_{window}']>1) ]
    boun_table.to_csv(f'{filename}_insulation_score_boundaries.bed', sep='\t', header=True, index=False, columns=columns,na_rep='-')

if __name__ == "__main__":
    main()
