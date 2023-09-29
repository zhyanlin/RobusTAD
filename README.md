# RobusTAD: Reference panel based annotation of nested topologically associating domains

![RobusTAD](robusTAD_overview.png "RobusTAD")


We also provide examples with data for TAD and loop annotations under **example folder**.

You can find scripts and data to **reproduce our analysis** in the manuscript at 
https://doi.org/10.5281/zenodo.8306238 .

## software dependencies
RobusTAD is developed and tested on Linux machines and relies on the following packages:
<pre>appdirs==1.4.4
click==8.0.1
cooler==0.8.11
einops
h5py
importlib_resources>=5.4.0
matplotlib
numpy
pandas
requests
setuptools
tqdm</pre>
## Installation
RobusTAD relies on several libraries including numba, scipy, etc. 
We suggest users using conda to create a virtual environment for it (It should also work without using conda, i.e. with pip). You can run the command snippets below to install RobusTAD:
<pre>
git clone https://github.com/zhyanlin/RobusTAD.git
cd RobusTAD
conda create --name robustad  --file requirements.txt python=3.9
conda activate robustad
</pre>
Install RobusTAD:
<pre>pip install --editable .</pre>
If fail, please try `python setup build` and `python setup install` first.

The installation requires network access to download libraries. Usually, the installation will finish within 5 minutes. The installation time is longer if network access is slow and/or unstable.

## Initialization (configuration)
After RobusTAD installation, you need to initialize RobusTAD. It loads reference panel into your local disk. You can run the following command:
<pre>
robustad config init</pre>
Then you will be asked to select (1) download the default reference panel or (2) load your own panel. The default one (~3.2GB) is for hg38 at 5kb resoltuion containing 177 samples. The easiest way to run RobusTAD is to load the default panel.

You can also select option (2) to load your own panel.


<b>Our reference panels are for human data at 5kb resolution!</b>

## Step 1. Domain boundary annotation

RobusTAD provides two modes for single sample based domain boundary annotation: (i) Using <code>lmcc</code> if you have a reference panel (RobusTADs provides a reference for hg38); (ii) Using <code>boundary</code> if you don't have any reference panel.

### (i). reference panel enabled domain boundary annotation with <code>robustad lmcc</code>
<pre>
Usage: robustad lmcc [OPTIONS] COOLFILE PREFIX

  LMCC based boundary annotation [default mode]

Options:
  --mind INTEGER   min distance allowed between neighboring boundaries in kb [minW]
  --alpha FLOAT    alpha-significant in FDR [0.05]
  --minW INTEGER   min window size in kb [50]
  --maxW INTEGER   max window size in kb [600]
  --chr TEXT       comma separated chromosomes
  --ratio FLOAT    minRatio for comparision [1.2]
  --ds INTEGER     development only; please leave as default.
  --resol INTEGER  resol [5000]
  --help           Show this message and exit.
</pre>
This is the default boundary annotation method in RobusTAD. It performs two steps: 1) study sample based boundary annotation; 2) reference panel based boundary refinement.


|     Paramters     |                                  Detail                                 |
|:-------------:|:-----------------------------------------------------------------------:|
|   alpha   | significant level in FDR. It controls the number of false positives in domain boundary annotations. We usually set it as 0.05. We can increase it to 0.1 in anlyzing low coverage data (e.g., ~100M). A larger value (0.1) produces more false positive annotations.   names                                                        |
| minW, maxW| minimum and maximum windows used in boundary score calculation. minW and maxW should be similar to the smallest and largest TAD sizes. We suggest keep the values as default. Setting minW<50 leds to too many small TAD boundary predictions which seem to be FP. |
|mind | minimum distance between two boundary annotation of the same type (i.e., left or right). Two left/right boundary with a distance smaller than mind will be merged into one. |
|ratio | minimum ratio (>=1) for rank-sum test. It is the $gammma$ paramter in our manuscript. It controls the sharpness of predicted boundary. Larger values produce fewer but more sharp boundaries. |
|COOLFILE | input file in .mcool format |
| PREFIX | output prefix. The output file name is PREFIX.bed |


### (ii). single sample based domain boundary annotation with <code>robustad boundary</code>
<pre>
Usage: robustad boundary [OPTIONS] COOLFILE PREFIX

  Individual sample only boundary annotation

Options:
  --resol INTEGER  resolution [5000]
  --mind INTEGER   min distance allowed between neighboring boundaries in kb [minW]
  --minW INTEGER   min window size in kb [50]
  --maxW INTEGER   max window size in kb [600]
  --ratio FLOAT    minRatio for comparision [1.2]
  --alpha FLOAT    alpha-significant in FDR [0.05]
  --chr TEXT       comma separated chromosomes
  --help           Show this message and exit.
</pre>
This command will annotate domain boundaries from the study sample. It does not involve boundary refinement. You can use this command if you don't have a reference panel.

### output format
It contains tab separated fields as follows:
<pre>Chr    Start    End    leftScore    leftCall    rightScore    RightCall</pre>

|     Field     |                                  Detail                                 |
|:-------------:|:-----------------------------------------------------------------------:|
|   Chr   | chromosome names                                                        |
| Start | start genomic coordinate                                               |
|   End   | end genomic coordinates (i.e. End=Start+resol)                        |
|     leftScore/rightScore     | left and right domain boundary scores [-1,1]                                               |
|       leftCall/rightCall      | binary indicator for left and right boundary annotation. 1: a boundary; 0: not a boundary.                                                    |


## Step 2. Domain annotation
After annotating domain boundaries, you can pair left and right boundary into nested TADs with <code>robustad assembly</code> 
<pre>
Usage: robustad assembly [OPTIONS] COOLFILE BOUNDARYFILE PREFIX

Options:
  --resol INTEGER   resolution [5000]
  --mintad INTEGER  min TAD size [50000]
  --maxtad INTEGER  max TAD size [3000000]
  --ratio FLOAT     min ratio for comparision [1.2]
  --delta FLOAT     min score for forming a possible pair [0.1]
  --chr TEXT        comma separated chromosomes
  --help            Show this message and exit.
</pre>
|     Paramters     |                                  Detail                                 |
|:-------------:|:-----------------------------------------------------------------------:|
|   mintad, maxtad   | minimum and maximum TADs allowed. We suggest to set mintad not smaller than than minw in step 1.                                                       |
| delta| minimum score for forming a possible pair [0-1]. |
|ratio | minimum ratio (>=1) for rank-sum test. It is the $gammma$ paramter in our manuscript. It controls the sharpness of predicted boundary. Larger values produce fewer but more sharp boundaries. |
|COOLFILE | input file in .mcool format |
|BOUNDARYFILE | domain boundary file produced by step 1.
| PREFIX | output prefix. The output file name is PREFIX.bed |

**NB**: You will see a progress bar during running the assembly function. The time for analyzing each item differs, and the slowest items appear at the beginning. Thus, the total time displayed are much longer than the actual time required. Given the total number of items 500-2000, RobusTAD can finish the job 3~70 min in a desktop with 6 cores. 

### output format
It contains tab separated fields as follows:
<pre>Chr    Start    End	score	nestedScore level</pre>

|     Field     |                           Detail                                 |
|:-------------:|:-----------------------------------------------------------------------:|
|   Chr1   | chromosome names                                                        |
| Start1 | start genomic coordinates                                               |
|   End   | end genomic coordinate                        |
|     Score     | TAD score (with subTADs excluded) [0~1]                                               |
|       nestedScore      | TAD score without masking subTADs [0~1]. Avoid using this one. It is inflated by nested TADs.                                                   |
|level| TAD levels. 0: inner-most TADs|


## Sanity check
It is hard to evaluate TAD annotations in the real experiments as we don't have ground truth. You can use the plot option for a quick check
<pre>robustad plot COOLFILE.mcool TADFILE.bed chr17:1000000-2000000</pre> 
It will produce an image of annotated Hi-C sub-matrix for the region chr17:1000000-2000000.

## Advanced usage
### 1. How to create and use my own reference panel?
Let's assume you already have a group of Hi-C contact maps in .mcool format. 
To create your own reference panel, you need to:
1. Use robustad boundary to compute boundary scores for all samples in the reference panel.
2. Use this robustad createdatabase command to create a database saved as output.
3. provide the new reference panel to lmcc.

(optional) Load the new reference panel by run, <pre>refhic config init</pre>
select (2) load your own panel.

### 2. How to edit config file?
User We don't ask to edit config file manually, but you can still do it if you want: <pre>refhic config edit</pre>


## Citation
If you use RobusTAD in your work, please cite our paper:

Yanlin Zhang, Rola Dali, and Mathieu Blanchette. RobusTAD: Reference panel based annotation of nested topologically associating domains.

## Contact
A GitHub issue is preferable for all problems related to using RobusTAD. 

For other concerns, please email Yanlin Zhang or Mathieu Blanchette (yanlin.zhang2@mail.mcgill.ca, blanchem@cs.mcgill.ca).