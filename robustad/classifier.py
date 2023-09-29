import numpy as np
import cooler,random
import click
import pandas as pd
@click.command()



@click.argument('input',type=str,default = None, required=True)
@click.argument('output',type=str,default = None, required=True)
@click.option('--resol',type=int,default=5000,help='resol [5000]')
def classifier(resol,input,output):
    input = pd.read_csv(input,sep='\t',header=None)
    chroms = set(input[0])
    pds = []
    for chrom in chroms:        
        leftBoundaryTimes = {}
        rightBoundaryTimes = {}
        chrInput=input[input[0]==chrom].reset_index(drop=True) 
        chrInput[[1,2]] = chrInput[[1,2]]//resol
        chrInput['leftBoundaryTimes'] = 0
        chrInput['rightBoundaryTimes'] = 0
        chrInput['maxbinTimes'] = -1
        binTimes =np.zeros(np.max(chrInput[[1,2]].values)+1)

        for i in range(len(chrInput)):
            leftB=chrInput.iloc[i, 1]
            rightB=chrInput.iloc[i, 2]
            binTimes[leftB:rightB+1]=binTimes[leftB:rightB+1]+1
            if leftB not in leftBoundaryTimes:
                leftBoundaryTimes[leftB]=1
            else:
                leftBoundaryTimes[leftB]+=1
            if rightB not in rightBoundaryTimes:
                rightBoundaryTimes[rightB]=1
            else:
                rightBoundaryTimes[rightB]+=1
        for boundary in leftBoundaryTimes:
            chrInput.loc[chrInput[1]==boundary,'leftBoundaryTimes']=leftBoundaryTimes[boundary]
        for boundary in rightBoundaryTimes:
            chrInput.loc[chrInput[2]==boundary,'rightBoundaryTimes']=rightBoundaryTimes[boundary]
        maxbinTimes=[]
        for i in range(len(chrInput)):
            leftB=chrInput.iloc[i, 1]
            rightB=chrInput.iloc[i, 2]
            maxbinTimes.append(np.max(binTimes[leftB:rightB+1]))
        chrInput['maxbinTimes'] = maxbinTimes
        largestTAD=[]
        largestTADLB = []
        largestTADRB = []
        for l1,r1 in zip(list(chrInput[1]),list(chrInput[2])):
            islargest=1
            for l2,r2 in zip(list(chrInput[1]),list(chrInput[2])):
                if l1==l2 and r1==r2:
                    pass
                elif l1>=l2 and r1<=r2:
                    islargest=0
            largestTAD.append(islargest)
            if islargest:
                largestTADLB.append(l1)
                largestTADRB.append(r1)
        
        largestTADLB = set(largestTADLB)
        largestTADRB = set(largestTADRB)
        islargestL = []
        islargestR = []
        for l,r in zip(list(chrInput[1]),list(chrInput[2])):
            if l in largestTADLB:
                islargestL.append(1)
            else:
                islargestL.append(0)
            if r in largestTADRB:
                islargestR.append(1)
            else:
                islargestR.append(0)
        chrInput['islargest'] = largestTAD
        chrInput['islargestLB'] = islargestL
        chrInput['islargestRB'] = islargestR

        chrInput[[1,2]] = chrInput[[1,2]]*resol
        pds.append(chrInput)
    pd.concat(pds).to_csv(output,sep='\t',index=False,header=True)
