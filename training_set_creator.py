import numpy as np
from metal_patch_selector import *
from random import shuffle

class TrainingSetCreator:

    def __init__(self, names, raw_data, regions, patch_dim, target_dict,getrotated=False
                        , step=1, transformdata=False, subtractmean=False, savepatches=False):
        '''
        names: list of strings which are the names of the datasets (to enforce an order)
        raw_data: dictionary with names as keys and images as values
        regions: dictionary of tuples (x1,y1,x2,y2) which is the rectangle selection
        patch_dim: a integer which represents the square side dimension
        target_dict: dictionary with names as keys and their classification number as value
        getrotated:a flag that activates the artificial increase of data patches by rotating existing patches
        step: a integer that indicates how many pixels away should the next patch be
        '''
        self.data=raw_data
        self.names=names
        self.regions=regions
        self.patch_dim=patch_dim
        self.target_dict=target_dict
        self.step=step
        self.nPatch=[None]*(len(self.names))
        self.transformdata=transformdata
        self.getrotated=getrotated
        if(self.transformdata):
            def f(x):
                return x
            self.transform=np.vectorize(f,otypes=[np.float32])
        self.subtractmean=subtractmean
        self.savepatches=savepatches
        self.patchdict={}
        #self.max=100.0

        self.compileIndexAccumulator()
    
    def getNumberOfAccPatches(self,n):
        '''
        Returns the total number of patches for the first n datasets
        '''
        #starts from 0
        count=0
        if n>1 and len(self.indacc)>=(n-1):
            i=n-1
            count=self.indacc[n-2]
        else:
            i=0
        #for every dataset
        for key in self.names:
            #but only for the first n
            if i<n:
                i+=1
            else:
                break
            #get the region to be taken
            x1,y1,x2,y2=self.regions[key]
            #rectangle = self.data[key][x1:x2, y1:y2]
            #total number of rows in the rectangle
            nRows=x2-x1+1
            #total number of rows of patches
            nRowsPatch=nRows-(self.patch_dim-1)
            nRowsPatch=nRowsPatch if self.step==1 else (nRowsPatch//self.step)+(1 if nRowsPatch%self.step>0 else 0)
            #total number of columns in the rectangle
            nCols=y2-y1+1
            #total number of columns of patches
            nColsPatch=nCols-(self.patch_dim-1)
            nColsPatch=nColsPatch if self.step==1 else (nColsPatch//self.step)+(1 if nColsPatch%self.step>0 else 0)
            #add it to the count
            count+=nRowsPatch*nColsPatch
            #count+=(rectangle.shape[0]-self.patch_dim+1)*(rectangle.shape[1]-self.patch_dim+1)
        return count
    
    def compileIndexAccumulator(self):
        '''
        Save in self.indacc[i] the total number of patches for the first i+1 datasets for every i
        '''
        self.indacc=[]
        #for every dataset
        for i in range(len(self.names)):
            self.indacc.append(self.getNumberOfAccPatches(i+1))

    '''
    def patchBinarySearch(self,i,start,end):
        
        Currently not used, not even tested
        
        if(start>=end):
            return start
        if(start+1==end):
            return start if i<=self.indacc[start] else end
        if(self.indacc[(start+end)/2]<i):
            return self.patchBinarySearch(i,((start+end)/2)+1,end)
        return self.patchBinarySearch(i,start,(start+end)/2)
    '''
    

    def getPatchesFromDataset(self, datasetindex, indexlist, onlyvalid=False):
        '''
        Yields the patches corresponding to the indexes in indexlist starting from the datasetindex-th dataset
        '''
        if(self.nPatch[datasetindex] is None):
            #the upper index limit is calculated using the accumulator indacc
            totpatches=self.indacc[0] if datasetindex==0 else self.indacc[datasetindex]-self.indacc[datasetindex-1]
            #get the name of the given dataset
            key=self.names[datasetindex]
            #get the region to be considered
            x1,y1,x2,y2=self.regions[key]
            #total number of rows in the rectangle
            nRows=x2-x1+1
            #total number of rows of patches
            nRowsPatch=nRows-(self.patch_dim-1)
            nRowsPatch=nRowsPatch if self.step==1 else (nRowsPatch//self.step)+(1 if nRowsPatch%self.step>0 else 0)
            #total number of columns in the rectangle
            nCols=y2-y1+1
            #total number of columns of patches
            nColsPatch=nCols-(self.patch_dim-1)
            nColsPatch=nColsPatch if self.step==1 else (nColsPatch//self.step)+(1 if nColsPatch%self.step>0 else 0)
            #assert totpatches==nRowsPatch*nColsPatch
            patch_n_el=self.patch_dim*self.patch_dim
            #save result for faster computation
            self.nPatch[datasetindex]=(nRowsPatch,nColsPatch)
        else:
            nRowsPatch,nColsPatch=self.nPatch[datasetindex]
            key=self.names[datasetindex]
            x1,y1,x2,y2=self.regions[key]
        
        patch_n_el=self.patch_dim*self.patch_dim
        for ind in indexlist:
            if self.savepatches:
                if (datasetindex,ind) in self.patchdict:
                    yield self.patchdict[(datasetindex,ind)]
                    continue
                else:
                    ind2=ind
            if self.getrotated:
                totpatches=self.indacc[0] if datasetindex==0 else self.indacc[datasetindex]-self.indacc[datasetindex-1]
                nRot=ind//totpatches
                ind=ind%totpatches
            #assert ind<totpatches
            riga=(ind//nColsPatch)*self.step
            colonna=(ind%nColsPatch)*self.step
            #ritorna la patch
            #TODO rifai con un solo yield patch in fondo e fai patch=None se serve
            if self.getrotated:
                patch=self.data[key][x1+riga:x1+riga+self.patch_dim, y1+colonna:y1+colonna+self.patch_dim]
                patch=np.rot90(patch,k=nRot)
                patch=patch[np.newaxis,:,:]
            else:
                patch=self.data[key][np.newaxis, x1+riga:x1+riga+self.patch_dim, y1+colonna:y1+colonna+self.patch_dim]
            if patch.size<patch_n_el or np.isnan(np.sum(patch)):
                yield None
            else:
                if self.transformdata:
                    if (not ((patch >= self.oldinterval[0]).all() and (patch <= self.oldinterval[1]).all())):
                        yield None
                        continue
                    if(onlyvalid):
                        yield patch
                        continue
                    patch=self.transform(patch)
                if(onlyvalid):
                    yield patch
                    continue
                if(self.subtractmean):
                    a=np.mean(patch,dtype=np.float32)
                    a=np.full(patch.shape,a)
                    patch=patch-a
                #if self.max>patch.min():
                #    self.max=patch.min()
                if self.savepatches:
                    self.patchdict[(datasetindex,ind2)]=patch
                yield patch
    
    def getPatches(self, indexlist_list):
        '''
        Yields the patches corresponding to the indexes in indexlist from each dataset 
        (include even empty lists if not interested in some datasets)
        '''
        for i in range(len(indexlist_list)):
            for patch in self.getPatchesFromDataset(i,indexlist_list[i]):
                yield patch,self.target_dict[self.names[i]]

    def getAllPatches(self):
        '''
        Yields all the patches from the selected regions
        '''
        indexlist_list=[]
        for i in range(len(self.names)):
            indexlist_list.append(range(self.indacc[0] if i==0 else self.indacc[i]-self.indacc[i-1]))
        for i,j in self.getPatches(indexlist_list):
            yield i,j
    
    def getMiniBatchesAndTargetsFromTupleList_sametest(self,indexlist, minibatch_dim):
        #remove excess indexes
        if(len(indexlist)%minibatch_dim)!=0:
            indexlist=indexlist[:-(len(indexlist)%minibatch_dim)]
        for i in range(len(indexlist)//minibatch_dim):
            mb=[]
            targets=[]
            for datasetindex,ipatch in indexlist[i*minibatch_dim:(i+1)*minibatch_dim]:
                patch=[]
                for p in self.getPatchesFromDataset(datasetindex,[ipatch]):
                    patch.append(p)
                patch=patch[0]

                if patch is not None:
                    mb.append(patch)
                    #0 se id pari, 1 se id dispari!
                    targets.append(ipatch%2)
                    
            yield mb,targets

    def getMiniBatchesAndTargetsFromTupleList_rottest(self,indexlist, minibatch_dim):
        #remove excess indexes
        if(len(indexlist)%minibatch_dim)!=0:
            indexlist=indexlist[:-(len(indexlist)%minibatch_dim)]
        for i in range(len(indexlist)//minibatch_dim):
            mb=[]
            targets=[]
            for datasetindex,ipatch in indexlist[i*minibatch_dim:(i+1)*minibatch_dim]:
                patch=[]
                for p in self.getPatchesFromDataset(datasetindex,[ipatch]):
                    patch.append(p)
                patch=patch[0]

                if patch is not None:
                    mb.append(patch)
                    #0 1 2 3 in base a rotazione
                    totpatches=self.indacc[0] if datasetindex==0 else self.indacc[datasetindex]-self.indacc[datasetindex-1]
                    targets.append(ipatch//totpatches)
                    
            yield mb,targets

    def getMiniBatchesAndTargetsFromTupleList(self,indexlist, minibatch_dim):
        #remove excess indexes
        if(len(indexlist)%minibatch_dim)!=0:
            indexlist=indexlist[:-(len(indexlist)%minibatch_dim)]
        for i in range(len(indexlist)//minibatch_dim):
            mb=[]
            targets=[]
            for datasetindex,ipatch in indexlist[i*minibatch_dim:(i+1)*minibatch_dim]:
                patch=[]
                for p in self.getPatchesFromDataset(datasetindex,[ipatch]):
                    patch.append(p)
                patch=patch[0]

                if patch is not None:
                    mb.append(patch)
                    targets.append(self.target_dict[self.names[datasetindex]])
                    
            yield mb,targets
    
    def getValidIndices(self):
        valid_indices=[]
        #for every dataset
        for datasetindex in range(len(self.names)):
            totpatches=self.indacc[0] if datasetindex==0 else self.indacc[datasetindex]-self.indacc[datasetindex-1]
            #for every patch in the dataset
            for ipatch in range(totpatches):
                #if the patch is valid, save the (dataset,patch_index)
                for patch in self.getPatchesFromDataset(datasetindex,[ipatch],onlyvalid=True):
                    if patch is not None:
                        valid_indices.append((datasetindex,ipatch))
                        if(self.getrotated):
                            totpatches=self.indacc[0] if datasetindex==0 else self.indacc[datasetindex]-self.indacc[datasetindex-1]
                            for i in range(3):
                                valid_indices.append((datasetindex,ipatch+(totpatches*(i+1))))

        return valid_indices

    def setTransform(self, oldinterval, newinterval):
        self.oldinterval=oldinterval
        self.newinterval=newinterval
        a1,b1=oldinterval
        a2,b2=newinterval
        newb1=b1-a1
        multiplier=(b2-a2)/(newb1)
        def transform(x):
            return ((x-a1)*multiplier)+a2
        self.transform=np.vectorize(transform,otypes=[np.float32])

        

def shuffleAndPartition(insieme,percentages):
    np.random.shuffle(insieme)
    totInd=len(insieme)
    cursor=0
    ret=[]
    for p in percentages[:-1]:
        quantity=int(p*totInd)
        ret.append(insieme[cursor:cursor+quantity])
        cursor=cursor+quantity
    ret.append(insieme[cursor:])
    return ret

if __name__=="__main__":
    #test1
    names=["a","b"]
    raw_data={"a":np.array(range(15)).reshape(3,5),"b":(np.array(range(6))+np.array(100)).reshape(2,3)}
    regions={"a":(0,0,2,4),"b":(0,0,1,2)}
    target_dict={"a":1,"b":2}
    print(raw_data)
    tsc=TrainingSetCreator(["a","b"],raw_data,regions,2,target_dict,step=1
                ,transformdata=False,subtractmean=False, getrotated=True, savepatches=True)
    #tsc.setTransform((0,15),(100,200))
    print(tsc.indacc)
    l=tsc.getValidIndices()
    #l = shuffleAndPartition(l,[1.0])[0]
    for minibatch,targets in tsc.getMiniBatchesAndTargetsFromTupleList(l,2):
        for i in range(len(minibatch)):
            print(targets[i])
            print(minibatch[i])
        print("*********************")
    for minibatch,targets in tsc.getMiniBatchesAndTargetsFromTupleList(l,2):
        for i in range(len(minibatch)):
            print(targets[i])
            print(minibatch[i])
        print("*********************")


    for i,j in tsc.getAllPatches():
        print(j)
        print(i)
    #test 2
    for i,j in tsc.getPatches([[0,7],[1]]):
        print(j)
        print(i)

    #test 3
    p=PatchSelector("../sample.h5", whitelist=['Argento_13_new4', 'Argento_15_new'], allow_print=False)
    print(p.names)
    regions=p.chooseRegions()
    raw_data=p.data
    target_dict={'Argento_13_new4':13,'Argento_15_new':15}
    tsc=TrainingSetCreator(p.names,raw_data,regions,3,target_dict)
    print("*********")
    tr,te=tsc.getTrainingTestingIndices(0.9)
    #validation
    tr,val=tsc.getTrainingTestingIndices(0.999999,subset=tr)
    print(val)
    print(te)
    for l in tsc.getMiniBatches(0,te[0],2):
        for p in l:
            print(p)

    #test 4
    for i,j in tsc.getAllPatches():
        print(j)
        print(i)