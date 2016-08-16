import numpy as np
from metal_patch_selector import *
from random import shuffle

class TrainingSetCreator:

    def __init__(self, names, raw_data, regions, patch_dim, target_dict,getrotated=False):
        '''
        names: list of strings which are the names of the datasets (to enforce an order)
        raw_data: dictionary with names as keys and images as values
        regions: dictionary of tuples (x1,y1,x2,y2) which is the rectangle selection
        patch_dim: a integer which represents the square side dimension
        target_dict: dictionary with names as keys and their classification number as value
        getrotated:a flag that activates the artificial increase of data patches by rotating existing patches
        '''
        self.data=raw_data
        self.names=names
        self.regions=regions
        self.patch_dim=patch_dim
        self.target_dict=target_dict

        self.compileIndexAccumulator()
    
    def getNumberOfAccPatches(self,n):
        '''
        Returns the total number of patches for the first n datasets
        '''
        #starts from 0
        count=0
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
            #total number of columns in the rectangle
            nCols=y2-y1+1
            #total number of columns of patches
            nColsPatch=nCols-(self.patch_dim-1)
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

    
    def patchBinarySearch(self,i,start,end):
        '''
        Currently not used, not even tested
        '''
        if(start>=end):
            return start
        if(start+1==end):
            return start if i<=self.indacc[start] else end
        if(self.indacc[(start+end)/2]<i):
            return self.patchBinarySearch(i,((start+end)/2)+1,end)
        return self.patchBinarySearch(i,start,(start+end)/2)

    
    def getPatchesFromDataset(self, datasetindex, indexlist):
        '''
        Yields the patches corresponding to the indexes in indexlist starting from the datasetindex-th dataset
        '''
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
        #total number of columns in the rectangle
        nCols=y2-y1+1
        #total number of columns of patches
        nColsPatch=nCols-(self.patch_dim-1)
        #assert totpatches==nRowsPatch*nColsPatch
        for ind in indexlist:
            #assert ind<totpatches
            riga=ind//nColsPatch
            colonna=ind%nColsPatch
            #ritorna la patch
            yield self.data[key][x1+riga:x1+riga+self.patch_dim,y1+colonna:y1+colonna+self.patch_dim]
    
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

    def getMiniBatches(self, datasetindex, indexlist, minibatch_dim):
        '''
        datasetindex: the index in the self.names, indicates which dataset is considered
        indexlist: is the list of indices which are requested
        minibatch_dim: is the dimension of the single minibatch

        by iterating on the call, a sequence of minibatches is returned
        '''
        #assert len(indexlist)%minibatch_dim==0
        for i in range(len(indexlist)//minibatch_dim):
            l=[]

            if(((i+1)*minibatch_dim)>len(indexlist)):
                for patch in self.getPatchesFromDataset(datasetindex, indexlist[i*minibatch_dim:]):
                    l.append(patch)
            else:
                for patch in self.getPatchesFromDataset(datasetindex, indexlist[i*minibatch_dim:(i+1)*minibatch_dim]):
                    l.append(patch)
            yield l
    
    def getTrainingTestingIndices(self,training_percent):
        '''
        training_percent: is the percentage of training elements required, the remaining are testing

        returns two lists of indices train and test. in train[i] we have the indices of the training set of the dataset i
                the remaining ones are testing set
        '''
        retTrain=[]
        retTest=[]
        for datasetindex in range(len(self.names)):
            #the upper index limit is calculated using the accumulator indacc
            totpatches=self.indacc[0] if datasetindex==0 else self.indacc[datasetindex]-self.indacc[datasetindex-1]
            indices = np.arange(totpatches)
            np.random.shuffle(indices)
            indices=indices.tolist()
            training_quantity=int(training_percent*totpatches)
            testing_quantity=totpatches-training_quantity
            retTrain.append(indices[:training_quantity])
            retTest.append(indices[training_quantity:])
        return retTrain,retTest


if __name__=="__main__":
    #test1
    names=["a","b"]
    raw_data={"a":np.array(range(15)).reshape(3,5),"b":(np.array(range(6))+np.array(100)).reshape(2,3)}
    regions={"a":(0,0,2,4),"b":(0,0,1,2)}
    target_dict={"a":1,"b":2}
    print(raw_data)
    tsc=TrainingSetCreator(["a","b"],raw_data,regions,2,target_dict)
    print(tsc.indacc)
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
    print("asd")
    tr,te=tsc.getTrainingTestingIndices(0.99999)
    print(te)
    for l in tsc.getMiniBatches(0,te[0],2):
        for p in l:
            print(p)

    #test 4
    for i,j in tsc.getAllPatches():
        print(j)
        print(i)
    