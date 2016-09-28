import numpy as np
import theano
import theano.tensor as T
from training_set_creator import *
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import lasagne
import time
import metal_patch_selector as mps
import training_set_creator as tscreator

class NNRegression:

    def __init__(self, input_shape, num_outputs, patch_size_array, pooling_size_array=[]
                    , num_filters=32,fully_con=[]):
        assert (len(pooling_size_array)==0 or len(pooling_size_array)==len(patch_size_array))
        self.input_shape=input_shape
        self.num_outputs=num_outputs
        self.patch_size_array=patch_size_array
        self.pooling_size_array=pooling_size_array
        self.num_filters=num_filters
        self.fully_con=fully_con
        self.buildNN()
        self.prepareTraining()

    def buildNN(self):
        self.input_var = T.tensor4('inputs')
        self.target_var = T.fvector('targets')
        self.network = lasagne.layers.InputLayer(shape=self.input_shape, input_var=self.input_var)
        for patch_size_index in range(len(self.patch_size_array)):
            patch_size=self.patch_size_array[patch_size_index]
            pool_size = self.pooling_size_array[patch_size_index] if len(self.pooling_size_array)>0 else 2
            self.network = lasagne.layers.Conv2DLayer(self.network, num_filters=self.num_filters, 
                            filter_size=(patch_size, patch_size), 
                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
            #self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(pool_size, pool_size))
        for num_units, dropout in self.fully_con:
            self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network, p=dropout),
                                num_units=num_units, nonlinearity=lasagne.nonlinearities.rectify)
        
        self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network,
                                         p=self.fully_con[-1][1] if len(self.fully_con)>0 else 0.5),
                            num_units=self.num_outputs, nonlinearity=None)

    def prepareTraining(self):
        #loss objective to minimize
        self.prediction = lasagne.layers.get_output(self.network)
        self.prediction=self.prediction[:,0]
        #self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.target_var)
        self.loss =  lasagne.objectives.squared_error(self.prediction, self.target_var)
        self.loss = self.loss.mean()

        self.params = lasagne.layers.get_all_params(self.network, trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
                self.loss, self.params, learning_rate=0.01, momentum=0.4)

        self.test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.test_prediction=self.test_prediction[:,0]
        self.test_loss = lasagne.objectives.squared_error(self.test_prediction, self.target_var)
        self.test_loss = self.test_loss.mean()
        self.test_acc = T.mean(T.le(T.pow(T.sub(self.test_prediction,self.target_var),2),0.01)
                            , dtype=theano.config.floatX)

        self.train_fn = theano.function([self.input_var, self.target_var], self.loss, updates=self.updates)

        self.val_fn = theano.function([self.input_var, self.target_var], [self.test_loss,self.test_acc])

        self.use = theano.function([self.input_var],[self.test_prediction])

    def train2(self,tsc,num_epochs, training_percent, testing_percent, validation_percent):
        partitions=tscreator.shuffleAndPartition(tsc.getValidIndices(),[training_percent,testing_percent,validation_percent])
        training_set=partitions[0]
        testing_set=partitions[1]
        validation_set=partitions[2]
        for epoch in range(num_epochs):
            #change during debug
            flag=False
            if flag:
                break
            np.random.shuffle(training_set)
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList(training_set,500):
                #print("Minibatch "+str(train_batches)+" Started...")
                #target=np.full((len(minibatch)),tsc.target_dict[tsc.names[datasetindex]],dtype=np.int)     
                train_err += self.train_fn(minibatch,targets)
                train_batches += 1
                #print("Minibatch "+str(train_batches)+" Ended")
            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList(validation_set,100):
                #target=np.full((len(minibatch)),tsc.target_dict[tsc.names[datasetindex]],dtype=np.int)
                err, acc = self.val_fn(minibatch, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
            
            print("Epoch {} of {} took {:.3f}s".format(
                                            epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.4f} %".format(val_acc / val_batches * 100))
            print("****")
            if flag:
                break
        test_err = 0
        test_acc = 0
        test_batches = 0
        for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList(testing_set,100):
            #target=np.full((len(minibatch)),tsc.target_dict[tsc.names[datasetindex]],dtype=np.int)
            err,acc = self.val_fn(minibatch,targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.4f} %".format( test_acc / test_batches * 100))


def generateData(dim,num):
    names=[]
    data={}
    regions={}
    targets={}
    srng = RandomStreams()
    rv_u = srng.uniform(dim)
    rv_u = T.mul(rv_u,50.0)
    rv_u = T.sub(rv_u,25.0)
    f = function([], rv_u)
    for i in range(num):
        name=str(i)
        names.append(name)
        data[name]=f()
        regions[name]=(0,0,dim[0]-1,dim[1]-1)
        #targets[name]=np.sum(data[name],dtype=np.float32)
        targets[name]=np.amax(data[name])
    return names,data,regions,targets



    

if __name__=="__main__":
    dim=5
    names, raw_data, regions, target_dict = generateData((dim,dim),50000)
    #modificato tsc per ottimizzare indacc (e altro?)
    tsc=TrainingSetCreator(names,raw_data,regions,dim,target_dict,transformdata=False
            ,subtractmean=False, getrotated=False, savepatches=True)
    print("Start NN creation")
    nn=NNRegression((None,1,dim,dim),2,[],fully_con=[(1024,0.0),(128,0.0)],num_filters=8)
    print("NN created, starting training...")
    nn.train2(tsc,50,0.80,0.10,0.10)

    '''
    #2.0
    a=np.array([0.0,0.5,0.5,1.0]).reshape(2,2)
    a=a[np.newaxis,:,:]
    print(a.sum())
    a=[a]
    print(nn.use(a))
    #4.0
    a=np.array([1.0,1.0,1.0,1.0]).reshape(2,2)
    a=a[np.newaxis,:,:]
    print(a.sum())
    a=[a]
    print(nn.use(a))
    #1.0
    a=np.array([2.0,-1.0,0.5,-0.5]).reshape(2,2)
    a=a[np.newaxis,:,:]
    print(a.sum())
    a=[a]
    print(nn.use(a))
    #1.0
    a=np.array([0.1,0.2,0.3,0.4],dtype=np.float32).reshape(2,2)
    a=a[np.newaxis,:,:]
    print(a.sum())
    a=[a]
    print(nn.use(a))
    '''
    a=np.array(range(25)).reshape(5,5)
    a=a[np.newaxis,:,:]
    print(a.sum())
    a=[a]
    print(nn.use(a))