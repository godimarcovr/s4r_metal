#!/usr/bin/env python
import numpy as np
import theano
import theano.tensor as T

import lasagne
import time
import metal_patch_selector as mps
import training_set_creator as tscreator

class NeuralNetCreator:

    def __init__(self, input_shape, num_outputs, patch_size_array, pooling_size_array=[]
                    , num_filters=32,fully_con=[]):
        '''
        input_shape: is a tuple that indicates the shape of the input to the network. Usually is (None,channels,rows,columns)
                    (16x16 RGB images would be (None,3,16,16))
        num_outputs: is the number of the outputs neurons to the network (usually number of classes)
        patch_size_array: dimensions of the convolutional layers.
        pooling_size_array: maxpooling size (must be [] or same size as patch_size_array)
        num_filters: how many filters in each convolutional layer
        fully_con: tuples (neurons, dropout rate) that describe the fullyconnected layers
        '''
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
        '''
        Builds the network structure
        '''
        #input is a 4d vector (an array of nchannels images righexcolonne)
        self.input_var = T.tensor4('inputs')
        #the output is a vector of integers (num_outputs long)
        self.target_var = T.ivector('targets')
        #input layer has the shape input_shape entering and is connected to input_var
        self.network = lasagne.layers.InputLayer(shape=self.input_shape, input_var=self.input_var)
        #for every desired convolutional_layer
        for patch_size_index in range(len(self.patch_size_array)):
            #get the filter size
            patch_size=self.patch_size_array[patch_size_index]
            #get the eventual pool size
            pool_size = self.pooling_size_array[patch_size_index] if len(self.pooling_size_array)>0 else 2
            #put the convolutional layer
            self.network = lasagne.layers.Conv2DLayer(self.network, num_filters=self.num_filters, 
                            filter_size=(patch_size, patch_size), 
                            nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
            #pooling layer currently disabled, decomment to use it
            #self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(pool_size, pool_size))
        #for each fullyconnected layer
        for num_units, dropout in self.fully_con:
            #add a dense layer
            self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network, p=dropout),
                                num_units=num_units, nonlinearity=lasagne.nonlinearities.rectify)
        #add the final layer as a SOFTMAX layer
        self.network = lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network,
                                         p=self.fully_con[-1][1] if len(self.fully_con)>0 else 0.5),
                            num_units=self.num_outputs, nonlinearity=lasagne.nonlinearities.softmax)

    def prepareTraining(self):
        '''
        Prepares the relevant functions
        '''
        #network output for training
        self.prediction = lasagne.layers.get_output(self.network)
        #the loss function to be minimized (the mean of the crossentropy between net output and targets)
        self.loss = lasagne.objectives.categorical_crossentropy(self.prediction, self.target_var)
        self.loss = self.loss.mean()

        #network parameters (weights and biases)
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)
        #update function (regulate learning rate and momentum here)
        self.updates = lasagne.updates.nesterov_momentum(
                self.loss, self.params, learning_rate=0.01, momentum=0.5)

        #network output for testing and validation (disables dropout etc.)
        self.test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        #test loss function
        self.test_loss = lasagne.objectives.categorical_crossentropy(self.test_prediction, self.target_var)
        self.test_loss = self.test_loss.mean()
        #test accuracy
        self.test_acc = T.mean(T.eq(T.argmax(self.test_prediction, axis=1), self.target_var)
                            , dtype=theano.config.floatX)
        
        #training function
        self.train_fn = theano.function([self.input_var, self.target_var], self.loss, updates=self.updates)
        #testing and validation function
        self.val_fn = theano.function([self.input_var, self.target_var], [self.test_loss,self.test_acc])
    
    def train2(self,tsc,num_epochs, training_percent, testing_percent, validation_percent):
        '''
        Given a TrainingSetCreator, a number of epochs and a partition of the training set, it trains the network
        '''
        #Shuffles and partitions the training set in a training set, testing set and validation set
        partitions=tscreator.shuffleAndPartition(tsc.getValidIndices(),[training_percent,testing_percent,validation_percent])
        training_set=partitions[0]
        testing_set=partitions[1]
        validation_set=partitions[2]
        #for each epoch
        for epoch in range(num_epochs):
            #change during debug if you want to exit and goto testing set
            flag=False
            if flag:
                break
            #randomly shuffles training set to achieve better results
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
        #test the results obtained
        for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList(testing_set,100):
            #target=np.full((len(minibatch)),tsc.target_dict[tsc.names[datasetindex]],dtype=np.int)
            err,acc = self.val_fn(minibatch,targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.4f} %".format( test_acc / test_batches * 100))

    '''
    #test
    def train2_sametest(self,tsc,num_epochs, training_percent, testing_percent, validation_percent):
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
            for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList_sametest(training_set,500):
                #print("Minibatch "+str(train_batches)+" Started...")
                #target=np.full((len(minibatch)),tsc.target_dict[tsc.names[datasetindex]],dtype=np.int)     
                train_err += self.train_fn(minibatch,targets)
                train_batches += 1
                #print("Minibatch "+str(train_batches)+" Ended")
            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList_sametest(validation_set,100):
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
        for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList_sametest(testing_set,100):
            #target=np.full((len(minibatch)),tsc.target_dict[tsc.names[datasetindex]],dtype=np.int)
            err,acc = self.val_fn(minibatch,targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.4f} %".format( test_acc / test_batches * 100))


    #test
    def train2_rottest(self,tsc,num_epochs, training_percent, testing_percent, validation_percent):
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
            for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList_rottest(training_set,500):
                #print("Minibatch "+str(train_batches)+" Started...")
                #target=np.full((len(minibatch)),tsc.target_dict[tsc.names[datasetindex]],dtype=np.int)     
                train_err += self.train_fn(minibatch,targets)
                train_batches += 1
                #print("Minibatch "+str(train_batches)+" Ended")
            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList_rottest(validation_set,100):
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
        for minibatch, targets in tsc.getMiniBatchesAndTargetsFromTupleList_rottest(testing_set,100):
            #target=np.full((len(minibatch)),tsc.target_dict[tsc.names[datasetindex]],dtype=np.int)
            err,acc = self.val_fn(minibatch,targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.4f} %".format( test_acc / test_batches * 100))
    '''

if __name__=="__main__":
    p=mps.PatchSelector("../sample.h5", whitelist=['Argento_7_new2','Argento_17_new'], allow_print=False)
    #p1=mps.PatchSelector("../sample.h5", whitelist=['Argento_13_new4'], allow_print=False)
    #print(p.names)
    regions=p.chooseRegions()
    raw_data=p.data
    #regions1=p1.chooseRegions()
    #regions['Argento_13_new4_1']=regions1['Argento_13_new4']
    #raw_data['Argento_13_new4_1']=p1.data['Argento_13_new4']
    target_dict={'Argento_7_new2':0,'Argento_17_new':1}
    patchdim=32
    #p.names.append('Argento_13_new4_1')
    tsc=tscreator.TrainingSetCreator(p.names,raw_data,regions,patchdim,target_dict,step=4
            ,transformdata=True,subtractmean=True,getrotated=False,savepatches=True)
    #new interval is to be
    tsc.setTransform((-0.45,0.45),(0.0,1.0))
    #tr,te=tsc.getTrainingTestingIndices(0.85)
    print("Start NN creation")
    nn=NeuralNetCreator((None,1,patchdim,patchdim),2,[5],fully_con=[(256,0.5),(256,0.5)],num_filters=8)
    print("NN created, starting training...")
    nn.train2(tsc,1000,0.80,0.10,0.10)
    #nn.train2_rottest(tsc,1000,0.80,0.10,0.10)



    #DA FARE
    #inganna con stesso campione e vedi se trova differenze (non funziona)
    #provo con altri campioni (funziona)
    #17 non trattato (confronto trattato e non trattato)