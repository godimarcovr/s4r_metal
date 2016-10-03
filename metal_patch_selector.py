import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

class PatchSelector:

    def __init__(self, filename, whitelist=[], allow_print=True, allsame=False):
        '''
        Constructor. Takes the hdf5 file path, a whitelist of accepted subgroups (default [] which means all of them)
        and a boolean flag to suppress prints if set to False
        '''
        #read dataset
        self.file = h5py.File(filename, "r")
        self.whitelist=whitelist
        self.data={}
        self.mask={}
        self.names=[]
        self.allow_print=allow_print
        self.allsame=allsame
        #for every subgroup
        for g in self.file:
            #empty whitelist means get everyone
            #if not empty, skip if the current group is not whitelisted
            if (self.whitelist) and (g not in self.whitelist):
                continue
            group=self.file[g]
            #save name
            self.names.append(g)
            #save data
            self.data[g]=group[g+" data"]
            #convert it to numpy array
            self.data[g]=self.data[g][()]
            #save mask
            self.mask[g]=group[g+" MASK"]
    
    def onselect(self, eclick, erelease):
        '''
        callback function on rectangle selection.
        save the rectangle boundaries in selected_regions and puts on the windows the next image. (images list)
        if there are no more images left it closes the window
        '''
        if eclick.ydata>erelease.ydata:
            eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
        if eclick.xdata>erelease.xdata:
            eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
        #print("("+eclick.xdata+","+eclick.ydata+")")
        #print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
        #print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
        #print('*******************')
        x1=eclick.ydata
        y1=eclick.xdata
        x2=erelease.ydata
        y2=erelease.xdata
        self.selected_regions[self.current_k]=(int(x1),int(y1),int(x2),int(y2))
        if self.allsame:
            while self.images:
                self.current_k,v=self.images.popitem()
                self.selected_regions[self.current_k]=(int(x1),int(y1),int(x2),int(y2))
            plt.close(self.figure)
        elif self.images:
            plt.cla()
            self.current_k,v=self.images.popitem()
            #print sample info
            if self.allow_print:
                print(self.current_k)
            plt.imshow(v)
        else:
            plt.close(self.figure)

    def chooseRegions(self):
        '''
        starts a graphical interface to select regions from the samples.
        '''
        #create window
        self.figure=plt.figure()
        ax=self.figure.add_subplot(111)
        self.images={}
        for key in self.names:
            #TODO change colormap or filter bad values somehow
            #apply mask
            self.images[key]=ma.masked_array(self.data[key],mask=self.mask[key])
        #get an arbitrary (key,value)
        self.current_k,v=self.images.popitem()
        #print sample info
        if self.allow_print:
            print(self.current_k)
        #show the first image
        plt.imshow(v)
        #initialize selected regions dictionary
        self.selected_regions={}
        #apply a RectangleSelector
        rs=widgets.RectangleSelector(ax, self.onselect, drawtype='box',
                    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=False))
        plt.show()
        #return results
        return self.selected_regions

        
if __name__=="__main__":
    #test
    p=PatchSelector("../sample.h5", whitelist=['Argento_13_new4', 'Argento_15_new'], allow_print=False)
    print(p.names)
    print(p.chooseRegions())