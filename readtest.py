'''
h5py file reading test, data visualization test, region selection test
'''

import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

images=[]
selected_regions=[]


def onselect(eclick, erelease):
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
    print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
    print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
    print('*******************')
    if(len(names)>0):
        print(names.pop())
    x1=eclick.xdata
    y1=eclick.ydata
    x2=erelease.xdata
    y2=erelease.ydata
    selected_regions.append((x1,y1,x2,y2))
    if(len(images)>0):
        plt.cla()
        plt.imshow(images.pop())
    else:
        plt.close(fig)

def getDataAndMask(f):
    '''
    Given a hdf5 file f, save the corresponding datasets marked with data and MASK contained in the groups of the root
    '''
    #TODO (maybe a configuration file to decide which samples are chosen?)
    #example Argento_13_new4 and Argento_15_new
    data=[]
    mask=[]
    names=[]
    for g in f:
        group=f[g]
        names.append(g)
        for dataset in group:
            if "data" in dataset:
                data.append(group[dataset])
            elif "MASK" in dataset:
                mask.append(group[dataset])
    return data,mask,names

if __name__=="__main__":
    print("Inizio programma...")
    #read file
    f = h5py.File("../sample.h5", "r")
    #explore dataset
    data,mask,names=getDataAndMask(f)
    
    #init window
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for data1 in data:
        #convert to numpy array
        data1=data1[()]
        #filter out outlier values for better colormap (TODO use something better, maybe there is a mask
        #or you can use a custom colormap without touching the data)
        data1[(data1<-0.4) | (data1>0.4)]=np.nan #RuntimeWarning? Tutti tranne il primo, boh!
        #apply mask to images
        mask1=mask.pop(0)
        images.append(ma.masked_array(data1,mask=mask1))
    #show the first image
    plt.imshow(images.pop())
    #create the rectangle selector with the callback
    rs=widgets.RectangleSelector(
        ax, onselect, drawtype='box',
        rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=False))
    #show the windows
    print(names.pop())
    plt.show()
    #print results
    print(selected_regions)
    print("Fine programma...")

