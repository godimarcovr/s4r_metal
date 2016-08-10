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

if __name__=="__main__":
    print("Inizio programma...")
    #read file
    f = h5py.File("../sample.h5", "r")
    #init variables (in the future a list of all images?)
    #(maybe a configuration file to decide which samples are chosen?)
    group13=group15=data13=data15=mask13=mask15=None
    #explore dataset
    for group in f:
        if "13" in group:
            group13=f[group]
            for dataset in group13:
                if "data" in dataset:
                    data13=group13[dataset]
                if "MASK" in dataset:
                    mask13=group13[dataset]
        if "15" in group:
            group15=f[group]
            for dataset in group15:
                if "data" in dataset:
                    data15=group15[dataset]
                if "MASK" in dataset:
                    mask15=group15[dataset]
    #init window
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #convert to numpy array
    data13=data13[()]
    data15=data15[()]
    #filter out outlier values for better colormap (TODO use something better, maybe there is a mask
    #or you can use a custom colormap without touching the data)
    data13[(data13<-0.4) | (data13>0.4)]=np.nan
    data15[(data15<-0.4) | (data15>0.4)]=np.nan #RuntimeWarning? 15 si ma 13 no, boh!
    #apply mask to images
    images.append(ma.masked_array(data13,mask=mask13))
    images.append(ma.masked_array(data15,mask=mask15))
    #show the first image
    plt.imshow(images.pop())
    #create the rectangle selector with the callback
    rs=widgets.RectangleSelector(
        ax, onselect, drawtype='box',
        rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=False))
    #show the windows
    plt.show()
    #print results
    print(selected_regions)
    print("Fine programma...")

