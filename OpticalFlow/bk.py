#!/usr/bin/python
import cv2
import sys
import numpy as np
import pandas as pd

from pyxmeans.xmeans import XMeans

from scipy import stats

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.metrics import silhouette_score

from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.mixture import GMM, DPGMM, VBGMM
import multiprocessing


from multiprocessing import Process, Pipe
from itertools import izip

# Plot result
from itertools import cycle

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]

def identify_blobs(image,processed):

    identified = image.copy()

    #BLOB DETECTION ...
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 0

    params.filterByColor = True 
    params.blobColor = 255

    params.filterByArea = True 

    params.minArea = 20
    params.maxArea = 1000

    params.filterByCircularity = False

    params.filterByConvexity = True 
    params.minConvexity = 0.5

    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)

    labels = detector.detect(processed)
    print "LABELS"
    print [label.size for label in labels]

    cv2.drawKeypoints(identified,labels,identified,color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return len(labels), identified


## TEST XMEANS ...
import time
from contextlib import contextmanager
@contextmanager
def TimerBlock(name):
    start = time.time()
    yield
    end = time.time()
    print "%s took %fs" % (name, end-start)
##


def show_clusters(data,preds,centers):

    '''
    Visualizes the PCA-reduced cluster data in two dimensions
    Adds cues for cluster centers and student-selected sample data
    '''

    predictions = pd.DataFrame(preds, columns = ['Cluster'])
    data = pd.DataFrame(data,columns=['x','y','r','g','b'])

    plot_data = pd.concat([predictions, data], axis = 1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize = (14,8))

    # Color map
    cmap = cm.get_cmap('gist_rainbow')

    for i, cluster in plot_data.groupby('Cluster'):   
    	cluster.plot(ax = ax, kind = 'scatter', x = 'y', y = 'x', \
    		color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=30);

	# Plot centers with indicators

    for i, c in enumerate(centers):
	ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
		alpha = 1, linewidth = 2, marker = 'o', s=200);
	ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100);

    # Set plot title
    ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");

def collect(arg):
    frames = []

    if isinstance(arg,list):
    # sequence of imgs
	for frame in arg:
	    frames.append(cv2.imread(frame))
    elif isinstance(arg,str):
	# video file
	cap = cv2.VideoCapture(arg)
	i = 0
	while True:
	    ret,frame = cap.read()
	    if not ret:
		break
	    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	    frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
	    frame = cv2.fastNlMeansDenoisingColored(frame)
	    frame = cv2.pyrMeanShiftFiltering(frame,5,5)
	    cv2.imwrite(str(i) + '.png',frame)
	    frames.append(frame)
	    i = i+1
	cap.release()

    return frames



def track(arg):
    cv2.namedWindow('frame')
    cv2.namedWindow('optical')

    cv2.moveWindow('frame', 100,100);
    cv2.moveWindow('optical',100,490);

    fig, ax = plt.subplots(figsize = (14,8))

    frames = collect(arg)

    prv = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frames[0])
    hsv[...,1] = 255

    for frame in frames:
	nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # REFORMAT DATA TO POINTS

	data = np.asarray([np.hstack((idx, np.log(1.+bgr[idx]))) for idx in np.ndindex(bgr.shape[:-1]) if sum(bgr[idx]) > 10],dtype=np.float64)

        #FIND BEST k 
        #n_range = range(2,10)
        #reduced_data = pd.DataFrame(data).sample(n=len(data)/4)
        #scores = parmap(lambda n:silhouette_score(reduced_data,KMeans(n_clusters=n).fit_predict(reduced_data)),n_range)
        #k = n_range[np.argmax(scores)]

        #GMM Clustering
	#clusterer = DPGMM(n_components=k)
	#preds = clusterer.fit_predict(data)
	#centers = clusterer.means_
	#show_clusters(data,preds,centers)
	#plt.show()

        # Meanshift Clustering
        bandwidth = estimate_bandwidth(data, quantile=0.2)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        # PLOT RESULT

        fig.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            ax.plot(data[my_members, 0], data[my_members, 1], col + '.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        #fig.title('Estimated number of clusters: %d' % n_clusters_)
        fig.canvas.draw()
        plt.show(block=False)

        print("number of estimated clusters : %d" % n_clusters_)


        # BLOB DETECTION

        #num_identified, identified = identify_blobs(frame,bgr)
	#cv2.imshow('identified', cv2.resize(identified,(0,0),fx=2,fy=2))

	cv2.imshow('frame', cv2.resize(frame,(0,0),fx=2,fy=2))
	cv2.imshow('optical', cv2.resize(bgr,(0,0),fx=2,fy=2))
	blended = cv2.addWeighted(frame,0.6,bgr,0.4,0.0)
	#cv2.imshow('optical', cv2.resize(blended,(0,0),fx=2,fy=2))

	k = cv2.waitKey(30) & 0xff
	if k == 27:
	    break
	prv = nxt

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
	print "USAGE : {} <filename>".format(sys.argv[0])
    else:
	#try:
	lim = int(sys.argv[1])
	track(['images/'+str(i)+'.png' for i in range(100,lim)])
	#except:
	#    track(sys.argv[1])
