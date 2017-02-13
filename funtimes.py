# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 10:12:17 2017

@author: raymo

This file will include reading, data cleaning,
and some data analysis on the ping pong tournament
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def decide_winner(row):
    """
    For use on ping pong pd data frames
    Decides the winner of a point based on the result (winner, miss)
                                        and the player (1,2)
    """
    if row['Result']=='m':
        if row['Player']==2:
            return 1
        else:
            return 2
    else: #result was w
        if row['Player']==1:
            return 1
        else:
            return 2
            
def convert_to_flts(player):
    '''
    function takes in a ping pong pd data frame
    converts the string values to numbers for use with sklearn tools
    returns a numpy array suitable for use in sklearn
    '''
    
    #flts=player.copy()
    flts=pd.DataFrame()
    flts=flts.dropna() #don't want nans in my machine learning
    flts['Hand'] = player['Hand'].replace(['f','b'],[1,2])
    flts['Incoming'] = player['Incoming'].replace(['b','p','s','l','m'],[1,2,3,4,5])
    flts['Outgoing'] = player['Outgoing'].replace(['b','p','s','l','m'],[1,2,3,4,5])
    flts['Length'] = player['Length'].replace(['s','m','l','x'],[1,2,3,4])
    flts['Result'] = player['Result'].replace(['w','m'],[1,2])
    flts['Winner'] = player['Winner']
    #leaving out game and match for the moment
    flts=flts.dropna() #don't want nans in my machine learning
    flts=flts.values[:,:].astype(float)
    return flts
    
def analyze_loops(player):
    '''
    function takes in a pingpong pd data frame
    calculates how good their loops are
    '''
    loopsout=player.loc[(player['Outgoing']=='l') & (player['Player']==1)]
    loopsin= player.loc[(player['Incoming']=='l') & (player['Player']==2)]
    omcounters=loopsin.loc[(loopsin['Result']=='m')].shape[0]
    mcounters=loopsout.loc[(loopsout['Incoming']=='l') & (loopsout['Result']=='m')].shape[0]
    ocounters=loopsin.loc[(loopsin['Result']=='w')].shape[0]
    counters=loopsout.loc[(loopsout['Incoming']=='l') & (loopsout['Result']=='w')].shape[0]
    opp_counterloop_rate=(ocounters)/(omcounters+ocounters)
    counterloop_rate=(counters)/(mcounters+counters)         
    return counterloop_rate, opp_counterloop_rate 

def length_winloss(player, plot=False):
    '''
    function takes in a ping pong pd dataframe
    calculates and makes a nice little 
    bar chart of their win/loss ratio vs point length
    '''
    won=player.loc[player['Winner']==1]
    lost=player.loc[player['Winner']==2]
    winlen=won['Length'].value_counts()/won.shape[0]
    lostlen=lost['Length'].value_counts()/lost.shape[0]
    wlratio=winlen.div(lostlen)
    wlratio=wlratio.fillna(value=0)
    if not plot:
        return wlratio
    else:
        #do plotting stuff
        array=wlratio.as_matrix()
        colorlist=[]
        N=len(array)
        for i in range(N):
            if array[i] < 0.5:
                colorlist.append('r')
            elif array[i] < 0.8:
                colorlist.append('y')
            elif array[i] < 1.2:
                colorlist.append('b')
            elif array[i] < 1.5:
                colorlist.append('c')
            else:
                colorlist.append('g')
        ind=np.arange(N)
        width=0.75
        fig , ax = plt.subplots()
        rects= ax.bar(ind, array, color = 'k')
        for i in range(N):
            rects[i].set_color(colorlist[i])
        ax.set_ylabel('Win/Loss')
        ax.set_xlabel('Match Length')
        ax.set_xticks(ind+width/2)
        ax.set_xticklabels(('short','medium','long','extralong'))
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%01.2f' % height,
                ha='center', va='bottom')
        plt.show
        return None
        
        
def results_winloss(player, plot=False):
    '''
    function takes in a ping pong pd dataframe
    calculates the amounts of winners and misses
    for both player and opponent
    '''
    won=player.loc[player['Winner']==1]
    lost=player.loc[player['Winner']==2]
    wincause=won['Player'].value_counts()/won.shape[0]
    wincause.name = "Win Cause"
    lostcause=lost['Player'].value_counts()/lost.shape[0]
    lostcause.name = "Lost Cause"
    if not plot:
        return wincause, lostcause
    else:
        #do plotting stuff
        warray=wincause.as_matrix()
        larray=lostcause.as_matrix()
        a=np.array([warray[0],larray[0]])
        b=np.array([warray[1],larray[1]])
        ind=np.array([1,1.5])
        width=0.25
        fig , ax = plt.subplots()
        ax.bar(ind, a, width, color = 'k')
        ax.bar(ind, b, width, bottom = a, color = 'w')
        ax.set_ylabel('Percent of pointss won by category')
        ax.text(1.05, a[0]+0.025, '{:.0%} winners'.format(b[0]),color='k')
        ax.text(1.55, a[1]+0.025, '{:.0%} winners'.format(b[1]),color='k')
        ax.text(1.05, a[0]-0.06, '{:.0%} errors'.format(a[0]),color='w')
        ax.text(1.55, a[1]-0.06, '{:.0%} errors'.format(a[1]),color='w')
        ax.set_xticks(ind+width/2)
        ax.set_xticklabels(('Your Points','Opponent Points'))
        #ax.legend((wrects[0],lrects[0]),('Opponent Errors','Winners'),loc='lower center')
        plt.show
        return None
        
def shots_winloss(player, plot=False):
    '''
    function takes in a ping pong pd dataframe
    and a string shot which is the type of shot we're interested in
    calculates for a particular shot how many resulted in a won point
    divided by how many resulted in a lost point
    '''
    winslist=[]
    for shot in ['s','p','b','l','m']:
        winners= player.loc[ (player['Winner']==1) & (player['Result']=='w') 
                        & (player['Outgoing']==shot) ].shape[0]
        opp_errors= player.loc[ (player['Winner']==1) & (player['Result']=='m') 
                        & (player['Incoming']==shot) ].shape[0]
        misses= player.loc[ (player['Winner']==2) & (player['Result']=='m') 
                        & (player['Outgoing']==shot) ].shape[0]
        winsratio= (winners+opp_errors)/(misses+winners+opp_errors)
        winslist.append([shot,winsratio])

    if not plot:
        return winslist
    else:
        ilist=[]
        wlist=[]
        for win in winslist:
            ilist.append(win[0])
            wlist.append(win[1])
        colorlist=[]
        N=len(wlist)
        for i in range(N):
            if wlist[i] < 0.3:
                colorlist.append('r')
            elif wlist[i] < 0.45:
                colorlist.append('y')
            elif wlist[i] < 0.55:
                colorlist.append('b')
            elif wlist[i] < 0.7:
                colorlist.append('c')
            else:
                colorlist.append('g')
        ind=np.arange(N)
        width=0.75
        fig , ax = plt.subplots()
        rects= ax.bar(ind, wlist)
        for i in range(N):
            rects[i].set_color(colorlist[i])
        ax.set_ylabel('Winer/Opp Error percent')
        ax.set_xlabel('Shot type')
        ax.set_xticks(ind+width/2)
        ax.set_xticklabels(ilist)
        ax.set_ylim([0,1])
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '{:.0%}'.format(height),
                ha='center', va='bottom')
        plt.show
        return None 
        

#Load up and clean some data
rawdata= pd.ExcelFile(r'C:\Users\raymo\Dropbox\Personal\Ping Pong\Loop Data.xlsx')
sanketh=pd.read_excel(rawdata, 'Sanketh All'); sanketh.name = 'sanketh'
nate=pd.read_excel(rawdata, 'Nate All'); nate.name = 'nate'
#nate's data is missing forehad/backhand data in the first game

sri=pd.read_excel(rawdata, 'Sri All'); sri.name = 'sri'
nate['Winner'] = nate.apply (lambda row: decide_winner (row),axis=1)
sanketh['Winner'] = sanketh.apply (lambda row: decide_winner (row),axis=1)
sri['Winner'] = sri.apply (lambda row: decide_winner (row),axis=1)

##analyzing nate
#length_winloss(nate,plot=True)
#analyze_loops(nate)
#results_winloss(nate,plot=True)    
#shots_winloss(nate,plot=True)
#
##analyzing sanketh
#length_winloss(sanketh,plot=True)
#analyze_loops(sanketh)
#results_winloss(sanketh,plot=True)
#shots_winloss(sanketh,plot=True)
##sanketh's results just against the super good dude
#wpimatch=sanketh.loc[sanketh['Match']==2]
#wpimatch.name="Sanketh v WPI #1"
#length_winloss(wpimatch,plot=True)
#results_winloss(wpimatch,plot=True)
#shots_winloss(wpimatch,plot=True)
##Comparing short poflts (usually service return or a setup)
#notm2misses=sanketh.loc[(sanketh['Winner']==2)&(sanketh['Length']=='s')&(sanketh['Match']!=2)].shape[0]
#m2misses=sanketh.loc[(sanketh['Winner']==2)&(sanketh['Length']=='s')&(sanketh['Match']==2)].shape[0]
#
##analyzing sri
#length_winloss(sri,plot=True)
#analyze_loops(sri)
#results_winloss(sri,plot=True)
#shots_winloss(sri,plot=True)

"""
#Apply sklearn stuff to each player
#playing with K-Nearest-Neighbor classification.
#I dont expect this to work well because I don't think there's enough 
#distance between values in the phase space
nateflts=convert_to_flts(nate)
indices = np.random.permutation(len(nateflts))
nate_X=nateflts[:,:-1]
nate_Y=nateflts[:,-1]
indices=np.random.permutation(len(nate_X))
nate_X_train=nate_X[indices[:-10]]
nate_Y_train=nate_Y[indices[:-10]]

nate_X_test=nate_X[indices[-10:]]
nate_Y_test=nate_Y[indices[-10:]]
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(nate_X_train, nate_Y_train)
predicted=knn.predict(nate_X_test)
actual=nate_Y_test
print(predicted, actual)

#I also tried using a support vector classification. This predicted quite well
#however I am really interested in finding relationships between data
#that I haven't found based on my own intuition, so black box isnt good enough

from sklearn import svm
clf=svm.SVC().fit(x_train,y_train)
clf.score(x_test,y_test)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
X=nate_X[:,1:2]
X=nate_X[:,1:3]
Y=nate_Y
x_min,x_max = 1, 5
y_min,y_max = 1, 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
clf2=svm.SVC().fit(X,Y)
Z=clf2.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.coolwarm)
plt.show()
"""


def ML_fun(player):
    '''
    takes in a ping pong pd data frame
    converts/cleans the data and then runs a decision tree
    and scores the prediction outcome
    '''
    flts=convert_to_flts(player)
    X=flts[:,:-1]
    Y=flts[:,-1]
    
    from sklearn import decomposition
    pca=decomposition.PCA(n_components=2)
    new_X=pca.fit_transform(X)
    #colors=np.where(Y==1,'b','r')
    #plt.scatter(new_x[:,0],new_x[:,1],c=colors,s=20)
    #plt.show()
    #plt.scatter(new_X,Y)
    print("PCA Vectors: ", pca.components_)
    
    #from sklearn import cross_validation
    #x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    #                                new_X,Y, test_size=0.25,random_state=1)
    
    from sklearn import tree
    dtm=tree.DecisionTreeClassifier(
                                    max_depth=3,
                                    min_samples_leaf=5,
                                    criterion='entropy'
                                    )
    dtm.fit(new_X,Y)
    print("Decision tree score: ", dtm.score(new_X,Y))
    print("PCA vector Importances: ", dtm.feature_importances_)
    
    from sklearn.neighbors import KNeighborsClassifier
    #knn= KNeighborsClassifier(
    #                          weights='distance'
    #                          )
    #knn.fit(x_train, y_train)
    #print("KNN score: ", knn.score(x_test,y_test))
    
    from matplotlib.colors import ListedColormap
    #Plotting bit for knn
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])#,'#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])#, '#0000FF'])
    weights = 'distance' #leaving out uniform cus distance is always doing better
    # we create an instance of Neighbours Classifier and fit the data.
    k=5
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights)
    clf.fit(new_X, Y)
    print("KNN score w/ weights = '%s' and k= %i:  " % (weights, k), clf.score(new_X,Y))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = new_X[:, 0].min() - 1, new_X[:, 0].max() + 1
    y_min, y_max = new_X[:, 1].min() - 1, new_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(new_X[:, 0], new_X[:, 1], c=Y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i, weights = '%s')"
              % (k, weights))
    plt.show()
    
    
    
    
    