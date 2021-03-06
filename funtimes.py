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
class PingPongPlayer(pd.DataFrame):
    '''
    PingPongPlayer is an extension of a pandas dataframe
    Class contains functions that operate on a specific data format
    corresponding to a recording of ping pong matches
    
    It contains functions which execute data analysis and plotting 
    '''
    
    
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
            
    def clean_frequent_issues(self):
        '''
        this goes through and removes some common data entry errors I've found
        and also adds the 'Winner' column, and returns the updated player
        '''
        self=self.dropna()
        for col in ['Hand','Incoming','Outgoing','Length','Result']:
            self[col]= self[col].apply( lambda s: s.replace(" ","") )
            self[col]= self[col].apply( lambda s: s.lower() )
        self['Winner'] = self.apply(lambda row: PingPongPlayer.decide_winner(row),axis=1)
        return PingPongPlayer(self)
        
    def convert_to_flts(self):
        '''
        function takes in a ping pong pd data frame
        converts the string values to numbers for use with sklearn tools
        returns a numpy array suitable for use in sklearn
        '''
        
        #flts=self.copy()
        flts=PingPongPlayer()
        flts=flts.dropna() #don't want nans in my machine learning
        flts['Hand'] = self['Hand'].replace(['f','b'],[1,2])
        flts['Incoming'] = self['Incoming'].replace(['b','p','s','l','m'],[1,2,3,4,5])
        flts['Outgoing'] = self['Outgoing'].replace(['b','p','s','l','m'],[1,2,3,4,5])
        flts['Length'] = self['Length'].replace(['s','m','l','x'],[1,2,3,4])
        flts['Result'] = self['Result'].replace(['w','m'],[1,2])
        flts['Winner'] = self['Winner']
        #leaving out game and match for the moment
        flts=flts.dropna() #don't want nans in my machine learning
        flts=flts.values[:,:].astype(float)
        return flts
        
    def analyze_loops(self):
        '''
        function takes in a pingpong pd data frame
        calculates how good their loops are
        '''
        loopsout=self.loc[(self['Outgoing']=='l') & (self['Player']==1)]
        loopsin= self.loc[(self['Incoming']=='l') & (self['Player']==2)]
        omcounters=loopsin.loc[(loopsin['Result']=='m')].shape[0]
        mcounters=loopsout.loc[(loopsout['Incoming']=='l') & (loopsout['Result']=='m')].shape[0]
        ocounters=loopsin.loc[(loopsin['Result']=='w')].shape[0]
        counters=loopsout.loc[(loopsout['Incoming']=='l') & (loopsout['Result']=='w')].shape[0]
        opp_counterloop_rate=(ocounters)/(omcounters+ocounters)
        counterloop_rate=(counters)/(mcounters+counters)         
        return counterloop_rate, opp_counterloop_rate 
    
    def length_winloss(self, plot=False):
        '''
        function takes in a ping pong pd dataframe
        calculates and makes a nice little 
        bar chart of their win/loss ratio vs point length
        '''
        won=self.loc[self['Winner']==1]
        lost=self.loc[self['Winner']==2]
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
            ax.set_ylim([0,array.max()+0.5])
            ax.set_xlabel('Match Length')
            ax.set_xticks(ind+width/2)
            ax.set_xticklabels(('short','medium','long','extralong'))
            plt.title('Ratio of points won by point length')
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%01.2f' % height,
                    ha='center', va='bottom')
            plt.show
            return None
            
            
    def results_winloss(self, plot=False):
        '''
        function takes in a ping pong pd dataframe
        calculates the amounts of winners and misses
        for both self and opponent
        '''
        won=self.loc[self['Winner']==1]
        lost=self.loc[self['Winner']==2]
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
            ax.set_ylabel('% of points won by category')
            plt.title('Points Won by Winners/Misses')
            ax.text(1.05, a[0]+0.025, '{:.0%} winners'.format(b[0]),color='k')
            ax.text(1.55, a[1]+0.025, '{:.0%} winners'.format(b[1]),color='k')
            ax.text(1.05, a[0]-0.06, '{:.0%} errors'.format(a[0]),color='w')
            ax.text(1.55, a[1]-0.06, '{:.0%} errors'.format(a[1]),color='w')
            ax.set_xticks(ind+width/2)
            ax.set_xticklabels(('Your Points','Opponent Points'))
            #ax.legend((wrects[0],lrects[0]),('Opponent Errors','Winners'),loc='lower center')
            plt.show
            return None
            
    def shots_winloss(self, plot=False):
        '''
        function takes in a ping pong pd dataframe
        and a string shot which is the type of shot we're interested in
        calculates for a particular shot how many resulted in a won point
        divided by how many resulted in a lost point
        '''
        winslist=[]
        for shot in ['s','p','b','l','m']:
            winners= self.loc[ (self['Winner']==1) & (self['Result']=='w') 
                            & (self['Outgoing']==shot) ].shape[0]
            opp_errors= self.loc[ (self['Winner']==1) & (self['Result']=='m') 
                            & (self['Incoming']==shot) ].shape[0]
            misses= self.loc[ (self['Winner']==2) & (self['Result']=='m') 
                            & (self['Outgoing']==shot) ].shape[0]
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
                    colorlist.append('c')
                elif wlist[i] < 0.7:
                    colorlist.append('b')
                else:
                    colorlist.append('g')
            ind=np.arange(N)
            width=0.75
            fig , ax = plt.subplots()
            rects= ax.bar(ind, wlist)
            for i in range(N):
                rects[i].set_color(colorlist[i])
            plt.title('Acc. By Shot Type')
            ax.set_ylabel('Winner+Opp. Error %')
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
            
    def quick_analyze(self,plot=True):
        self.results_winloss(plot)
        self.length_winloss(plot)
        self.shots_winloss(plot)
        
    def ML_fun(self):
        '''
        takes in a ping pong pd data frame
        converts/cleans the data and then runs a decision tree
        and scores the prediction outcome
        '''
        flts=self.convert_to_flts()
        X=flts[:,:-1]
        Y=flts[:,-1]
        
        from sklearn import decomposition
        pca=decomposition.PCA(n_components=2)
        new_X=pca.fit_transform(X)
        #colors=np.where(Y==1,'b','r')
        #plt.scatter(new_x[:,0],new_x[:,1],c=colors,s=20)
        #plt.show()
        #plt.scatter(new_X,Y)
        print("PCA Vectors: \n 'Hand'    'Incoming'   'Outgoing'    'Length'     'Miss/Winner' \n", pca.components_)
        
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
        
''' End of PingPongPlayer Class'''        
    
#Load up and clean some data
rawdata= pd.ExcelFile(r'Loop Data.xlsx')
sanketh=PingPongPlayer(pd.read_excel(rawdata, 'Sanketh All')); sanketh.name = 'sanketh'
nate=PingPongPlayer(pd.read_excel(rawdata, 'Nate All')); nate.name = 'nate'
sri=PingPongPlayer(pd.read_excel(rawdata, 'Sri All')); sri.name = 'sri'
wpimatch=PingPongPlayer(sanketh.loc[sanketh['Match']==2]); wpimatch.name="Sanketh v WPI #1"



    
    
    
    
    