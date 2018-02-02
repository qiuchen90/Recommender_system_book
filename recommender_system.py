## This is a recommender system that the user can choose to use
## content based recommendation or 
import numpy as np
import scipy.io as sio
import numpy.linalg as LA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix

class recommender_system(object):
    def __init__(self):
        self.prediction = None
    def predict_content(self, R, type='user'):
        # User pairwise_distances function from sklearn to calculate the cosine
        # similarity between users and items respectively
        if type == 'user':
            similarity = pairwise_distances(csr_matrix(R), metric='cosine')
            mean_user_rating = R.mean(axis=1) 
            #You use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (R - mean_user_rating[:, np.newaxis]) 
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
#            pred = mean_user_rating[uid] + similarity[uid, :].dot(ratings_diff) / np.sum(np.abs(similarity[uid, :]))
        elif type == 'item':
            similarity = pairwise_distances(csr_matrix(R).T, metric='cosine')
 #           pred = R[uid,:].dot(similarity) / np.abs(similarity).sum(axis=0, keepdims=True)
            pred = R.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        return pred
    def predict_model(self, R, d, type='MSE'):
        return None
    def MSE(self,R,d):
        # Replace the nan values by zero
        index=np.isnan(R)
        Rnew = R.copy()
        row, col = Rnew.shape
        Rnew[index]=0
        U,s,V = LA.svd(Rnew, full_matrices=False)
        s[d:]=0
        out = np.dot(U, np.dot(np.diag(s), V))
        Rnew[index]=out[index]
        prediction = out
        return prediction
    def MLF(self, R, d, Lambda):
        row, col= R.shape
        U =  np.random.rand(row, d)
        V = np.random.rand(d, col)
        Rnew = R.copy()
        for i in range(100):
            current = np.dot(U, V)
            U=self.updateU(Rnew,V,Lambda)
            V=self.updateV(Rnew,U,Lambda)
        out = np.dot(U,V)
        prediction = out
        return prediction
    def updateU(self, R,V,Lambda):
        U = np.linalg.solve(np.dot(V, V.T)+Lambda*np.eye(len(V)), np.dot(V, R.T))        
        return U.T
    def updateV(self, R,U,Lambda):
        V = np.linalg.solve(np.dot(U.T,U)+Lambda*np.eye(len(U[0])), np.dot(U.T, R))
        return V
    def rmse(self, prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten() 
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))



