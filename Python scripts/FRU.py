import math
import numpy as np
import pandas as pd
from numpy import linalg as la
import concurrent.futures
import time
import multiprocessing

class create_regions():
    '''
    Parameters
    -----------
    df: Pandas dataframe object
    chunk_number: the number of parts to divide the dataframe based of processors, default is 4
    
    Returns 
    -----------
    membership_matrix: 2-D numpy array of size(number of decision classes, number of instances). Ones show that the instance belongs to the relevant decision class and zeros the opposite
    indices: the start and end indices of the instances per part/chunk
    D: 1-D numpy array containing the unique decision classes
    '''
    def __init__(self, Z, membership_matrix, D, numeric, nominal, distance, im, con, sm):
        '''
        Parameters
        -----------
        Z: 2-D numpy array of dimension (number of instances, number of features)
        membership_matrix: 2-D numpy array of size(number of decision classes, number of instances). Ones show that the instance belongs to the relevant decision class and zeros the opposite
        D: 1-D numpy array. Each entry represents one of the decision clases
        numeric: 1-D boolean array, each entry corresponds to a feature in the data. If True, the feature is numeric
        nominal: 1-D boolean array, each entry corresponds to a feature in the data. If True, the feature is nominal
        distance: sting specifying the distance function to be used, two options: HMOM and HEOM
        im: implicator function
        con: conjunction function
        sm: level of smoothing parameter to better separate the regions
        '''
        
        self.Z = Z
        self.membership_matrix = membership_matrix
        self.D = D
        self.numeric = numeric
        self.nominal = nominal
        self.distance = distance
        self.im = im
        self.con = con
        self.sm = sm
        
    def run_PNB(self, ind):
        '''
        Assigns three membership values to the input instance. The membership values correspond to the fuzzy-rough positive, negative and boundary region. 

        Parameters
        -----------
        ind: index of instance in the data. 
        
        Returns 
        -----------
        Three 2-D arrays corresponding the fuzzy-rough positive, negative and boundary region
        Each dimension of an array corresponds to a class of the target feature.
        '''
        
        POS = np.zeros((len(self.D), self.Z.shape[0]))
        NEG = np.zeros((len(self.D), self.Z.shape[0]))
        BND = np.zeros((len(self.D), self.Z.shape[0]))
        print('start PNB', ind)

        index = ind[0]
        for x_i in range(int(ind[0]),int(ind[1])):
            for k in range(len(self.D)):
                if x_i == index:
                    print(x_i)
                    index += 2500
                    
                if self.distance == 'HMOM':
                    s = np.sum(
                        np.abs(
                            np.subtract(self.Z.loc[x_i,self.numeric[:-1]].T.values,self.Z.loc[:,self.numeric[:-1]].values)),
                        axis=1) + np.sum(self.Z.loc[x_i,self.nominal[:-1]].T != self.Z.loc[:,self.nominal[:-1]],axis=1).values
                    distance_vector = np.exp(-self.sm * s.astype('float64')) # .astype('float16')
                    
                if self.distance == 'HEOM':
                    s = np.sum(
                        np.subtract(self.Z.loc[x_i,self.numeric[:-1]].T.values,self.Z.loc[:,self.numeric[:-1]].values)**2,
                        axis=1) + np.sum(self.Z.loc[x_i,self.nominal[:-1]].T != self.Z.loc[:,self.nominal[:-1]],axis=1).values
                    s = s**0.5
                    distance_vector = np.exp(-self.sm * s.astype('float64')) # .astype('float16')
                    
                POS[k][x_i], NEG[k][x_i], BND[k][x_i] = self.process(k, x_i, distance_vector, self.membership_matrix)
                
        return POS, NEG, BND

    def process(self, k, x_i, vector, membership_matrix):
        '''
        Calculates the membership values to the fuzzy-rough positive, negative and boundary regions

        Parameters
        -----------
        k: decision class of x_i
        x_i: index of instance in the data.
        vector: 1-D distance vector of dimension (1,instances). Each of the 1000 entries represents the distance between x_i and another instance in the data.
        membership_matrix: 2-D numpy array of size(number of decision classes, number of instances). Ones show that the instance belongs to the relevant decision class and zeros the opposite
       
        Returns 
        -----------
        Three 2-D arrays corresponding the fuzzy-rough positive, negative and boundary region
        Each dimension of an array corresponds to a class of the target feature.
        '''

        #print(k, x_i, vector)
        # fuzzy binary relation between x_i and all other instances
        fuzzy_relation = membership_matrix[k][x_i] * vector

        # hard membership mu_X_K(y)
        membership_Xk_y = membership_matrix[k]

        fuzzy_implication = self.implicator(fuzzy_relation, membership_Xk_y, self.im)
        #print(x_i, k, 'implicator', fuzzy_implication)
        infinum = min(1, fuzzy_implication)
        low_approximation = min(infinum, membership_matrix[k][x_i])
        #print('low done')

        fuzzy_conjunction = self.conjunction(fuzzy_relation, membership_Xk_y, self.con)
        #print(x_i, k,'conjunction:',fuzzy_conjunction)
        supremum = max(0, fuzzy_conjunction)
        upper_approximation = min(supremum, membership_matrix[k][x_i])
        
        return low_approximation, 1-upper_approximation, upper_approximation-low_approximation

    def implicator(self, a, b, mode):
        if mode == 'Luka':
            return min(np.min(1 - a + b), 1)

        if mode == 'Fodor':
            con1 = np.where(a<=b, 1, a)
            con2 = np.where((con1!=1) & (a>b), np.maximum(1-a,b), con1)
            return np.min(con2)

        if mode == 'Godel':
            return np.min(np.where(a<=b,1,b).flatten())

        if mode == 'Goguen':
            return np.min(np.where(a<=b,1,b/a).flatten())

    def conjunction(self, a, b, mode):
        if mode == 'Luka':
            return max(np.max(a + b - 1), 0)

        if mode == 'Standard':
            return min(np.min(a), np.min(b))

        if mode == 'Drastic':
            empty = np.empty((1000,))
            empty[np.where(b==1)] = a[np.where(b==1)]
            empty[np.where(a==1)] = b[np.where(a==1)]
            empty[np.where((a!=1) & (b!=1))] = 0
            
            return max(empty)

        if mode == 'Algebraic':
            return np.max(a*b)


class FRU_experimental_setup():

    def __init__(self, df):
        '''
        Parameters
        -----------
        df: Pandas dataframe object
        '''
        self.df = df

    def initialize_objects(self, chunk_number = 4):
        '''
        Parameters
        -----------
        chunk_number: the number of parts to divide the dataframe based of processors, default is 4
        
        Returns 
        -----------
        membership_matrix: 2-D numpy array of size(number of decision classes, number of instances). Ones show that the instance belongs to the relevant decision class and zeros the opposite
        indices: the start and end indices of the instances per part/chunk
        D: 1-D numpy array containing the unique decision classes
        '''
        
        # decision classes
        D = self.df.iloc[:,-1].unique()

        # initialize membership matrix - hard/crisp membership function in our paper
        membership_matrix = np.zeros((len(D), self.df.shape[0]))
        for k,k_index in zip(D,range(len(D))):
            for instance in range(self.df.shape[0]):
                instance_decision_class = self.df.values[instance,-1]
                membership_matrix[k_index][instance] = 1.0 if instance_decision_class == k else 0.0

        # divide dataset into chunks to feed into the processors for multiprocessing
        indices = []
        chunks = chunk_number
        increment=np.floor(self.df.shape[0]/chunks)
        for i in range(chunks):
            if i == (chunks-1):
                indices.append([int(i*increment), self.df.shape[0]])
            else: 
                indices.append([int(i*increment), int((i+1)*increment)])
        return membership_matrix, indices, D

    def uncertainty(self, full, prot, decision_class):
        '''
        Parameters
        ----------
        full: membership values of instances to the boundary regions using the full set of features,
        prot: membership values of instances to the boundary regions using the set of features without including the protected feature
        decision_class: index of the decision class

        Returns
        -------
        FRU-value attached to the specified decision class for the protected attribute that was removed
        '''
        
        POS_full, NEG_full, BND_full = full
        POS_prot, NEG_prot, BND_prot = prot

        diff_prot = BND_prot[decision_class] - BND_full[decision_class]
        diff_prot = np.where(diff_prot < 0, 0, diff_prot)

        return round((float(la.norm(diff_prot) / la.norm(BND_full[decision_class]))),2)

    def fr_regions(self, distance, im, con, sm, initial_conditions):
        '''
        Computes the membership values attached to the three fuzzy-rough regions for all instances in the dataset attached to all features
        The function leverages multiprocessing to speed up the computations

        Parameters
        ----------
        distance: 'HEOM' or 'HMOM'
        im: implicator function
        con: conjunction function
        sm: level of smoothing parameter to better separate the regions
        '''
        # target label is excluded
        membership_matrix, indices, D = initial_conditions
        Z = self.df.iloc[:,:-1]

        numeric = [False if self.df[col].dtype == 'object' else True for col in self.df]
        nominal = [True if self.df[col].dtype == 'object' else False for col in self.df]

        start = time.perf_counter()
        list_of_lists = [] # gathers up the membership values as computed for each dataset chunk
        #if __name__ == '__main__':
        with concurrent.futures.ProcessPoolExecutor() as executor:
                for result in executor.map(create_regions(Z, membership_matrix, D, numeric, nominal, distance, im, con, sm).run_PNB, indices):
                    list_of_lists.append(result)

        finish = time.perf_counter()
        print(f'Finished in {round(finish-start,2)} second(s)')

        # merge the four membership chunks
        results = np.array(list_of_lists[0])[:,:,indices[0][0]:indices[0][1]]
        chunk_no = 1
        for index in indices[1:]: 
            results = np.append(results, np.array(list_of_lists[chunk_no])[:,:,index[0]:index[1]],axis=2)
            chunk_no += 1

        return results
