import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import multiprocessing as mp
from itertools import repeat
from scipy.stats import wishart
from numpy.random import multivariate_normal
import random
#from types import SimpleNamespace


class BPMF():
    def __init__(self, dim=10, num_epochs=10):#, batch_size=1000):
        self.dim=dim #number of latent features
      
        self.epoch=num_epochs 
       
        self.Item_vector = None
        self.User_vector = None

        self.num_User = None
        self.num_Item = None

        self.Item_lambda = None
        self.User_lambda = None

        #Variables used in Gaussian-Wishart
        self.mu_0=None
        self.v_0=None
        self.inv_W_0=None
        self.beta_0=None

        #Data used in training updates
        self.train_data=None
        self.train_rating=None

        #Predictions variables:
        self.sum_train_predict_rating=None
        self.sum_test_predict_rating=None
        self.sum_test_modi_predict_rating=None

        #Variables used to draw U and V
        self.mu_U=None
        self.Lambda_U=None
        self.mu_V=None
        self.Lambda_V=None

        
    def initialize_parameters(self):
        #Parameters predetermined used in the Gaussian-Wisshart:
        self.mu_0=np.zeros((self.dim,1))
        self.v_0=self.dim
        self.inv_W_0=np.linalg.inv(np.eye(self.dim)*50)
        self.beta_0=2

        #Test without PMF initalization
        self.User_vector = np.random.normal(0.0, 1.0/5,(self.dim, self.num_User))
       
        self.Item_vector = np.random.normal(0.0, 1.0/5, (self.dim, self.num_Item))
       
    
    
    def sample_hyperparameters(self, User_vector, N):
        #Definition of all the hyperparameters used
        U_bar=np.average(User_vector, axis=1, keepdims=True)
        U_i_U_bar=User_vector-U_bar #Extra added U_i - U_bar
        S_bar=np.dot(U_i_U_bar, U_i_U_bar.T)#Not divided by N as we later just have to multiply by N in W_0_star
        mu_0_U_bar=self.mu_0-U_bar #Ekstra added mu_0 - U_bar
        beta_0_star=self.beta_0+N
        v_0_star=self.v_0+N
        mu_0_star=(self.beta_0+self.mu_0+N*U_bar)/(self.beta_0+N)
        W_star_inv=self.inv_W_0+S_bar+self.beta_0*N/(self.beta_0+N)*np.dot(mu_0_U_bar, mu_0_U_bar.T)
        W_star=np.linalg.inv(W_star_inv)

        #Drawing lambda_U from the wishart distribution
        Lambda_U = wishart(df=v_0_star, scale=W_star).rvs()
        
        #calculating covarians matrix used in drawing from the multivariate normal distriubtion
        cov=np.linalg.inv(beta_0_star*Lambda_U)

        #Drawing mu_u for multivariate normal distribution
        mu_U=multivariate_normal(mu_0_star[:,0],cov)
        #print('Mu_U: ', mu_U, ' Lambda_U: ', Lambda_U)
        
        return mu_U, Lambda_U

    def sample_features_U(self, i):
        idx=(self.train_data[:,0]==i)
        V_j = self.Item_vector[:,self.train_data[idx,1]]
    
        Lambda_i_star=self.Lambda_U + 1*np.dot(V_j, V_j.T) #The 1 is alpha
        Lambda_i_star_inv=np.linalg.inv(Lambda_i_star)
        mu_i_star=np.dot(Lambda_i_star_inv,(1*np.dot(self.train_rating[idx],V_j.T)+np.dot(self.Lambda_U,self.mu_U))) #The 1 is alpha
        return multivariate_normal(mu_i_star, Lambda_i_star_inv)
    
    def sample_features_V(self, j):
        idx=(self.train_data[:,1]==j)
        test=self.train_data[idx,0]
        #print(test)
        U_i = self.User_vector[:,test]
        
        Lambda_i_star=self.Lambda_V + 1*np.dot(U_i, U_i.T) #The 1 is alpha
        Lambda_i_star_inv=np.linalg.inv(Lambda_i_star)
        mu_i_star=np.dot(Lambda_i_star_inv,(1*np.dot(self.train_rating[idx],U_i.T)+np.dot(self.Lambda_V,self.mu_V))) #The 1 is alpha
        return multivariate_normal(mu_i_star, Lambda_i_star_inv)


    def train(self, train_data, test_data, user_pmf, item_pmf, burn_in=0):
        self.num_User = int(max(np.amax(train_data[:, 0]), np.amax(test_data[:, 0])))+1
        self.num_Item = int(max(np.amax(train_data[:, 1]), np.amax(test_data[:, 1])))+1

        self.initialize_parameters()
        train_rating, test_rating = self.standardize(train_data, test_data)

        #Removing rating column for smaller data
        train_data=train_data[:,:2]
        test_data=test_data[:,:2]

        #Convert them to self values
        self.train_data=train_data
        self.train_rating=train_rating

        #Creating modified test set
        idx=np.in1d(test_data[:,1],train_data[:,1])
        test_modi=test_data[idx,:]
        test_modi_rating=test_rating[idx]
        idx=np.in1d(test_modi[:,0],train_data[:,0])
        test_modi=test_modi[idx,:]
        test_modi_rating=test_modi_rating[idx]

        rmse_train=[]
        rmse_test=[]
        rmse_test_modi=[]

        User=np.zeros((1,self.num_Item))


        #Initialize where to keep the sum of all the ratings to use in monte_carlo
        self.sum_train_predict_rating=np.empty(np.shape(train_rating)[0])
        self.sum_test_predict_rating=np.empty(np.shape(test_rating)[0])
        self.sum_test_modi_predict_rating=np.empty(np.shape(test_modi_rating)[0])

        #Initializiation of U^0 and V^0, from PMF
        self.User_vector = user_pmf
        self.Item_vector = item_pmf

        #used for burn-in:
        #burn_in_=np.empty((6,0))
        #burn_in_temp=np.empty((6,1))
        #burn_std=np.empty((6,0))
        #columns=user_dim[:,1]#random.sample(range(self.num_User), 3)
        #rows=user_dim[:,0]#random.sample(range(self.dim), 3)
        #V_columns=user_dim[3:,1]#random.sample(range(self.num_Item), 3)
        #V_rows=user_dim[3:,0]#random.sample(range(self.dim), 3)
        #user_dim[:3,1]=U_columns
        #user_dim[:3,0]=U_rows
        #user_dim[3:,1]=V_columns
        #user_dim[3:,0]=V_rows

        #To use for prediction in report
        Harry_potter_predict=[]
        Other_predict=[]

        #The gibbs sampler algorithm, run for epochs+burn_in
        for epoch in tqdm(range(self.epoch+burn_in)):
            #Sample hyperparameters for both User and Item feature vectors
            self.mu_U, self.Lambda_U=self.sample_hyperparameters(self.User_vector, self.num_User)
            self.mu_V, self.Lambda_V=self.sample_hyperparameters(self.Item_vector, self.num_Item)

            #for the t-iteration
            U_new = np.empty((self.dim, self.num_User))
            V_new = np.empty((self.dim, self.num_Item))

            #My try to Paralellize does not work:
            #pool = mp.Pool(mp.cpu_count())
            #with mp.Pool(processes=1) as pool:
            #    U_new=pool.map(self.sample_features_U, range(self.num_User))

            for i in range(self.num_User):
                U_new[:,i]=self.sample_features_U(i)


            self.User_vector=U_new
            
            for j in range(self.num_Item):
                V_new[:,j]=self.sample_features_V(j)

            self.Item_vector=V_new

            #Burn_in
#            for j,i,k in zip(columns, rows, range(6)):
#                if k<3:
#                    burn_in_temp[k,0]=self.User_vector[i,j]
#                else:
#                    burn_in_temp[k,0]=self.Item_vector[i,j]
#
#            if (epoch<10):
#                burn_in_=np.append(burn_in_, burn_in_temp, axis=1)
#            else:
#                burn_in_=burn_in_[:,1:]
#                burn_in_=np.append(burn_in_, burn_in_temp, axis=1)
#                temp=np.std(burn_in_, axis=1, keepdims=True)
#                #print(temp)
#                burn_std=np.append(burn_std, temp, axis=1)
#                #print(burn_std)    



            if epoch>=burn_in:
            #print('Train, epoch: ',epoch,' RMSE: ',self.evaluate(train_data, train_rating, epoch, name='Train' ))
            #print('Test, epoch: ',epoch,' RMSE: ',self.evaluate(test_data, test_rating, epoch, name='Test'))
                rmse_train.append(self.evaluate(train_data, train_rating, epoch-burn_in, name='Train' ))
                rmse_test.append(self.evaluate(test_data, test_rating, epoch-burn_in, name='Test'))
                rmse_test_modi.append(self.evaluate(test_modi, test_modi_rating, epoch-burn_in, name='Test_modi'))

                Harry_potter_predict.append(self.predict(13103,3614))
                list_=[self.predict(13103,6041), self.predict(1155,6041), self.predict(999,6041), self.predict(1309,6041), self.predict(13103,5918), self.predict(1155,5918), self.predict(999,5918), self.predict(1309,5918)]
                Other_predict.append(list_)

                User=((epoch-burn_in)/(epoch-burn_in+1)*User)+((1/(epoch-burn_in+1))*np.dot(self.User_vector[:,13103].T, self.Item_vector))

                if epoch==(self.epoch+burn_in-1):
                    MAE_=[]
                    MAE_.append(mean_absolute_error(train_rating,self.sum_train_predict_rating/(epoch-burn_in+1)))
                    MAE_.append(mean_absolute_error(test_rating,self.sum_test_predict_rating/(epoch-burn_in+1)))
                    MAE_.append(mean_absolute_error(test_modi_rating,self.sum_test_modi_predict_rating/(epoch-burn_in+1)))



        return rmse_train, rmse_test, rmse_test_modi, User, Harry_potter_predict, Other_predict, MAE_#, burn_std, user_dim#,

    def standardize(self, train_data, test_data):
        std=np.std(train_data[:,2])
        mean=np.mean(train_data[:,2])
        Transformed_train=(train_data[:,2]-mean)/std   
        Transformed_test=(test_data[:,2]-mean)/std
        return Transformed_train, Transformed_test

    def predict(self, user_id, item_id):
        r_ij=self.User_vector[:,user_id].T @ self.Item_vector[:,item_id]
        return r_ij

    def evaluate(self, train_index, train_rating, iteration, name='Train'):
        predictions=[]

        for i in range(train_index.shape[0]):
            predictions.append(self.predict(train_index[i,0],train_index[i,1]))

        #Because of monte carlo estimation
        if name=='Train':
            self.sum_train_predict_rating+=predictions
            pred=self.sum_train_predict_rating/(iteration+1)
            RMSE=mean_squared_error(train_rating,pred, squared=False)
        elif name=='Test':
            self.sum_test_predict_rating+=predictions
            pred=self.sum_test_predict_rating/(iteration+1)
            RMSE=mean_squared_error(train_rating, pred, squared=False)
        else:
            self.sum_test_modi_predict_rating+=predictions
            pred=self.sum_test_modi_predict_rating/(iteration+1)
            RMSE=mean_squared_error(train_rating, pred, squared=False)

        return RMSE



if __name__ == "__main__":
    #np.random.seed(0)

    #Using same split of training and test data as in PMF
    train_data=np.load('Data/Train_set.npy')
    test_data=np.load('Data/Test_set.npy')
 

    for dimension, burn_in in zip([30],[50]):
        user_pmf=np.load(f'Data/pred/pmf_UserVector_dim_{dimension}_lambda_5.npy')
        item_pmf=np.load(f'Data/pred/pmf_ItemVector_dim_{dimension}_lambda_5.npy') 

        start_time = time.time()
        bpmf=BPMF(dim=dimension, num_epochs=450)

        rmse_train, rmse_test, rmse_test_modi, User, Harry_potter_predict, Other_predict, MAE_=bpmf.train(train_data=train_data, test_data=test_data, user_pmf=user_pmf, item_pmf=item_pmf, burn_in=burn_in)
        #rmse_train, rmse_test, rmse_test_modi, burn_std, user_dim=bpmf.train(train_data=train_data, test_data=test_data, user_pmf=user_pmf, item_pmf=item_pmf)
        #rmse_train, rmse_test, rmse_test_modi, User=bpmf.train(train_data=train_data, test_data=test_data, user_pmf=user_pmf, item_pmf=item_pmf, burn_in=burn_in)

        print("--- %s seconds ---" % (time.time() - start_time))
        rmse_train=np.array(rmse_train)
        rmse_test=np.array(rmse_test)
        rmse_test_modi=np.array(rmse_test_modi)


        #print(rmse_train)    
        np.save(f'Data/rmse/bpmf_rmse_train_dim_{dimension}_alt.npy',rmse_train) #saves rmse train 
        np.save(f'Data/rmse/bpmf_rmse_test_dim_{dimension}_alt.npy',rmse_test) #saves rmse test
        np.save(f'Data/rmse/bpmf_rmse_test_modi_dim_{dimension}_alt.npy',rmse_test_modi) #saves rmse test


        #Burn_in
        #np.save(f'Data/rmse/bpmf_dim_{dimension}_burn_std_alt.npy',burn_std)
        #np.save(f'Data/rmse/bpmf_dim_{dimension}_user_item_dim_burn_in_alt.npy',user_dim)

        #Savse mean absolute error of only last iteration
        #np.save(f'Data/rmse/bpmf_MAE_dim_{dimension}_al.npy',np.array(MAE_))

        #Saves for prediction:
        np.save(f'Data/pred/bpmf_HP_predict_dim_{dimension}.npy', np.array(Harry_potter_predict))
        np.save(f'Data/pred/bpmf_Other_predict_dim_{dimension}.npy', np.array(Other_predict))
        np.save(f'Data/pred/bpmf_13013_predict_dim_{dimension}.npy', User)