import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


class PMF():
    def __init__(self, dim=10):
        self.dim=dim #number of latent features
        self.epoch=None #number of epoch before stop

        self.Item_vector = None
        self.User_vector = None

        self.num_User = None
        self.num_Item = None

        self.Item_lambda = None
        self.User_lambda = None



    def initialize_parameters(self, lambda_Item,lambda_User, no_epochs):
        #initilization of parameters
        self.Item_lambda=lambda_Item
        self.User_lambda=lambda_User
        self.epoch=no_epochs
        
        #Initialization of user and item vector as N(0,1/lambda)
        self.User_vector = np.random.normal(0.0, 1.0/lambda_User,(self.dim, self.num_User))
       
        self.Item_vector = np.random.normal(0.0, 1.0/lambda_Item, (self.dim, self.num_Item))
    

    def update_parameters(self, train_index, train_rating):
        #Update parameters in U and V using MAP
        for i in range(self.num_User):
            idx=(train_index[:,0]==i)
            V_j = self.Item_vector[:,train_index[idx,1]]
            self.User_vector[:, i] = np.dot(np.dot(train_rating[idx],V_j.T)\
                ,np.linalg.inv(np.dot(V_j,V_j.T)+np.identity(self.dim)*self.User_lambda))
           
        for j in range(self.num_Item):
            idx=(train_index[:,1]==j)
            U_i=self.User_vector[:,train_index[idx,0]]
            self.Item_vector[:,j] = np.dot(np.dot(train_rating[idx],U_i.T)\
                ,np.linalg.inv(np.dot(U_i,U_i.T)+np.identity(self.dim)*self.Item_lambda))

    
    def predict(self, user_id, item_id):
        #Predicting rating from user i of itemj
        r_ij=self.User_vector[:,user_id].T @ self.Item_vector[:,item_id]
        return r_ij

    
    def evaluate(self, train_index, train_rating):
        #Evaluates the models prediction on either train og test set
        predictions=[]

        for i in range(train_index.shape[0]):
            predictions.append(self.predict(train_index[i,0],train_index[i,1]))

        return mean_squared_error(train_rating, predictions, squared=False)

    def standardize(self, train_data, test_data):
        #Transforming train and test data, using standardization og train
        std=np.std(train_data[:,2])
        mean=np.mean(train_data[:,2])
        Transformed_train=(train_data[:,2]-mean)/std   
        Transformed_test=(test_data[:,2]-mean)/std
        return Transformed_train, Transformed_test
    

    def fit(self, n_epochs, train_data, test_data, lambda_):
        #Training or fit of the model

        #Finding max numbers of user, and items being either in the test or train data
        self.num_User = int(max(np.amax(train_data[:, 0]), np.amax(test_data[:, 0])))+1
        self.num_Item = int(max(np.amax(train_data[:, 1]), np.amax(test_data[:, 1])))+1

        train_rating, test_rating = self.standardize(train_data, test_data)

        #Creating modified testset that do not contain cold-start samples
        idx=np.in1d(test_data[:,1],train_data[:,1])
        test_modi=test_data[idx,:]
        test_modi_rating=test_rating[idx]
        idx=np.in1d(test_modi[:,0],train_data[:,0])
        test_modi=test_modi[idx,:]
        test_modi_rating=test_modi_rating[idx]

        self.initialize_parameters(lambda_, lambda_, n_epochs)
        rmse_train=[]
        rmse_test=[]
        rmse_test_modi=[]
        #loss=[]

        rmse_train.append(self.evaluate(train_data, train_rating))
        rmse_test.append(self.evaluate(test_data, test_rating))
        rmse_test_modi.append(self.evaluate(test_modi, test_modi_rating))

        #the pmf algorithm run for n epochs:
        for i in tqdm(range(self.epoch)):
            self.update_parameters(train_data, train_rating)
            #loss.append(self.loss_function(train_data))

            #if (i+1) % 10 == 0 :
            #print('current epoch: ',i, ' no of epochs: ',n_epochs )
        
            rmse_train.append(self.evaluate(train_data, train_rating))

            rmse_test.append(self.evaluate(test_data, test_rating))

            rmse_test_modi.append(self.evaluate(test_modi, test_modi_rating))
            
            #loss.append(self.loss_function(train_data, train_rating))
            #print('Loss function', i + 1, ':', loss[i])
        
        return rmse_train, rmse_test, rmse_test_modi, self.Item_vector,  self.User_vector


def split_rating_data(data, size=0.2):
    train_data=np.empty((0,3), int)
    test_data=np.empty((0,3),int)
    rand=np.random.rand(data.shape[0])
    number=0

    for line in data:
        if rand[number] < size:
            test_data = np.append(test_data, line.reshape(1,-1), axis=0)
        else:
            train_data = np.append(train_data, line.reshape(1,-1), axis=0)
        number+=1

    return train_data, test_data,


        
if __name__ == "__main__":
    np.random.seed(0)
    #Random seed for split of test and training set
    #data=np.load('Data/small_test.npy')



    data=np.load('Data/pref_tuple.npy')
    size=[0.2]# [0.4, 0.6, 0.8]
    dimension=30
    #dimension=30
    #lambdas_=[4.0, 4.5, 5.5, 6.0]
    lambda_=5

    #Running the algorithm testing for different.
    for s in size:
        train_data, test_data=split_rating_data(data, size=s)
        #np.save('Data/Test_set.npy', test_data)
        #np.save('Data/Train_set.npy', train_data)

        std=np.std(train_data[:,2])
        mean=np.mean(train_data[:,2])
        param=np.array([std, mean])
        np.save(f'Data/pred/pmf_transformation.npy', param) #saves the transformation data, for predicitiong

 
    
        start_time = time.time()
        pmf = PMF(dim=dimension)
        #Training the model
        rmse_train, rmse_test, rmse_test_modi, V, U= pmf.fit(n_epochs=60, train_data=train_data, test_data=test_data, lambda_=lambda_)
        print("--- %s seconds ---" % (time.time() - start_time))
        rmse_train=np.array(rmse_train)
        rmse_test=np.array(rmse_test)
        rmse_test_modi=np.array(rmse_test_modi)
        #prediction=np.array(prediction).reshape(-1,1)
        #res=np.append(train_data, prediction, axis=1)
        
        np.save(f'Data/rmse/pmf_rmse_train_dim_{dimension}_lambda_{lambda_}_train_size_{s}.npy',rmse_train) #saves rmse train 
        np.save(f'Data/rmse/pmf_rmse_test_dim_{dimension}_lambda_{lambda_}_train_size_{s}.npy',rmse_test) #saves rmse test
        np.save(f'Data/rmse/pmf_rmse_test_modi_dim_{dimension}_lambda_{lambda_}_train_size_{s}.npy',rmse_test_modi) #saves rmse on modified test set
    
        np.save(f'Data/pred/pmf_UserVector_dim_{dimension}_lambda_{lambda_}.npy', U) #saves the user feature matrix
        np.save(f'Data/pred/pmf_ItemVector_dim_{dimension}_lambda_{lambda_}.npy', V) #saves the item latent feature matrix


   
    





