
# coding: utf-8

# In[1]:


from PIL import Image
import glob
import numpy as np


# In[2]:


def one_hot(Y, num_classes):
        tem = []
        for i in range(Y.shape[0]):
            tem.append(np.eye(num_classes)[:,Y[i]].reshape(-1,1))
        return np.array(tem).reshape(np.array(tem).shape[0],-1)

    
def loadd(path):
    all_files = glob.glob(path + '/*/*.jpg')
    X = []
    Y = []
    for i in all_files:
        #read in grayscale and convert to numpy array
        X.append(np.array(Image.Image.convert(Image.open(i), 'L')))
        Y.append(int(i.split('/')[2]))
    
    X[71]= X[71][:438,:]
    X = np.array(X).reshape(np.array(X).shape[0],-1)
    Y = one_hot(np.array(Y),14)
    return X, Y




# In[3]:


path_train = 'un_resized_images/fidels'
path_test = 'un_resized_images/fidels_test/'
X, Y = loadd(path_train)
Xt, Yt = loadd(path_test)


print(X.shape, Y.shape)
print(Xt.shape, Yt.shape)


# In[321]:


class log_model():
    def __init__(self, x, num_classes):
        self.weights = np.zeros((x.shape[1],num_classes))
        self.biases = np.zeros((num_classes,1))
    
 
    def forward(self, x):
        #print('X:',x.shape, 'w:',self.weights.shape, 'b:',self.biases.shape)
        ret = np.matmul(self.weights.T, x.T) + self.biases
        #print("forward: ", ret.shape)
        return ret 
    
    
    def softmax(self, forward):
        forward -= np.max(forward, axis=0)
        ret = np.exp(forward) / (np.sum(np.exp(forward), axis=0))
        #print("softmax: ", ret.shape)
        return ret
    
    
    def loss(self, softmax, y):
        ret = -np.sum(np.multiply(y.T,np.log(softmax+0.001)))
        #print("loss: ", ret)
        
        return ret
    
    
    def optimize(self, a, x, y):
        #print("soft:",a.shape, "y:", y.shape)
        derivative = np.subtract(a.T, y)
        #print("der:",derivative.shape)
        #print('dw.shape',np.dot(x.T,derivative).shape)
        dw = np.dot(x.T,derivative)
        #print('db.shape',np.sum(derivative, axis=0).shape)
        db = np.sum(derivative, axis=0).reshape(-1,1)
        #print("dw:\n ", dw)
        #print("db: \n", db)
        
        return dw, db 
  

    def update(self,lr, dw, db):
        #print(loss)
        self.weights -= np.multiply(lr,dw)
        self.biases -= np.multiply(lr,db)
         #print("updated w: \n", self.weights)
        #print("updated b: \n", self.biases)
    
    def train(self, X, Y, lr, iteration):
        accuracy= 0
        for i in range(iteration):
            forward = self.forward(X)
            softmax = self.softmax(forward)
            loss = self.loss(softmax, Y)
            dw, db = self.optimize(softmax, X, Y)
            self.update(lr,dw, db)
            print('iteration:', i,'--loss=', loss)
            
            accuracy = np.equal(np.argmax(Y.T, axis=-1), np.argmax(softmax, axis=-1)).mean()
            
        #print('w:',self.weights, 'b:', self.biases)
        print('Train accuracy: ', accuracy*100, '%')
         
        return self.weights, self.biases
    def test(self, Xt, Yt):
        #print(Xt.shape, Yt.T.shape)
        #Xt = Xt[:2,:]
        #Yt = Yt[:2,:]
        #print(Xt.shape, Yt.T.shape)
        accuracy = 0
        prediction = self.softmax(self.forward(Xt)).T
        #print(Yt.shape, prediction.shape)
        accuracy = np.equal(np.argmax(Yt, axis=-1), np.argmax(prediction, axis=-1)).mean()
        print('Done!')
        print('Accuracy: ', accuracy*100, '%')
   
    


# In[322]:


num_classes = 14
log = log_model(X, num_classes)
w,b = log.train(X, Y, 0.002, 6)


# In[323]:


log.test(Xt, Yt)

