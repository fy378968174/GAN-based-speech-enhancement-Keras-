# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:02:48 2017

@author: FY
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 23:09:12 2017

@author: FY
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:56:19 2017 

@author: FY
"""
import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD,Adam,RMSprop
from scipy.io.wavfile import read, write
from keras.models import Model, Sequential
from keras.layers import Convolution1D, AtrousConvolution1D,Input, Lambda, merge
from keras.layers.merge import Concatenate
from keras.layers import Conv1D,Conv2D,Conv2DTranspose,Reshape
from scipy import io
from keras.layers import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers import BatchNormalization
from keras import backend as K   #用于定义自己的cost function
import matplotlib.pyplot as plt
import get_train_data
# Changes the traiable argument for all the layers of model
# to the boolean argument "trainable"
def make_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable
def D_loss(y_true,y_pred):
    return 0.5*K.mean((y_pred-y_true)**2,axis = -1)
def GAN_loss(fake_output,true_input):
    def lossfun(y_true, y_pred):
        return 0.5*K.mean((((y_pred-y_true)**2)),axis = -1)+100*K.mean(K.abs(fake_output-true_input))
    return lossfun
def G_loss(fake_output,true_input):
    def lossfun(y_true, y_pred):
        return 1*K.mean(K.abs(fake_output-true_input)) 
    return lossfun
optim =RMSprop(lr=0.0002)
#optim =SGD(lr=0.0005)
# --------------------Generator Model--------------------
#该部分有输入有两个输入，inputs是noisy信号，是需要经过神经网络进行增强的信号
#另一个输入inputs1是clean信号，该部分信号仅参与loss function的计算
inputs=Input(shape = (1024,1))  #noisy
inputs1=Input(shape =(1024,1))  #clean
input=([inputs,inputs1])
# encoder
reshape=(Reshape((1024,1,1),input_shape=(1024,1)))(inputs)
cov1=(Conv2D(64, 31, strides=4,padding='same'))(reshape)
cov1=(PReLU())(cov1)    
cov2=(Conv2D(128, 31, strides=4,padding='same'))(cov1)
cov2=(PReLU())(cov2)
cov3=(Conv2D(256, 31, strides=4,padding='same'))(cov2)
cov3=(PReLU())(cov3)
# decoder
# self.G.add(Conv2DTranspose(1024,31,strides=1,padding='same'))
# self.G.add(PReLU())
cov4=(Conv2DTranspose(256,31, strides=(1,1),padding='same'))(cov3)
cov4=(PReLU())(cov4)
z1 = merge([cov3,cov4], mode='sum')
cov5=(Conv2DTranspose(128,31, strides=(4,1),padding='same'))(z1)
cov5=(PReLU())(cov5)
z2=merge([cov2,cov5], mode='sum')
cov6=(Conv2DTranspose(64,31, strides=(4,1),padding='same'))(z2)
cov6=(PReLU())(cov6)
z3=merge([cov1,cov6], mode='sum')
cov7=(Conv2DTranspose(16,31, strides=(4,1),padding='same'))(z3)
cov7=(PReLU())(cov7)
cov8=(Conv2DTranspose(1,31, strides=(1,1),activation='tanh',padding='same'))(cov7)
#cov8=(PReLU())(cov8)
cov8=(Reshape((1024,1)))(cov8)
G = Model([inputs,inputs1],output = cov8)
G.compile(loss=G_loss(cov8,inputs1),optimizer=optim)
#G.compile(loss='mse',optimizer=optim)
G.summary()
# --------------------Discriminator Model--------------------
inputs=Input((1024,2))   #condition GAN
# encoder
#model.add(Reshape((16384,1,1),input_shape=input_shape2))
d1=(Conv1D(64, 31, strides=4,padding='same'))
d_hidden1=d1(inputs)
d2=(BatchNormalization())
d_hidden2=d2(d_hidden1)
d3=(LeakyReLU(alpha=0.3))
d_hidden3=d3(d_hidden2)
d4=(Conv1D(128, 31, strides=4,padding='same'))
d_hidden4=d4(d_hidden3)
d5=(BatchNormalization())
d_hidden5=d5(d_hidden4)
d6=(LeakyReLU(alpha=0.3))
d_hidden6=d6(d_hidden5)
d7=(Conv1D(256, 31, strides=4,padding='same'))
d_hidden7=d7(d_hidden6)
d8=(BatchNormalization())
d_hidden8=d8(d_hidden7)
d9=(LeakyReLU(alpha=0.3))
d_hidden9=d9(d_hidden8)
d10=(Conv1D(1, 31, strides=1,padding='same'))
d_hidden10=d10(d_hidden9)
d11=(BatchNormalization())
d_hidden11=d11(d_hidden10)
d12=(LeakyReLU(alpha=0.3))
d_hidden12=d12(d_hidden11)
d13=(Flatten())
d_hidden13=d13(d_hidden12)
d14=Dense(16,activation='sigmoid')   
d_output =d14(d_hidden13)

D= Model(input = inputs,output=d_output)
D.compile(loss=D_loss, optimizer=optim)
#D.compile(loss='mse', optimizer=optim)
D.summary()
# --------------------GAN Model--------------------
make_trainable(D,False)
inputs=Input(shape = (1024,1))   #noisy
inputs1=Input(shape = (1024,1))  #clean

input=([inputs,inputs1])
g_output = G(input)
d_input = Concatenate()([g_output,inputs])
#d_input=np.concatenate((g_output,inputs))
gan_hidden = d1(d_input)
gan_hidden = d2(gan_hidden)
gan_hidden = d3(gan_hidden)
gan_hidden = d4(gan_hidden)
gan_hidden = d5(gan_hidden)
gan_hidden = d6(gan_hidden)
gan_hidden = d7(gan_hidden)
gan_hidden = d8(gan_hidden)
gan_hidden = d9(gan_hidden)
gan_hidden = d10(gan_hidden)
gan_hidden = d11(gan_hidden)
gan_hidden = d12(gan_hidden)
gan_hidden = d13(gan_hidden)
gan_output = d14(gan_hidden)

GAN =Model([inputs,inputs1],output=gan_output) 
GAN.compile(loss=GAN_loss(g_output,inputs1), optimizer=optim) 
GAN.summary()
G.load_weights('cnn_generator_weights.h5')
# --------------------Main Code--------------------
if __name__ == '__main__':

    batch_size=114
    n_epochs = 50
    n_minibatches = 10
    
    for i in range(n_epochs):
            print ('Epoch:', i+1)
            for index in range(n_minibatches):
                # --------------------load data--------------------
                (noisy_batch,real_batch)=get_train_data.get_train_data_for_gan()
                
                noisy_batch1=np.reshape(noisy_batch,(batch_size,1024,1))
                real_batch=np.reshape(real_batch,(batch_size,1024,1))
                
                combined_G_batch=([noisy_batch1,real_batch])
                
                fake_batch = G.predict(combined_G_batch)
                
                fake_batch=np.concatenate((fake_batch,noisy_batch1)) # condition GAN
                real_batch=np.concatenate((real_batch,noisy_batch1)) # condition GAN
                fake_batch=np.reshape(fake_batch,(batch_size,1024,2))
                real_batch=np.reshape(real_batch,(batch_size,1024,2))
                print('--------------------enhanced speech Generated!--------------')
                combined_X_batch = np.concatenate((real_batch, fake_batch))
                one_label=np.ones([1*batch_size, 16])
                zero_label=np.zeros([1*batch_size, 16])
                combined_y_batch =np.vstack((one_label,zero_label))
    
                make_trainable(D,True)
                combined_X_batch=np.reshape(combined_X_batch,(2*batch_size,1024,2))
                d_loss = D.train_on_batch(combined_X_batch, combined_y_batch)
                print('--------------------Discriminator trained!------------------')
                print(d_loss)
                
                make_trainable(D,False)
                g_loss = GAN.train_on_batch(combined_G_batch,one_label)
                print('--------------------GAN trained!----------------------------')
                print(g_loss)
        
    G.save_weights('cnn_generator_weights.h5') #保存最后的模型或权重   
    G.save('cnn_generator_model.h5') 

    (test_noisy,test_clean)=get_train_data.get_train_data_for_gan()   

    test_noisy=np.reshape(test_noisy,(114,1024,1))
    test_clean=np.reshape(test_clean,(114,1024,1))
    test=([test_noisy,test_clean])
    test_gend_audio = G.predict(test) 
    test_gend_audio=np.reshape(test_gend_audio,(114*1024))
    test_clean=np.reshape(test_clean,(114*1024))
    test_noisy=np.reshape(test_noisy,(114*1024))
    test_gend_audio=test_gend_audio/np.max(test_gend_audio)
    plt.subplot(311)
    plt.plot(test_noisy)
    plt.title('noisy')
    plt.subplot(312)
    plt.plot(test_clean)
    plt.title('clean')
    plt.subplot(313)
    plt.plot(test_gend_audio)
    plt.title('enhanced')
    plt.show()
    
    plt.subplot(211)
    plt.plot(test_noisy)
    plt.title('noisy')
    plt.subplot(212)
    plt.plot(test_gend_audio)
    plt.title('enhanced')
    plt.tight_layout()
    plt.show()