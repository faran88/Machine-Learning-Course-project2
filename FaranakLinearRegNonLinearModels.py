# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:30:42 2020

@author: faranak abri
"""
import numpy as np
import matplotlib.pyplot as plt 
"""------------------------------------------------------"""
def LinearRegressionNonLinearModel(n_train=10):
    
    x_train=np.random.uniform(0,1,n_train)
    t_train=np.sin(2*np.pi*x_train)+np.random.normal(0,0.3,n_train)
    
    
    n_test=100
    x_test=np.random.uniform(0,1,n_test)
    t_test=np.sin(2*np.pi*x_test)+np.random.normal(0,0.3,n_test)
    
    x=np.ones((n_test,2))
    x[:,1]=x_test
    x_test=x
    
    x=np.ones((n_train,2))
    x[:,1]=x_train
    x_train=x
    
    E_train=np.zeros(10)
    E_test=np.zeros(10)
    
    for degree in range (10):
        phi_train=np.ones((n_train,degree+1))
        phi_test=np.ones((n_test,degree+1))
        
        for i in range (1,degree+1):
            phi_train[:,i]=pow(x_train[:,1],i)
            phi_test[:,i]=pow(x_test[:,1],i)
        
        w=(np.linalg.pinv(phi_train.T@phi_train))@(phi_train.T@t_train)
        y_train=phi_train@w
        y_test=phi_test@w
        
        error_train=(sum(abs(t_train-y_train)))/n_train
        error_test=(sum(abs(t_test-y_test)))/n_test   
        
        J_train=((phi_train@w)-t_train).T@((phi_train@w)-t_train)
        E_train[degree]=np.sqrt(J_train/n_train)
        J_test=((phi_test@w)-t_test).T@((phi_test@w)-t_test)
        E_test[degree]=np.sqrt(J_test/n_test)
        
    
    
    #E_train=(E_train - np.min(E_train))/(np.max(E_train) - np.min(E_train))
    #E_test=(E_test - np.min(E_test)) / (np.max(E_test) - np.min(E_test))
    E_train=E_train/np.max(E_test)
    E_test= E_test/np.max(E_test)
    
    plt.figure()
    plt.title("Linear Regression with Polynomial models - N_train=%i" %n_train)
    plt.xlabel("M(degree)")
    plt.ylabel("E_RMS")
    plt.plot(E_train,'o-',color='blue', label="Train")
    plt.plot(E_test,'o-',color='red', label="Test")
    plt.legend(loc='upper') 
    plt.show()
"""------------------------------------------------------"""    
LinearRegressionNonLinearModel(10)
LinearRegressionNonLinearModel(100)