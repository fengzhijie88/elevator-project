#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:34:43 2019

@author: Zhijie
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import collections

def transport(waitingT_,waitingR_,population,popT_,number): #this update the queue after trasporting away "number" of people
    R=len(population) #rounds of waiting
    newpopT_=popT_
    newpopu=population

    while number>0:
        R-=1
        ppl=newpopu[-1]
        if ppl==0: newpopu=newpopu[:-1]
        
        else:
            go=min(ppl,number)
            waitingT_=np.concatenate((waitingT_, popT_[:int(go)]))
            waitingR_=np.concatenate((waitingR_, R*np.ones(go)))
            newpopT_=newpopT_[int(go):]
            newpopu[-1]-=go
            number-=go
            if number>0:newpopu=newpopu[:-1]
       
            
    return waitingT_,waitingR_,newpopu,newpopT_
#initialize fixed model parameters
tauv=1
meantaus=5
F=100
Total=2#number of iteration
result=[]
capacity=1
#ratelimit2=(capacity+1)/(taus*capacity+2*F*tauv+taus)
seednum=68
#caplist=range(10,51)
#for capacity in caplist:
ratelimit=1/(meantaus+2*F*tauv/capacity) #taus=5 #F here need to be fmax
ratelist=np.linspace(0.004,0.004, num=1)
rate=0.004
timemax=2*tauv*F+capacity*6
#for rate in ratelist:
test=[]
test2=[]
for i in range(1000):
 #for _ in range(2):
    seednum+=1
    gen=np.random.RandomState(seednum)
    N=max(int(round(2*rate*F*tauv/(1-rate*meantaus)-1)),1)#start with mean value of no capacity case
    #N=1
    taus=5
    fmax=50
    time=2*tauv*fmax+1*taus
    #initialize data collector lists 
    travelT_=[time]#cycle time of elevators
    waitingN_=[N]#number of people waiting when elevator arrives
    waitingT_=np.array([0])#time from a person arrive to geting on the elevator
    waitingR_=np.array([0])#number of rounds waited before getting on elevator
    extratime_=[]#time the elevator waited until a person arrive
    fmax_=[fmax]#highestfloor the elvator go in each cycle
    population=np.array([N])#new comers inserted in the left hand side of the list, --index is the waiting round
    #popT_=np.array([0])#list saving the waiting time of people in the queue
    N=gen.poisson(rate*time)
    population=np.array([N])
    popT_=gen.uniform(0,time,size=N)
    print(N)
    for i in range(int(Total)):
        #tauv=np.random.uniform(low=0,high=2)
        gen=np.random.RandomState(seednum)
        seednum+=1
       
        if int(N)==0:
            extratime=gen.exponential(1/rate)
            extratime_.append(extratime)
            waitingT_=np.concatenate((waitingT_,np.array([0])))
            waitingR_=np.concatenate((waitingR_,np.array([0])))
            waitingN_.append(1)
            #fmax=gen.random_integers(1,F)
            fmax=50
            fmax_.append(fmax)            
            taus=5
            ########################update the queue
            #taus=gen.uniform(4,6)
            time=2*tauv*fmax+1*taus
            travelT_.append(time)
            newcome=gen.poisson(rate*time)
            N=newcome
            population=np.array([N])
            popT_=gen.uniform(0,time,size=newcome)
            
        else: 
            waitingN_.append(N)
            Naway=min(N,capacity)#number of people taken away by the elevator
            #fmax=np.max(gen.random_integers(1,F,size=Naway))
            fmax=50
            N-=Naway
            waitingT_,waitingR_,population,popT_=transport(waitingT_,waitingR_,population,popT_,Naway)
            fmax_.append(fmax)
            
            #################################update the queue
            #taus=np.mean(gen.uniform(4,6,size=Naway))
            taus=5
            time=2*tauv*fmax+Naway*taus
            travelT_.append(time)
            newcome=gen.poisson(rate*time)
            N+=newcome
            if len(popT_):popT_+=time
            popT_=np.concatenate((popT_,gen.uniform(0,time,size=newcome)))
            population=np.insert(population,0,0)    
            population[0]+=newcome
            
       
        if sum(population)!=len(popT_):print('bug') 
   
    #c=collections.Counter(waitingT_)
    #plt.hist(waitingT_,bins='auto',density=True)
    #result.append([capacity,rate,np.mean(waitingT_),np.mean(travelT_),np.mean(waitingN_),np.mean(fmax_),np.mean(waitingR_)])#timemax*rate])
    test2.append(waitingN_[-2])
    test.append(waitingT_[-1]) 
    c=collections.Counter(test)