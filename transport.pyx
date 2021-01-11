#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:05:12 2019

@author: Zhijie
"""
import numpy as np
def transport(waitingT_,waitingR_,population,popT_,int number): #this update the queue after trasporting away "number" of people
    R=len(population) #rounds of waiting
    newpopT_=popT_
    newpopu=population
    waitingT_+=popT_[:number]
    while number>0:
        R-=1
        ppl=newpopu[-1]
        if ppl==0: newpopu=newpopu[:-1]
        
        else:
            go=int(min(ppl,number))
            #waitingR_=np.concatenate((waitingR_, R*np.ones(go)))
            waitingR_[R]+=go
            newpopT_=newpopT_[go:]
            newpopu[-1]-=go
            number-=go
            if newpopu[-1]==0:newpopu=newpopu[:-1]
       

    return waitingT_,waitingR_,newpopu,newpopT_
