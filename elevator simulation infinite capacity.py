#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from fast_histogram import histogram1d
import collections
import math
from scipy.stats import binned_statistic
import scipy


# In[2]:


############## input parameters ################
tauv=1
taus=5
Cap=50 #the capacity of each elevator, not using "capacity" to prevent confusion by looping variable

F=100
n=1

####### Input number of iteration ####################

Total=100000 #number of iteration
seednum=25
######################################
def binning(x,y,size):
    newx=[]
    newy=[]
    for i in range(int(len(y)/size)):
        newx.append(sum(x[(i-1)*size:(i)*size])/size)
        newy.append(sum(y[(i-1)*size:(i)*size])/size)
    return np.array(newx),np.array(newy)

    
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
        

def histo(datalist,labelstring): # all the histogram plot with same style
    up=max(datalist)
    binn=int(up/3)
    plt.scatter(np.linspace(0,up,num=binn),1/3*histogram1d(np.float64(datalist),bins=binn,range=[0,up])/len(datalist),label=labelstring,c='g',zorder=10,s=5)

def model_func(t, A, K, C): #fitted exponential distribution
    return A * np.exp(K * np.array(t)) + C

def fit_exp_linear(t, y, C=0): #fit data with exponential distribution
    y = y - C
    y = np.log(y)
    #K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K, C

def linear(x,m,b):
    return m*x+b


def plotN(waitingN_,fit=True,binsize=5): # for plotting discrete distribution, better than histogram 
    c=collections.Counter(waitingN_)
    sortedc=list(c)
    sortedc.sort()
    #up=np.max(waitingN_)
    x=[]
    y=[]
    for item in sortedc:
        x.append(item)
        y.append(c[item])
    x=np.array(x)
    y=np.array(y)
    low=int(max(x)/3)
    up=int(max(x)-max(x)/5)
    newx,newy=binning(np.array(x),np.array(y),5)
    if binsize:plt.scatter(newx,newy/len(waitingN_),label="simulation",c='g',s=5,zorder=10)
    else: plt.scatter(x,np.array(y)/len(waitingN_),label="simulation",c='g',s=5,zorder=10)
    


ratelimit=1/taus

############## input the rates to be shown ##############
#ratelist=np.linspace(ratelimit*0.5,ratelimit*0.5,num=1)
#ratelist=np.linspace(0.08948637489087131,0.08948637489087131,num=1) #0.8
#ratelist=[0.08948637, 0.09619785, 0.10290933, 0.10962081] #0.8 - 0.98, 4, cap 50


# In[3]:


rate=ratelimit*0.5
from scipy.stats import poisson

def poisson_probability(actual, mean):
    p = math.exp(-mean)
    for i in range(actual):
        p *= mean
        p /= i+1
        if p>1:print('stop')
    return p
    
def Remove(duplicate): #remove duplicates
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

#timerange=1000
Nrange=range(0,200)
N_0=N=int(round(2*rate*F*tauv/(1-rate*taus)-1))
#N_0=1
def Pfmax(x,n):
    y=(x-n*taus)/2/tauv
    if y<1 or y>F or not y.is_integer(): return 0
    else:
        prob=math.pow((y/F),n)-math.pow((y-1)/F,n)
        if prob>1:print('stop')
        return(prob)
def PN(n,P_T,i):#i is the iteration
    pn=0
    for ti in range(len(timerange)):
        pn+=poisson.pmf(n, rate*timerange[ti])*P_T[i][ti]
    return pn    

def PT(t,P_N,i):
    pt=0
    for ni in range(len(Nrange)):
        pt+= Pfmax(t,Nrange[ni])*P_N[i][ni]
    return pt  
P_T=[]
P_N=[]#this list should eventually contain 1 more element
P_T0=[]
P_N0=[]
#initialize, setup the domain of time
timelist=[]
for fmax in range(F):
    for N in Nrange:
        timelist.append(2*tauv*fmax+N*taus)
timelist.sort()
timerange=Remove(timelist) #this solve the problem of not normalized
'''
for t in range(timerange):
    P_T0.append(Pfmax(t,N_0))
P_T.append(P_T0)
'''
'''
for n in Nrange:
    #P_N0=list(np.zeros(490))
    P_N0.append(poisson.pmf(n, N_0))
      
P_N.append(P_N0)
'''

for n in Nrange:
    if n!=N_0:
        P_N0.append(0)
    else: P_N0.append(1)    
P_N.append(P_N0)


#iteration
Total=4
for i in range(Total):
    print(i)
    P_Ti=[]
    P_Ni=[]
    #P_Ni=list(np.zeros(490))
    
    for t in timerange:
        P_Ti.append(PT(t,P_N,-1))
        
    P_T.append(P_Ti) 
    
    
    for n in Nrange:
        P_Ni.append(PN(n,P_T,-1))
   
    P_N.append(P_Ni)
    
       


# In[ ]:


travelT_=[]
waitingT_=np.array([])
waitingN_=[]
extratime_=[]
fmax_=[]
extratime=0
time=0
N=int(2*rate*F*tauv/(1-rate*taus)-1)#start with mean value
for i in range(int(Total)):
    gen=np.random.RandomState(seednum)
    seednum+=1
    if N==0:#if N=0, shift realtime until next person comes
        extratime=gen.exponential(1/rate)
        extratime_.append(extratime)
        N=1      

    waitingN_.append(N)
    waitingT_=np.concatenate((waitingT_,gen.uniform(0,time,size=N)))
    fmax=np.max(gen.random_integers(1,F,size=N))
    fmax_.append(fmax)
    ################################################
    taus=2.5
    time=2*tauv*fmax+N*2*taus
    travelT_.append(time)
    N=gen.poisson(rate*time)
            
#####################Saving data#################
    #np.savetxt('/Users/Zhijie/Desktop/new_data/'+str(round(rate/ratelimit,2))+'_n'+str(n)+'_Cinf_waitingN_.dat', waitingN_)
    #np.savetxt('/Users/Zhijie/Desktop/new_data/'+str(round(rate/ratelimit,2))+'_n'+str(n)+'_Cinf_waitingT_.dat', waitingT_)
    #np.savetxt('/Users/Zhijie/Desktop/new_data/'+str(round(rate/ratelimit,2))+'_n'+str(n)+'_Cinf_travelT_.dat', travelT_) 


# In[4]:


ratelist=[0.1]#redundant
plt.figure(figsize=(4.3,3))
n=1
color=['purple','b',_,'orange']
plt.plot([0,39,39.0001],[0,1,0],label="iteration 0",linewidth=1,c=color[0])
for iteration in [1,3]:
    plt.plot(Nrange,P_N[iteration],label="iteration "+str(iteration),linewidth=iteration+0.5,c=color[iteration])  
    


plt.xlabel(r"$N$")
plt.ylabel(r"$Q(N)$")

plt.ylim(0.0001,2)
plt.xlim(12,83)

xpoi=range(10,80)
plt.plot(xpoi,scipy.stats.norm.pdf(xpoi,39,7.21),label="Gaussian",linestyle='--',zorder=9,c='r')
for rate in ratelist:
    waitingN_=np.loadtxt('/Users/Zhijie/Desktop/new_data/'+str(round(rate/ratelimit,2))+'_n'+str(n)+'_Cinf_waitingN_.dat')
    plotN(waitingN_,0,0)

plt.yscale('log')
plt.legend()
plt.savefig("passenger number.pdf",bbox_inches='tight')


# In[6]:


ratelist=[0.1]
#plt.figure(figsize=(4*1.1,3*1.1))
plt.figure(figsize=(4.3,3))
n=1
color=['purple','b',_,'orange']
for iteration in [0,1,3]:
    x,y=binning(timerange,P_T[iteration],5)
    plt.plot(x,y,label="iteration "+str(iteration),linewidth=iteration+1,zorder=iteration,c=color[iteration])   


for rate in ratelist:
    #waitingN_=np.loadtxt('/Users/Zhijie/Desktop/new_data/'+str(round(rate/ratelimit,2))+'_n'+str(n)+'_Cinf_waitingN_.dat')
    travelT_=np.loadtxt('/Users/Zhijie/Desktop/new_data/'+str(round(rate/ratelimit,2))+'_n'+str(n)+'_Cinf_travelT_.dat')
    #plotN(travelT_,0,binsize=5)
    histo(travelT_,labelstring="simulation")
    #plt.hist(travelT_,bins="auto",density=True,histtype="step")

x=np.linspace(260,530,500)
plt.plot(x,scipy.stats.norm.pdf(x,390,36.5),label="Gaussian",linestyle='--',zorder=11,c='r')

plt.xlabel(r"$T$")
plt.ylabel(r"$P(T)$")


plt.xlim(270,570)
plt.ylim(0.0001,0.2)
plt.yscale('log')
plt.legend()
plt.savefig("cycle time.pdf",bbox_inches='tight')


# In[12]:


##waiting time distribution
def W(t,pt):
    w=0
    for T in range(len(timerange)):
        if timerange[T]>=t: w+=pt[T]/timerange[T]
    
    return w

def G(t,pt):
    w=0
    for T in range(500):
        if T>=t: w+=pt[T]/T
    return w

plt.figure(figsize=(4,3))
n=1
for rate in ratelist:
    waitingT_=np.loadtxt('/Users/Zhijie/Desktop/new_data/'+str(round(rate/ratelimit,2))+'_n'+str(n)+'_Cinf_waitingT_.dat')
    histo(waitingT_,labelstring="simulation")
    
    
for iteration in [1,2,3]:
    x=[]
    W_T=[]
    for time in range(timerange[-1]):
        x.append(time)
        W_T.append(W(time,P_T[iteration]))
        print(sum(W_T[1:]))

    plt.plot(x,W_T,label="iteration "+str(iteration))    

x=np.linspace(0,700,700)
Gaussian=scipy.stats.norm.pdf(x,390,36.5)
x=[]
W_T=[]

for time in range(timerange[-1]):
        x.append(time)
        W_T.append(G(time,Gaussian))

#print(sum(W_T[1:]))
plt.plot(x,W_T,label="Gaussian") 
    
plt.xlabel("waiting time")
plt.ylabel("probability")
plt.xlim(0,500)
plt.ylim(-0.0001,0.0028)
plt.legend()
plt.savefig("waiting time iteration.pdf",bbox_inches='tight')

