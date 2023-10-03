print("""
      Charles Hughes-Fagan - 20334056 - CA2 Statistical Thermodynamics
      charlesh@tcd.ie
      This Program takes a few minutes ! 
      """)




import numpy as np
import matplotlib.pyplot as plt
import math
import random



N=1000
m = 1         
M = 100
k_B = 1
T_0 = 1       
F = 10       
t=0



x_v_state = np.zeros([N,2])    #state[ith particle][x,or v] of 





def eq_x(F,N,T):
    """
    From PV =FX = n*k_b*T
    """
    return (N * k_B * T)/F

X_0 = 2*(eq_x(F, N, T_0))




    
"""
each particle has a random x, between (-x0,x0) and random v with standard 
deviation v =sqrt(kb*t/m), from KE = 1/, average is 0 since we assume equal 
amounts going both ways.
"""




#print(random.gauss(0, np.sqrt((k_B*T_0)/m)))

x_v_state[:,0] = np.random.uniform(-X_0, X_0, N)   #rand_x
#print(x_v_state[:,0])
x_v_state[:,1] = np.random.normal(0, np.sqrt((k_B*T_0)/m),N) # rand v
#print(x_v_state[:,:])





#plt.plot(x_v_state[:,1],x_v_state[:,0])

def tau(x,v,X,V):
    if v >= 0:
        return (M/F)*(V-v+ np.sqrt(((V-v)**2)-2*(F/M)*(x-X)))
    else:
        return (M/F)*(V+v+ np.sqrt(((V+v)**2)+2*(F/M)*(x+X)))




def x_new(x,v,t):
    return (x+v*t)


def V_new(V,t):
    return (V-((F/M)*t))


def vel_post_coll(v,V):
    if v >= 0:
        W = (2*m*v+(M-m)*V)/(m+M)
        w = (2*M*V+(m-M)*v)/(m+M)
    elif v < 0:
        W = (-2*m*v+(M-m)*V)/(m+M)
        w = (-2*M*V+(m-M)*v)/(m+M)
    return [w,W]




t=0




x = x_v_state[:,0]
x = np.random.uniform(-X_0, X_0, N) 
v = x_v_state[:,1]
v =  np.random.normal(0, np.sqrt((k_B*T_0)/m),N)
X = X_0
V = 0 # rand v



X_arr = []
v_arr =[]
t_arr = []
t = 0


#final loop




v_av = np.average(abs(v))    
    
#print(v)
tempature_ini =    m/k_B * v_av**2


z = [] # array to hold last few elements of X
wei = [] # array to hold weighting of last few elements of X


for i in range(10*N):
    """
    find min tau
    """
    tau_array =[]
    for i in range(len(x)):    
        tau_array.append(tau(x[i], v[i], X, V))
        
    min_tau = min(tau_array)
    
    min_tau_index = tau_array.index(min_tau) # index of particle with shortest time till collision
    
    #increase time by min_tau
    
    V = V_new(V,min_tau)
    for i in range(len(x)):
        
        x[i] = x_new(x[i],v[i],min_tau)
        
    

    
    t = t+ min_tau     
    #print(max(x), "and ", abs(x[min_tau_index]) , min_tau ) 
    
    X = abs( abs(x[min_tau_index]) )
    
    if (i> (99/100)*N):
        z.append(X)
        wei.append(min_tau)
    
    #print("Piston at" , X)
    W = vel_post_coll(v[min_tau_index],V)[1] 
    w = vel_post_coll(v[min_tau_index],V)[0]
    
    V = W
    X_arr.append(X)
    t_arr.append(t)
    
    v[min_tau_index] = w

    
    
    #print("X is" , X, "after ", t)
    
    #print(x_v_state[10,1])
    
    
    

print(v_av)
#tempature_apre =    m/k_B * v_av**2
#print(tempature_apre/tempature_ini)
    
"""
    From PV/T = FX/T = FX_0/T_0
"""
    

X_bar = np.average(z, weights=wei)

print("X_bar is equal to ", X_bar)

eq_x_line = np.zeros(len(t_arr))+166.7



plt.plot(t_arr,X_arr , label = "Displacement of Piston")
plt.plot(t_arr,eq_x_line , 'r--' , label = "Equilibrium Position")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)




plt.title("Time vs X-position of the Piston")
plt.xlabel("Time")
plt.ylabel("X-position of the Piston")
plt.show()


"""


part b
"""

plt.clf()




X_bar_values =[]


def entirety(F):
    N=1000
    m = 1         
    M = 100
    k_B = 1
    T_0 = 1       
    t=0
    
    
    
    x_v_state = np.zeros([N,2])    #state[ith particle][x,or v] of 
    
    
    
    
    
    def eq_x(F,N,T):
        """
        From PV =FX = n*k_b*T
        """
        return (N * k_B * T)/F
    
    X_0 = 2*(eq_x(F, N, T_0))
    
    
    
    
        
    """
    each particle has a random x, between (-x0,x0) and random v with standard 
    deviation v =sqrt(kb*t/m), from KE = 1/, average is 0 since we assume equal 
    amounts going both ways.
    """
    
    
    
    
    #print(random.gauss(0, np.sqrt((k_B*T_0)/m)))
    
    x_v_state[:,0] = np.random.uniform(-X_0, X_0, N)   #rand_x
    #print(x_v_state[:,0])
    x_v_state[:,1] = np.random.normal(0, np.sqrt((k_B*T_0)/m),N) # rand v
    #print(x_v_state[:,:])
    
    
    
    
    
    #plt.plot(x_v_state[:,1],x_v_state[:,0])
    
    def tau(x,v,X,V):
        if v >= 0:
            a = np.sqrt(((V-v)**2)-2*(F/M)*(x-X))
            return (M/F)*(V-v+a)
        elif v < 0:
            a = np.sqrt(((V+v)**2)+2*(F/M)*(x+X))
            return (M/F)*(V+v+a)
    
    
    
    
    def x_new(x,v,t):
        return (x+v*t)
    
    
    def V_new(V,t):
        return (V-((F/M)*t))
    
    
    def vel_post_coll(v,V):
        if v >= 0:
            W = (2*m*v+(M-m)*V)/(m+M)
            w = (2*M*V+(m-M)*v)/(m+M)
        elif v < 0:
            W = (-2*m*v+(M-m)*V)/(m+M)
            w = (-2*M*V+(m-M)*v)/(m+M)
        return [w,W]
    
    
    
    
    t=0
    
    
    
    
    x = x_v_state[:,0]
    x = np.random.uniform(-X_0, X_0, N) 
    v = x_v_state[:,1]
    v =  np.random.normal(0, np.sqrt((k_B*T_0)/m),N)
    X = X_0
    V = 0 # rand v
    
    
    
    X_arr = []
    v_arr =[]
    t_arr = []
    t = 0
    
    
    #final loop
    
    
    
    
    v_av = np.average(abs(v))    
        
    #print(v)
    tempature_ini =    m/k_B * v_av**2
    
    
    z = [] # array to hold last few elements of X
    wei = [] # array to hold weighting of last few elements of X
    
    
    for i in range(10000):
        """
        find min tau
        """
        tau_array =[]
        for i in range(len(x)):    
            tau_array.append(tau(x[i], v[i], X, V))
            
        min_tau = min(tau_array)
        
        min_tau_index = tau_array.index(min_tau) # index of particle with shortest time till collision
        
        #increase time by min_tau
        
        V = V_new(V,min_tau)
        for i in range(len(x)):
            
            x[i] = x_new(x[i],v[i],min_tau)
            
        
    
        
        t = t+ min_tau     
        #print(max(x), "and ", abs(x[min_tau_index]) , min_tau ) 
        
        X = abs( abs(x[min_tau_index]) )
        
        if (i> (99/100)*N):
            z.append(X)
            wei.append(min_tau)
        
        #print("Piston at" , X)
        W = vel_post_coll(v[min_tau_index],V)[1] 
        w = vel_post_coll(v[min_tau_index],V)[0]
        
        V = W
        X_arr.append(X)
        t_arr.append(t)
        
        v[min_tau_index] = w
     
        
        
        #print("X is" , X, "after ", t)
        
        #print(x_v_state[10,1])
        
        
        
    last_X = X_arr[(len(X_arr)-1)]
    
    #v_av = np.sqrt(abs(v))    
        
    #print(v)
    #tempature_apre =    m/k_B * v_av**2
    #print(tempature_apre/tempature_ini)
        
    """
        From PV/T = FX/T = FX_0/T_0
    """
        
    
    X_bar = np.average(z, weights=wei)
    X_bar_values.append(X_bar)
    
    print(X_bar)
    
    #eq_x_line = np.zeros(len(t_arr))+eq_x(F, N, tempature )
    
    
    
    #plt.plot(t_arr,X_arr)
    #plt.plot(t_arr, eq_x_line , 'r--' )
    
    
forces = [0.1,0.3,1,3,10,30,100]

ideal_x_predictions = []   



for i in range(len(forces)):
    entirety(forces[i])
    ideal_x_predictions.append((10)*(166.66)/forces[i])




plt.plot(forces,X_bar_values , label="F-X calculation")
#plt.plot(forces, ideal_x_predictions, 'r--')
plt.yscale('log')


plt.title("Force Applied to Piston vs X-bar")
plt.xlabel("Force")
plt.ylabel("X-bar ")



"""
We have FX=constant.   
we have an array for Forces.
The predicted value for F = 10 is X-bar = 166.66
"""

possib_forc = np.arange(0.2,100,1)

pred_X = np.zeros(len(possib_forc))

for i in range(len(pred_X)):
    pred_X[i] = (1666.6)/(possib_forc[i])


plt.plot(possib_forc,pred_X , 'r--' , alpha = 0.6 , label="prediction")


plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)




plt.show()







"""
Part C:
"""





N=4000
m = 1         
M = 100
k_B = 1
T_0 = 1       
F = 10       
t=0



x_v_state = np.zeros([N,2])    #state[ith particle][x,or v] of 





def eq_x(F,N,T):
    """
    From PV =FX = n*k_b*T
    """
    return (N * k_B * T)/F

X_0 = 2*(eq_x(F, N, T_0))




    
"""
each particle has a random x, between (-x0,x0) and random v with standard 
deviation v =sqrt(kb*t/m), from KE = 1/, average is 0 since we assume equal 
amounts going both ways.
"""




#print(random.gauss(0, np.sqrt((k_B*T_0)/m)))

x_v_state[:,0] = np.random.uniform(-X_0, X_0, N)   #rand_x
#print(x_v_state[:,0])
for i in range(N):
    x_v_state[i,1] = np.random.choice([-np.sqrt(k_B/m*T_0) ,np.sqrt(k_B/m*T_0)]) # rand v
#print(x_v_state[:,:])




#plt.plot(x_v_state[:,1],x_v_state[:,0])

def tau(x,v,X,V):
    if v >= 0:
        a = np.sqrt(((V-v)**2)-2*(F/M)*(x-X))
        return (M/F)*(V-v+a)
    elif v < 0:
        a = np.sqrt(((V+v)**2)+2*(F/M)*(x+X))
        return (M/F)*(V+v+a)




def x_new(x,v,t):
    return (x+v*t)


def V_new(V,t):
    return (V-((F/M)*t))


def vel_post_coll(v,V):
    if v >= 0:
        W = (2*m*v+(M-m)*V)/(m+M)
        w = (2*M*V+(m-M)*v)/(m+M)
    elif v < 0:
        W = (-2*m*v+(M-m)*V)/(m+M)
        w = (-2*M*V+(m-M)*v)/(m+M)
    return [w,W]




t=0




x = x_v_state[:,0]
x = np.random.uniform(-X_0, X_0, N) 
v = x_v_state[:,1]

X = X_0
V = 0 # rand v



X_arr = []
v_arr =[]
t_arr = []
t = 0


#final loop


v_av = np.average(abs(v))    
    
#print(v)
tempature_ini =    m/k_B * v_av**2


z = [] # array to hold last few elements of X
wei = [] # array to hold weighting of last few elements of X

P = 4*N
print(v)


for j in range(P):
    """
    find min tau
    """
    tau_array =[]
    for i in range(len(x)):    
        tau_array.append(tau(x[i], v[i], X, V))
        
    min_tau = min(tau_array)
    
    min_tau_index = tau_array.index(min_tau) # index of particle with shortest time till collision
    
    #increase time by min_tau
    
    V = V_new(V,min_tau)
    for i in range(len(x)):
        
        x[i] = x_new(x[i],v[i],min_tau)
        
    lst=[ 2 , np.floor(P*(2/10)), np.floor(P*(3/10)), np.floor(P*(4/10)), np.floor(P*(5/10)) ,  np.floor(P*(6/10)), np.floor(P*(7/10)),np.floor(P*(8/10)),np.floor(P*(9/10)), P ]
 
    if j in lst:
        print(j)
        plt.hist(v,bins=100)
        plt.ylabel("amount of particles with v")
        plt.xlabel("Velocity (v) of particles")
        plt.show()

    
    t = t+ min_tau  
    
    
    #print(max(x), "and ", abs(x[min_tau_index]) , min_tau ) 
    
    X = abs( abs(x[min_tau_index]) )
    
    if (i> (99/100)*N):
        z.append(X)
        wei.append(min_tau)
    
    #print("Piston at" , X)
    W = vel_post_coll(v[min_tau_index],V)[1] 
    w = vel_post_coll(v[min_tau_index],V)[0]
    
    V = W
    X_arr.append(X)
    t_arr.append(t)
    
    v[min_tau_index] = w
 
    
    
    #print("X is" , X, "after ", t)
    
    #print(x_v_state[10,1])
    
    
    

print(v_av)
#tempature_apre =    m/k_B * v_av**2
#print(tempature_apre/tempature_ini)
    
"""
    From PV/T = FX/T = FX_0/T_0
"""
    

X_bar = np.average(z, weights=wei)

print("X_bar is equal to ", X_bar)
"""
eq_x_line = np.zeros(len(t_arr))+166.7



plt.plot(t_arr,X_arr , label = "Displacement of Piston")
plt.plot(t_arr,eq_x_line , 'r--' , label = "Equilibrium Position")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)




plt.title("Time vs X-position of the Piston")
plt.xlabel("Time")
plt.ylabel("X-position of the Piston")


"""


"""
Part D:

"""


print("""
      Enthalpy one
      """)






import numpy as np
import matplotlib.pyplot as plt
import math
import random



N=1000
m = 1         
M = 100
k_B = 1
T_0 = 1       
F = 10       
t=0



x_v_state = np.zeros([N,2])    #state[ith particle][x,or v] of 





def eq_x(F,N,T):
    """
    From PV =FX = n*k_b*T
    """
    return (N * k_B * T)/F

X_0 = 2*(eq_x(F, N, T_0))




    
"""
each particle has a random x, between (-x0,x0) and random v with standard 
deviation v =sqrt(kb*t/m), from KE = 1/, average is 0 since we assume equal 
amounts going both ways.
"""




#print(random.gauss(0, np.sqrt((k_B*T_0)/m)))

x_v_state[:,0] = np.random.uniform(-X_0, X_0, N)   #rand_x
#print(x_v_state[:,0])
x_v_state[:,1] = np.random.normal(0, np.sqrt((k_B*T_0)/m),N) # rand v
#print(x_v_state[:,:])





#plt.plot(x_v_state[:,1],x_v_state[:,0])

def tau(x,v,X,V):
    if v >= 0:
        return (M/F)*(V-v+ np.sqrt(((V-v)**2)-2*(F/M)*(x-X)))
    else:
        return (M/F)*(V+v+ np.sqrt(((V+v)**2)+2*(F/M)*(x+X)))




def x_new(x,v,t):
    return (x+v*t)


def V_new(V,t):
    return (V-((F/M)*t))


def vel_post_coll(v,V):
    if v >= 0:
        W = (2*m*v+(M-m)*V)/(m+M)
        w = (2*M*V+(m-M)*v)/(m+M)
    elif v < 0:
        W = (-2*m*v+(M-m)*V)/(m+M)
        w = (-2*M*V+(m-M)*v)/(m+M)
    return [w,W]




t=0




x = x_v_state[:,0]
x = np.random.uniform(-X_0, X_0, N) 
v = x_v_state[:,1]
v =  np.random.normal(0, np.sqrt((k_B*T_0)/m),N)
X = X_0
V = 0 # rand v



X_arr = []
v_arr =[]
t_arr = []
t = 0


#final loop




v_av = np.average(abs(v))    
    
#print(v)
tempature_ini =    m/k_B * v_av**2


z = [] # array to hold last few elements of X
wei = [] # array to hold weighting of last few elements of X
H_arr = []

for i in range(2*N):
    """
    find min tau
    """
    tau_array =[]
    KE_arr =[]
    for i in range(len(x)):    
        tau_array.append(tau(x[i], v[i], X, V))
        
    min_tau = min(tau_array)
    
    
    min_tau_index = tau_array.index(min_tau) # index of particle with shortest time till collision
    
    #increase time by min_tau
    
    V = V_new(V,min_tau)
    for i in range(len(x)):
        
        x[i] = x_new(x[i],v[i],min_tau)
        KE_arr.append(1/2 * m * (v[i])**2)
            
    H = F*X + M/2*V +np.sum(KE_arr)
    H_arr.append(H)

    
    t = t+ min_tau     
    #print(max(x), "and ", abs(x[min_tau_index]) , min_tau ) 
    
    X = abs( abs(x[min_tau_index]) )
    
    if (i> (99/100)*N):
        z.append(X)
        wei.append(min_tau)
    
    #print("Piston at" , X)
    W = vel_post_coll(v[min_tau_index],V)[1] 
    w = vel_post_coll(v[min_tau_index],V)[0]
    
    V = W
    X_arr.append(X)
    t_arr.append(t)
    
    v[min_tau_index] = w

    
    
    #print("X is" , X, "after ", t)
    
    #print(x_v_state[10,1])
    
    
    

print(v_av)
#tempature_apre =    m/k_B * v_av**2
#print(tempature_apre/tempature_ini)
    
"""
    From PV/T = FX/T = FX_0/T_0
"""
    

X_bar = np.average(z, weights=wei)

print("X_bar is equal to ", X_bar)

#eq_x_line = np.zeros(len(t_arr))+166.7

plt.plot(t_arr,H_arr , label = "Enthalpy")
plt.xlabel("Time")
plt.ylabel("Enthalpy")

plt.ylim(0,3000)



plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)







"""
plt.plot(t_arr,X_arr , label = "Displacement of Piston")
plt.plot(t_arr,eq_x_line , 'r--' , label = "Equilibrium Position")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)




plt.title("Time vs X-position of the Piston")
plt.xlabel("Time")
plt.ylabel("X-position of the Piston")

"""
















