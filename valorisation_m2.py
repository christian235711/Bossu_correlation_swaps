import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


# RAMIRES ALIAGA Christian

################################################

T = np.linspace(0,2,10)
w = np.linspace(0,0.3,10)
T,w=np.meshgrid(T,w)

r=0
v0=20

def ajustement_convex():
    c= np.sqrt(v0*np.exp(r*T))*( 1-np.exp( (-1/6)*(w**2)*T  ) )
    #c= (1/6)*np.sqrt(v0*np.exp(r*T))*(w**2)*T  # OU APPROXIMATION

    return c


fig=plt.figure(figsize=(10, 6)); plt.title("Ajustement de convexité entre la variance et la volatilité des swaps")
ax=Axes3D(fig)
plt.gca(projection='3d').plot_surface(T,w,ajustement_convex().T,rstride=1,cmap=mpl.cm.coolwarm, cstride=1,linewidth=0,antialiased=False)

ax.set_xlabel(xlabel="Maturité T")
ax.set_ylabel(ylabel="Volatilité de la volatilité w")
ax.set_zlabel(zlabel="Ajustement de convexité c ")



#################################################

# les valeurs des paramètres sont les mêmes que ceux de l'article
T= 3
K=20
sigma=0.2
r= 0

St=np.arange(17,23,0.1)
vt=St   
w=0.2

def phi(x): # fonction de répartition de la loi normale centrée réduite
    return stats.norm.cdf(x, loc = 0, scale = 1)


def prix_call_black_scholes(t):
    d1= ( np.log(St/K) + ( r + (sigma**2)/2 )*(T-t) ) / (np.sqrt(T-t)*sigma)

    d2= d1 - sigma*np.sqrt(T-t)

    Ct = St*phi(d1) - K*np.exp(-r*(T-t) )*phi(d2)
    return Ct


def payoff(St,K):   
    return np.maximum(0,St-K)


def prix_call_toy_model(t):
    d1 = ( np.log(np.sqrt( vt*np.exp(r*(T-t))/K ) ) + (1/3)*(w**2)*T*((T-t)/T)**3 )/( (1/np.sqrt(3))*w*np.sqrt(T)*((T-t)/T)**(3/2)  )

    d2 = d1 - (2/np.sqrt(3))*w*np.sqrt(T)*((T-t)/T)**(3/2)
    
    ft = vt*phi(d1)-K*np.exp(-r*(T-t))*phi(d2)
    return ft


plt.figure(figsize=(10, 6)); plt.title("Le prix du call sur la variance réalisée à t=0")
plt.plot(St,payoff(St,K)/K*100, label = "Payoff")
plt.plot(St,prix_call_black_scholes(0)/K*100, label="Black-Scholes" )
plt.plot(vt,prix_call_toy_model(0)/K*100, label="Toy Model")
plt.ylabel("Prix du call \n (%Strike)")
plt.legend()


plt.figure(figsize=(10, 6)); plt.title("Le prix du call sur la variance réalisée à t=1")
plt.plot(St,payoff(St,K)/K*100, label="Payoff")
plt.plot(St,prix_call_black_scholes(1)/K*100, label= "Black-Scholes")
plt.plot(vt,prix_call_toy_model(1)/K*100,label="Toy Model" )
plt.ylabel("Prix du call \n (%Strike)")
plt.legend()


plt.figure(figsize=(10, 6)); plt.title("Le prix du call sur la variance réalisée à t=2")
plt.plot(St,payoff(St,K)/K*100, label="Payoff")
plt.plot(St,prix_call_black_scholes(2)/K*100, label="Black-Scholes")
plt.plot(vt,prix_call_toy_model(2)/K*100, label="Toy Model")
plt.ylabel("Prix du call \n (%Strike)")
plt.legend()


plt.show()






