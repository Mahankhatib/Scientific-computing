#nozzle simulation
#BIG PICTURE: Combustion Chamber => Converging => throat => Diverging
#sequence initilization
#steady_adiabaticc_isentropic-1D flow , gamma=specific heat ratio

import mpmath
from mpmath import findroot #dangerous in this case: needs good initial guess
import numpy as np
from math import sqrt
from math import log
import matplotlib.pyplot as plt
from scipy.optimize import brentq
#-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$

def nozzle_profile(x):
    return np.abs(x-10)+1

def A(x):
    return 2*nozzle_profile(x)

#Nozzle length
nozzle_positions=np.linspace(5,15,400)
x_inlet=nozzle_positions[0]
x_outlet=nozzle_positions[-1]
#Chamber/ambient Charactaristics:
T_chamber=3000
P_chamber=1e6
P_amb=1e5
#-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$

#FINDIN throat  ==>>  failed becasue of discrit.
#area_val=A(nozzle_positions)
#throat_ind=np.argmin(area_val)

#x_throat=nozzle_positions[throat_ind]
#A_star=area_val[throat_ind]
x_throat=10 #case specific
A_star=A(10)
print(f"throat at x={x_throat:.4f}")
print(f"area at throat A={A_star:.4f}")


gamma=1.4 #specific heat ratio of air(ideal gas)
cp=1005 # speicfic heat at constant pressure
R=287
#find mach number M => showing how fast/slow movement wrt. sound!

#eq= F(M)-R(x)
def area_ratio(x):
    return A(x)/A_star


def area_ratio_M(M):
    exponent=(gamma+1)/(2*(gamma-1))
    base=(2/(gamma+1))*(1+((M**2)*(gamma-1))/2)
    return (1/M)*(base**exponent)

##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$

#Thermodynamic Property Distributions
 
def mach_at_x(x):
    R=area_ratio(x)
    if abs(x-x_throat)<1e-6: #special case: sonic: M=1
        return 1
    def g(M):
        return area_ratio_M(M)-R

    #initial guess
    if x>x_throat: #supersonic
        return brentq(g,1.001,10)
    if x<x_throat: #subsonic
        return brentq(g,1e-6,0.999)
M_data=[mach_at_x(x) for x in nozzle_positions]


def T(M):
    scaled_T=1/(1+((gamma-1)/2)*(M**2))
    return scaled_T
T_data=[T(M) for M in M_data]
T_exit=T_data[0]/(1+(((gamma-1)/2)*((M_data[-1])**(2))))

def P(M):
    scaled_P=T(M)**(gamma/(gamma-1))
    return scaled_P
P_data=[P(M) for M in M_data]
P_exit=P_data[-1]*P_chamber

def Rho(M):
    scaled_Rho=T(M)**(1/(gamma-1))
    return scaled_Rho
Rho_data=[Rho(M) for M in M_data]

def S(T,P):
    return cp*log(T)-R*log(P)
S_data=[S(T*T_chamber,P*P_chamber) for T,P in zip(T_data,P_data)]
    
##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$
#Shock detection
dx=nozzle_positions[2]-nozzle_positions[1]
n=len(nozzle_positions)
grad_S=np.zeros(n)
f=list(S_data)
grad_S[0]=(f[1]-f[0])/(dx)        #forward diff.
grad_S[n-1]=(f[n-1]-f[n-2])/(dx)  #backward diff.
for i in range(1,n-1):            #central diff.
    grad_S[i]=(f[i+1]-f[i-1])/(2*dx)

#plt.plot(nozzle_positions,grad_S)
#plt.grid(True)

##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$

#Dynamic Properties

def mass_flow(R=287):
    base=2/(gamma+1)
    exponent=(gamma+1)/(2*(gamma-1))
    K=sqrt((gamma)/(R*T_data[0]*T_chamber))
    return A(5)*P_data[0]*P_chamber*K*((base)**(exponent))

def velocity(x):
    M=mach_at_x(x)
    return M*(sqrt(gamma*R*T(M)*T_chamber))
v_data=[velocity(x) for x in nozzle_positions]
#for x,v in zip(nozzle_positions,v_data):
    #print(f"x={x:.2f} ==> v={v:.2f}")
#print(f"Ve={M_data[-1]*sqrt(gamma*287*T_data[-1]*T_chamber)}") #CONFIRMATION

def Thrust():
    mdot=mass_flow()
    v_exit=v_data[-1]
    P_chamber=P_amb
    A_exit=A(15)
    return (mdot*v_exit)+((P_exit-P_chamber)*(A_exit))
#print(f"thrust={Thrust()/1000} kN")

##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$

# acceleration and momentum conservation verification ...$$$$
grad_v=np.zeros(n)
g=list(v_data)
#combo of first order forward/backward + second order
grad_v[0]=(g[1]-g[0])/(dx)
grad_v[n-1]=(g[n-1]-g[n-2])/(dx)
for i in range(1,n-1):
    grad_v[i]=(g[i+1]-g[i-1])/(2*dx)
#convective acceleration
conv_a=np.zeros(n)
for k in range(n):
    conv_a[k]=v_data[k]*grad_v[k]
#plt.plot(nozzle_positions,conv_a)
LHS=np.zeros(n)
rho_0=P_chamber/(R*T_chamber)
for j in range(n):
    LHS[j]=Rho_data[j]*rho_0*conv_a[j]
grad_P=np.zeros(n)
h=list(P_data)
grad_P[0]=P_chamber*(P_data[1]-P_data[0])/(dx)
grad_P[n-1]=P_chamber*(P_data[n-1]-P_data[n-2])/(dx)
for i in range(1,n-1):
    grad_P[i]=P_chamber*(P_data[i+1]-P_data[i-1])/(2*dx)
#plt.plot(nozzle_positions,LHS,label="LHS",color="k")
#plt.plot(nozzle_positions,-1*grad_P,label="RHS (- gradeint of pressure)",color="r")
residual=LHS+grad_P
plt.figure()
plt.plot(nozzle_positions,residual,label="residual")

##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$

#PLOTTING NOZZLE PROFILE
    
#boundaries of the nozzle
xi=nozzle_positions[0]
xf=nozzle_positions[-1]
x_lower_bound=xi*np.ones(200)
x_higher_bound=xf*np.ones(200)
y_bound=np.linspace(-1*nozzle_profile(xi),nozzle_profile(xi),200)

plt.figure()
plt.plot(nozzle_positions,nozzle_profile(nozzle_positions),color="black")
plt.plot(nozzle_positions,(-1)*nozzle_profile(nozzle_positions),color="black")
plt.plot(x_lower_bound,y_bound,color="red")
plt.plot(x_higher_bound,y_bound,color="red")
plt.grid(True)
plt.title("Nozzle Profile")
plt.xlabel("Nozzle Position : x[m]")
plt.ylabel("Nozzle Profile  : y[m]")
#plt.show()
##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-

#PLOTTING : Mach Number(x)

plt.figure()
plt.subplot(2,2,1)
plt.plot(nozzle_positions,np.array(M_data),color="black")
plt.axvline(x_throat,linestyle="--")
plt.axhline(1,linestyle="dotted")
plt.grid(True)
plt.xlabel("Nozzle Position : x[m]")
plt.ylabel("Mach Number")
plt.title("Mach Number Evolution")
#plt.show()
##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-

#PLOTTING : Temperature(x)

plt.subplot(2,2,2)
plt.plot(nozzle_positions,np.array(T_data))
plt.axvline(x_throat,linestyle="--")
plt.axhline(T(1),linestyle="--")
plt.grid(True)
plt.xlabel("Nozzle Position : x[m]")
plt.ylabel("Temperature ratio T/T0")
plt.title("Temperature Distribution")
#plt.show()
##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-

#PLOTTING : Pressure(x)

plt.subplot(2,2,3)
plt.plot(nozzle_positions,np.array(P_data),color="red")
plt.axvline(x_throat,linestyle="--")
plt.axhline(P(1),linestyle="--")
plt.grid(True)
plt.xlabel("Nozzle Position : x[m]")
plt.ylabel("Pressure ratio P/P0")
plt.title("Pressure Distribution")
##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-

#PLOTTING : Density(x)

plt.subplot(2,2,4)
plt.plot(nozzle_positions,np.array(Rho_data),color="green")
plt.axvline(x_throat,linestyle="--")
plt.axhline(Rho(1),linestyle="--")
plt.grid(True)
plt.xlabel("Nozzle Position : x[m]")
plt.ylabel("Density ratio rho/rho0")
plt.title("Density Distribution")
##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$

#2D- Contour plots

# Mach number
def Mach_contour():
    #create empty grid
    X=nozzle_positions
    Y=np.linspace(-nozzle_profile(xi),nozzle_profile(xi),40)
    x_grid,y_grid=np.meshgrid(X,Y)
    #create mach number grid
    M_grid=np.zeros_like(x_grid)
    for i,x in enumerate(X):
        M=mach_at_x(x)
        for j,y in enumerate(Y):
            if abs(y)<nozzle_profile(x)*0.9:# if farther than 90% of R
                M_grid[j,i]=M # core
            else:
                M_grid[j,i]=M*0.8 #approx
    plt.figure()
    plt.contourf(x_grid,y_grid,M_grid,levels=20,cmap="viridis")
    plt.colorbar(label="Mach Number")
    plt.plot(X,nozzle_profile(X),"r--",linewidth=2)
    plt.plot(X,-1*nozzle_profile(X),"r--",linewidth=2)
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.title("Mach Number Distribution")
    plt.show()
    plt.axis("equal")


# Velocity
def Velocity_contour():
    X=nozzle_positions
    Y=np.linspace(-nozzle_profile(xi),nozzle_profile(xi),40)
    x_grid,y_grid=np.meshgrid(X,Y)
    V_grid=np.zeros_like(x_grid)
    for i,x in enumerate(X):
        v=velocity(x)
        for j,y in enumerate(Y):
            V_grid[j,i]=v
    plt.figure()
    plt.contourf(x_grid,y_grid,V_grid,levels=20,cmap="plasma")
    plt.colorbar(label="Velocity [m/s]")
    plt.plot(X,nozzle_profile(X),"r--",linewidth=2)
    plt.plot(X,-1*nozzle_profile(X),"r--",linewidth=2)
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.title("Velocity Distribution")
    plt.show()
    plt.axis("equal")
##-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$


    



    


