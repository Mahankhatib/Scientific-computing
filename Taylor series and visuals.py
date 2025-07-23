import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
x=sp.symbols("x")

def mclauren_series(fx,n,t=None,plot=None):
    res=fx.subs(x,0)
    for i in range(1,n+1):
        res=res+((sp.diff(fx,x,i).subs(x,0))*(x**i)/(sp.factorial(i)))
    if plot is not None:
        domain=np.linspace(-10, 10,200)
        res_num_app=sp.lambdify(x, res,"numpy")
        res_num_actual=sp.lambdify(x,fx,"numpy")
        y_app=res_num_app(domain)
        y_actual=res_num_actual(domain)
        plt.figure()
        plt.plot(domain,y_actual,label="Original function")
        plt.plot(domain,y_app,label="Approximated function")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()        
    if t is not None:
        return f"In x={t} : App.:{res.subs(x,t).evalf():.6f} , actual:{fx.subs(x,t).evalf():.6f}"
    else:
        return res 
print(mclauren_series(sp.sin(x)*sp.exp(-x),3,None,True))        
    
                 