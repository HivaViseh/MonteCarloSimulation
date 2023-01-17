from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt



xData = np.array([0.026,0.053,0.081,0.108,0.149,0.191,0.218,0.273,0.301,0.328,0.369,0.493,0.521,0.521,0.548,0.603,0.741,0.796,0.892,0.961,0.988,1.016,1.016,1.043,1.098,1.291,1.318,1.717,1.813,1.978,3.003
])
yData = np.array([0.469,0.733,0.117,0.059,0.495,1,0.696,0.784,0.337,0.436,0.037,0.502,0.692,0.143,0.359,0.538,0.176,0.3,0.905,0.516,0.996,0.359,0.864,0.615,0.22,0.615,0.41,0.733,0.85,1,0.996
])

m = GEKKO(remote=False)

# nonlinear regression
a,b,c = m.Array(m.FV,3,value=0,lb=-10,ub=10)
x = m.MV(xData); y = m.CV(yData)
a.STATUS=1; b.STATUS=1; c.STATUS=1; y.FSTATUS=1
m.Equation(y==1.0/(1.0+m.exp(-a*(x-b)))+c)

# cubic spline
z = m.Var()
m.cspline(x,z,xData,yData,True)

m.options.IMODE = 2; m.options.EV_TYPE = 2
m.solve()

# stats (from other answer)
absError = y.value - yData
SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(yData))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)
print('Parameters', a.value[0], b.value[0], c.value[0])

# deep learning
from gekko import brain
b = brain.Brain()
b.input_layer(1)
b.layer(linear=1)
b.layer(tanh=2)
b.layer(linear=1)
b.output_layer(1)
b.learn(xData,yData,obj=1,disp=False) # train
xp = np.linspace(min(xData),max(xData),100)
w = b.think(xp) # predict

plt.plot(xData,yData,'k.',label='data')
plt.plot(x.value,y.value,'r:',lw=3,label=r'$1/(1+exp(-a(x-b)+c)$')
plt.plot(x.value,z.value,'g--',label='c-spline')
plt.plot(xp,w[0],'b-.',label='deep learning')
plt.legend(); plt.show()