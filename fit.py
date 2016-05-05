import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,a,b,c):
    return a*np.exp(-(x-b)**2 / (2*c**2))

x_space = np.linspace(-0.5, 2.5, 100)

y_data = func(x_space, 3, 1, 2)

popt, pcov = curve_fit(func, x_space, y_data)
perr = np.sqrt(np.diag(pcov))

plt.plot(x_space, func(x_space, popt[0], popt[1], popt[2]))
print(popt)
plt.show()

