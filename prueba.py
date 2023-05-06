import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Leer datos
data = pd.read_csv('./data/train.csv')

# Extraer dichas columnas
newData = data[['GrLivArea','SalePrice']]
print(newData.head())

# Contruir x - y
x = newData['GrLivArea']
y = newData['SalePrice']

# Standarizar los datos
x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x] 

print("Shape of X: ", x.shape)
print("Shape of y:", y.shape)


def gradient_descent(x, y, w, iterations, eta):
    wt=w
    x_t = np.transpose(x)
    for i in range(0, iterations):
        Aux = np.dot(x, wt) - y
        RSS = np.dot(x_t, Aux)
        wt1 = wt - eta*RSS
        wt=wt1
    return wt1, RSS


import numpy as np

# Learning rate
eta = 0.001 

# Número de iteraciones
iterations = 1000 #No. of iterations

# Seed para inicializar w
np.random.seed(123)
w0 = np.random.rand(2)

weights, loss = gradient_descent(x, y, w0, iterations, eta)
print(weights)
# Definir figure
fig = plt.figure()
ax = plt.axes()
plt.title('Sale Price vs Living Area')
plt.xlabel('Living Area in square feet (normalised)')
plt.ylabel('Sale Price ($)')
plt.scatter(x[:,1], y, color='red')
line, = ax.plot([], [], lw=2)
annotation = ax.text(-1, 700000, '')
annotation.set_animated(True)
plt.close()

# Generar animacion de los datos
def init():
    line.set_data([], [])
    annotation.set_text('')
    return line, annotation

# Función para la animación
def animate(i):
    x = np.linspace(-5, 20, 1000)
    y = weights[i][1]*x + weights[i][0]
    line.set_data(x, y)
    annotation.set_text('loss = %.2f e10' % (loss[i]/10000000000))
    return line, annotation

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=300, interval=0, blit=True)

anim.save('animation.gif', writer='imagemagick', fps = 30)

# Visualizar la animación
import io
import base64
from IPython.display import HTML

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))