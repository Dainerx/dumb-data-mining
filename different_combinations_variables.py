from sklearn import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import numpy as np

#C
def C():
    iris = datasets.load_iris()
    classes = iris.target
    colormap = np.array(['#0b559f', '#ffff00','#000000'])
    
    plt.title('comb 1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(iris.data[:,0], iris.data[:,1], c=colormap[classes]) #c is color
    pop_a = mpatches.Patch(color='#0b559f', label=iris.target_names[0])
    pop_b = mpatches.Patch(color='#ffff00', label=iris.target_names[1])
    pop_c = mpatches.Patch(color='#000000', label=iris.target_names[2])
    plt.legend(handles=[pop_a,pop_b,pop_c])
    plt.show()
    
    plt.title('comb 2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(iris.data[:,0], iris.data[:,2],  c=colormap[classes]) #c is color
    pop_a = mpatches.Patch(color='#0b559f', label=iris.target_names[0])
    pop_b = mpatches.Patch(color='#ffff00', label=iris.target_names[1])
    pop_c = mpatches.Patch(color='#000000', label=iris.target_names[2])
    plt.legend(handles=[pop_a,pop_b,pop_c])
    plt.show()
    
    plt.title('comb 3')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(iris.data[:,0], iris.data[:,3], c=colormap[classes]) #c is color
    pop_a = mpatches.Patch(color='#0b559f', label=iris.target_names[0])
    pop_b = mpatches.Patch(color='#ffff00', label=iris.target_names[1])
    pop_c = mpatches.Patch(color='#000000', label=iris.target_names[2])
    plt.legend(handles=[pop_a,pop_b,pop_c])
    plt.show()
    
    plt.title('comb 4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(iris.data[:,1], iris.data[:,2],  c=colormap[classes]) #c is color
    pop_a = mpatches.Patch(color='#0b559f', label=iris.target_names[0])
    pop_b = mpatches.Patch(color='#ffff00', label=iris.target_names[1])
    pop_c = mpatches.Patch(color='#000000', label=iris.target_names[2])
    plt.legend(handles=[pop_a,pop_b,pop_c])
    plt.show()
    
    plt.title('comb 5')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(iris.data[:,1], iris.data[:,3], c=colormap[classes]) #c is color
    pop_a = mpatches.Patch(color='#0b559f', label=iris.target_names[0])
    pop_b = mpatches.Patch(color='#ffff00', label=iris.target_names[1])
    pop_c = mpatches.Patch(color='#000000', label=iris.target_names[2])
    plt.legend(handles=[pop_a,pop_b,pop_c])
    plt.show()
    
    plt.title('comb 6')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(iris.data[:,2], iris.data[:,3], c=colormap[classes]) #c is color
    pop_a = mpatches.Patch(color='#0b559f', label=iris.target_names[0])
    pop_b = mpatches.Patch(color='#ffff00', label=iris.target_names[1])
    pop_c = mpatches.Patch(color='#000000', label=iris.target_names[2])
    plt.legend(handles=[pop_a,pop_b,pop_c])
    plt.show()

