# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 09:07:43 2020

@author: lsamsi
"""

# import modules 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt


# declare variables
tab20b = cm.get_cmap('tab20b', 8)

# make contrasts as RGBA
contrastofhue = [[1,0,0, 1],
[0,1,0, 1], 
[0,0,1, 1]
]

contrastofhue2 = [[1,0,0, 1],
[1, 1, 0, 1],
[0,0,1, 1]
]

contrastofhue3 = [[1,0,0, 1], 
[1, 1, 0, 1], 
[0,1,0, 1], 
[0,0,1, 1], 
[.5, 0, .5, 1]
]

lightdarkcontrast = [
[1,1,1, 1],
[0,0,0, 1]
]
lightdarkcontrast2 = [
[.8, .8, .8, 1],
[1, 1, 1, 1],
[0, 0, 0, 1],
[.2, .2, .2, 1]
]
lightdarkcontrast3 = [
[1, 1, 0, 1],
[.5, 0, .5, 1]
]
coldwarmcontrast = [[0,0,1, 1],
[1, 0.647, 0, 1] 
]
coldwarmcontrast2 = [[.5, 0, .5, 1],
 [0,0,1, 1],
[1, 0.647, 0, 1],
[1, 1, 0, 1]
]
coldwarmcontrast3 = [[0,1,0, 1],
[.5, 0, .5, 1],
[0,0,1, 1],
[1, 0.647, 0, 1],
[1, 1, 0, 1],
[1,0,0, 1]
]
complementarycontrastob = [[0,0,1, 1],
[1, 0.647, 0, 1]
]
complementarycontrastgr = [[0,1,0, 1],
[1,0,0, 1]
]
complementarycontrastvy = [[.5, 0, .5, 1], 
[1, 1, 0, 1]
]
contrastofsat= [[.66, 0.2, .2, 1], 
 [1,0,0, 1]
]
contrastofsat2= [[0,1,0, 1],
 [1,0,0, 1]
]
contrastofsat3= [[0.62, 0.65, 0.45, 1],
[0,1,0, 1],
]

contrastofext = [ 
[0.62, 0.65, 0.45, 1],
[0.62, 0.65, 0.45, 1],
[0.62, 0.65, 0.45, 1],
[0.62, 0.65, 0.45, 1],
[1,0,0, 1]
]
contrastofext2 = [ 
[.66, 0.2, .2, 1], 
[.66, 0.2, .2, 1], 
[.66, 0.2, .2, 1], 
[.66, 0.2, .2, 1], 
[1,0,0, 1]
]
contrastofext3 = [ 
[.66, 0.2, .2, 1], 
[.66, 0.2, .2, 1], 
[.66, 0.2, .2, 1], 
[.66, 0.2, .2, 1], 
[1, 1, 0, 1]
]

# register qualitative colormap 
contrastofhue = ListedColormap(contrastofhue)
plt.register_cmap(name='Contrast of Hue', cmap=contrastofhue)
contrastofhue2 = ListedColormap(contrastofhue2)
plt.register_cmap(name='Contrast of Hue 2', cmap=contrastofhue2)
contrastofhue3 = ListedColormap(contrastofhue3)
plt.register_cmap(name='Contrast of Hue 3', cmap=contrastofhue3)

lightdarkcontrast = ListedColormap(lightdarkcontrast)
plt.register_cmap(name='Light-dark Contrast', cmap=lightdarkcontrast)
lightdarkcontrast2 = ListedColormap(lightdarkcontrast2)
plt.register_cmap(name='Light-dark Contrast 2', cmap=lightdarkcontrast2)
lightdarkcontrast3 = ListedColormap(lightdarkcontrast3)
plt.register_cmap(name='Light-dark Contrast 3', cmap=lightdarkcontrast3)

coldwarmcontrast = ListedColormap(coldwarmcontrast)
plt.register_cmap(name='Cold-warm Contrast', cmap=coldwarmcontrast)
coldwarmcontrast2 = ListedColormap(coldwarmcontrast2)
plt.register_cmap(name='Cold-warm Contrast 2', cmap=coldwarmcontrast2)
coldwarmcontrast3 = ListedColormap(coldwarmcontrast3)
plt.register_cmap(name='Cold-warm Contrast 3', cmap=coldwarmcontrast3)

complementarycontrastob = ListedColormap(complementarycontrastob)
plt.register_cmap(name='Complementary Contrast: Blue-Orange', cmap=complementarycontrastob)
complementarycontrastgr = ListedColormap(complementarycontrastgr)
plt.register_cmap(name='Complementary Contrast: Green-Red', cmap=complementarycontrastgr)
complementarycontrastvy = ListedColormap(complementarycontrastvy)
plt.register_cmap(name='Complementary Contrast: Violet-Yellow', cmap=complementarycontrastvy)

contrastofsat = ListedColormap(contrastofsat)
plt.register_cmap(name='Contrast of Saturation', cmap=contrastofsat)
contrastofsat2 = ListedColormap(contrastofsat2)
plt.register_cmap(name='Contrast of Saturation 2', cmap=contrastofsat2)
contrastofsat3 = ListedColormap(contrastofsat3)
plt.register_cmap(name='Contrast of Saturation 3', cmap=contrastofsat3)

contrastofext = ListedColormap(contrastofext)
plt.register_cmap(name='Contrast of Extension', cmap=contrastofext)
contrastofext2 = ListedColormap(contrastofext2)
plt.register_cmap(name='Contrast of Extension 2', cmap=contrastofext2)
contrastofext3 = ListedColormap(contrastofext3)
plt.register_cmap(name='Contrast of Extension 3', cmap=contrastofext3)


cmaps = [
         ('Color Contrast', [
            'Contrast of Hue', 'Contrast of Hue 2', 'Contrast of Hue 3', 'Light-dark Contrast', 'Light-dark Contrast 2', 'Light-dark Contrast 3',
            'Cold-warm Contrast', 'Cold-warm Contrast 2', 'Cold-warm Contrast 3',
            'Complementary Contrast: Blue-Orange', 'Complementary Contrast: Green-Red', 'Complementary Contrast: Violet-Yellow', 
            'Contrast of Saturation', 'Contrast of Saturation 2', 'Contrast of Saturation 3',
            'Contrast of Extension', 'Contrast of Extension 2', 'Contrast of Extension 3'])]


nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

#%%

def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' Colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        print(name)
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
    for ax in axes:
        ax.set_axis_off()


#%%
if __name__ == '__main__': 
    
    for cmap_category, cmap_list in cmaps:
        plot_color_gradients(cmap_category, cmap_list, nrows)   
    plt.show()




