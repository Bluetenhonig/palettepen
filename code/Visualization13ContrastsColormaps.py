# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 09:07:43 2020

@author: lsamsi
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap # discrete 
from matplotlib.colors import LinearSegmentedColormap # gradient (by interpolation)


tab20b = cm.get_cmap('tab20b', 8)
# print(tab20b)

# print('tab20b(range(8))', tab20b(range(8)))
# print('tab20b(np.linspace(0, 1, 8))', tab20b(np.linspace(0, 1, 8)))


# create Custom Qualitative Colormaps
contrastofhue = [[1,0,0, 1], # R, G, B, A (0-1)
[0,1,0, 1], 
[0,0,1, 1]
]

contrastofhue2 = [[1,0,0, 1], # R, G, B, A (0-1)
[1, 1, 0, 1],
[0,0,1, 1]
]

contrastofhue3 = [[1,0,0, 1], # R, G, B, A (0-1)
[1, 1, 0, 1], 
[0,1,0, 1], 
[0,0,1, 1], 
[.5, 0, .5, 1]
]

lightdarkcontrast = [ # R, G, B, A (0-1)
[1,1,1, 1],
[0,0,0, 1]
]
lightdarkcontrast2 = [
[.8, .8, .8, 1],
[1, 1, 1, 1],
[0, 0, 0, 1], # R, G, B, A (0-1)
[.2, .2, .2, 1]
]
lightdarkcontrast3 = [ # R, G, B, A (0-1)
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

# contrast of hue
# light-dark contrast
# cold-warm contrast
# complementary contrast
# contrast of saturation
# contrast of extension 

#BLACK: [0,0,0, 1]
#WHITE: [1,1,1, 1]
#RED: [1,0,0, 1]
#ORANGE: [1, 0.647, 0, 1]
#YELLOW: [1, 1, 0, 1]
#GREEN: [0,1,0, 1]
#BLUE: [0,0,1, 1]
#VIOLET: [.5, 0, .5, 1]

import numpy as np
import matplotlib.pyplot as plt


# Have colormaps separated into categories:
# http://matplotlib.org/examples/color/colormaps_reference.html
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

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list, nrows)

plt.show()




