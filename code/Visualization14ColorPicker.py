# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:49:53 2020

@author: Anonym
"""

# color picker: choose rgb color 
import tkinter as tk
import tkinter.ttk as ttk
from tkcolorpicker import askcolor

root = tk.Tk()
style = ttk.Style(root)
style.theme_use('default') # choose from  [ "clam", "alt", "default", "classic"]

askcolor = askcolor((255, 255, 0), root)
#root.mainloop()
#root.quit()
root.destroy()

rgb = askcolor[0]
print('Your chosen rgb color: ', rgb)
