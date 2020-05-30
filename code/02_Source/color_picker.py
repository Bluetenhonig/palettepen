# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:28:05 2020

@author: Anonym
"""

import tkinter as tk
import tkinter.ttk as ttk
from tkcolorpicker import askcolor

root = tk.Tk()
style = ttk.Style(root)
style.theme_use('clam')

askcolor = askcolor((255, 255, 0), root)
root.mainloop()

print(askcolor[0])