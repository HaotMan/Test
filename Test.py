#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 17:02:21 2018

@author: tang2
"""


import numpy as np


def cha():
    global c, a
    c += 1
    a += 1
    

if __name__ == "__main__":
    a = 3
    c = 9
    cha()
    print(c, a)