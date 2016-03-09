# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:45:49 2013

@author: Myhay
"""

import csv

f = file('C:\Users\Myhay\Desktop\PracticaSI\Practica1_Network\iris.data', 'r')
r= csv.reader(f)
list_data = list(r)
f.close()

print(list_data)
