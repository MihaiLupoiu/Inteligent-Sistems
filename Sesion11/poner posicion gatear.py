# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:22:47 2013

@author: Usuario
"""

from Nao import Nao
from math import *
from pyevolve import *

nao = Nao()
#nao.stand_up() #se pone de pie
nao.initCrawling() #se pone en posicion de gatear

variable = nao.crawl([1.28,0.1,0.21,0.,0.035,0.039,-2.,0.12,-0.86,pi,0.06,0.33,pi/2,0.005,-0.11,-2.,0.008,1.8,0.])
