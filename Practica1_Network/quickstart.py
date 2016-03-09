# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Documents and Settings\Administrador\.spyder2\.temp.py
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

net = buildNetwork(2, 3, 1,  hiddenclass=TanhLayer, bias=True)
print("Output of the network before training")
print(net.activate ([0,0]))
print(net.activate ([1,0]))
print(net.activate ([0,1]))
print(net.activate ([1,1]))

ds = SupervisedDataSet(2, 1)
ds.addSample((0,0),(0,))
ds.addSample((0,1),(1,))
ds.addSample((1,0),(1,))
ds.addSample((1,1),(0,))

trainer = BackpropTrainer(net, ds, momentum=0.9)

for epoch in range(0,300):
    trainer.train()

print("Output of the network after training")
print(net.activate ([0,0]))
print(net.activate ([1,0]))
print(net.activate ([0,1]))
print(net.activate ([1,1]))