# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:28:08 2013

@author: Usuario
"""

from Nao import Nao
from math import *
from pyevolve import *

nao = Nao()
#nao.stand_up() se pone de pie
#nao.initCrawling() se pone en posicion de gatear
#nao.stand_up()

#variable = nao.crawl([1.28,0.1,0.21,0.,0.035,0.039,-2.,0.12,-0.86,pi,0.06,0.33,pi/2,0.005,-0.11,-2.,0.008,1.8,0.])

#print variable 

def eval_func(chromosome):

   robot_parameter = []
   VECTOR = chromosome.getInternalList()
   max_vector = [1.8, 0.5, 0.5, 3.14, 0.4, 0.1, 3.14, 0.4, 0, 3.14, 0.2, 0.5, 3.14, 0.1, -0.2, 3.14, 0.2, 1.9, 3.14]
   min_vector = [0.8, 0, -0.5, -3.14, 0.2, -0.1, -3.14, 0, -1.2, -3.14, 0, 0, -3.14, 0, -0.5, -3.14, 0, 1.6, -3.14]
   
   
   for i in range(19):
       robot_parameter.append( VECTOR[i] * (max_vector[i] - min_vector[i]) + min_vector[i])
       
   score = nao.crawl(robot_parameter)
   
   return score

# Genome instance, 1D List of 50 elements
genome = G1DList.G1DList(19)

# Sets the range max and min of the 1D List
genome.setParams(rangemin=0, rangemax=1)

# The evaluator function (evaluation function)
genome.evaluator.set(eval_func)

# Genetic Algorithm Instance
ga = GSimpleGA.GSimpleGA(genome)
# Set the Roulette Wheel selector method, the number of generations and
# the termination criteria
ga.selector.set(Selectors.GRouletteWheel)
ga.setGenerations(5)
ga.setMinimax(Consts.minimaxType["maximize"])
ga.setCrossoverRate(1.0)
ga.setMutationRate(0.02)
ga.setPopulationSize(10)
ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)


# Sets the DB Adapter, the resetDB flag will make the Adapter recreate
# the database and erase all data every run, you should use this flag
# just in the first time, after the pyevolve.db was created, you can
# omit it.

sqlite_adapter = DBAdapters.DBSQLite(identify="gatear", resetDB=True)
ga.setDBAdapter(sqlite_adapter)

# Do the evolution, with stats dump
# frequency of 20 generations
ga.evolve(freq_stats=1)

# Best individual
print ga.bestIndividual()