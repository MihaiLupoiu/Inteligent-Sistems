# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:53:09 2013

@author: Usuario
"""

from pyevolve import *


x = [1, 2, 3, 8, 0, 2, 0, 4, 1, 0]

def eval_func(chromosome):
   score = 0.0

   # iterate over the chromosome elements (items)
   for value in chromosome:
      if value==0:
         score += 1.0

   return score

# Genome instance
genome = G1DList.G1DList(20)

# The evaluator function (objective function)
genome.evaluator.set(eval_func)

genome.setParams(rangemin=0, rangemax=10)

ga = GSimpleGA.GSimpleGA(genome)


# Sets the DB Adapter, the resetDB flag will make the Adapter recreate
# the database and erase all data every run, you should use this flag
# just in the first time, after the pyevolve.db was created, you can
# omit it.
sqlite_adapter = DBAdapters.DBSQLite(identify="ex1", resetDB=True)
ga.setDBAdapter(sqlite_adapter)

# Do the evolution, with stats dump
# frequency of 10 generations
ga.evolve(freq_stats=10)

# Best individual
print ga.bestIndividual()