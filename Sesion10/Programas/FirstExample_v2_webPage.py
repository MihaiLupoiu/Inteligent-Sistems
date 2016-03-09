# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Documents and Settings\Usuario\.spyder2\.temp.py
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

# Genome instance, 1D List of 50 elements
genome = G1DList.G1DList(50)

# Sets the range max and min of the 1D List
genome.setParams(rangemin=0, rangemax=10)

# The evaluator function (evaluation function)
genome.evaluator.set(eval_func)

# Genetic Algorithm Instance
ga = GSimpleGA.GSimpleGA(genome)

# Set the Roulette Wheel selector method, the number of generations and
# the termination criteria
ga.selector.set(Selectors.GRouletteWheel)
ga.setGenerations(500)
ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)

# Sets the DB Adapter, the resetDB flag will make the Adapter recreate
# the database and erase all data every run, you should use this flag
# just in the first time, after the pyevolve.db was created, you can
# omit it.
sqlite_adapter = DBAdapters.DBSQLite(identify="ex1", resetDB=True)
ga.setDBAdapter(sqlite_adapter)

# Do the evolution, with stats dump
# frequency of 20 generations
ga.evolve(freq_stats=20)

# Best individual
print ga.bestIndividual()