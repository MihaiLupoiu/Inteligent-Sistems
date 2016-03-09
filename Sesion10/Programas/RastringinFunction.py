# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:00:49 2013

@author: Usuario
"""

from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Mutators, Initializators
from pyevolve import Selectors
from pyevolve import Consts
from pyevolve import DBAdapters
import math

# This is the Rastrigin Function, a deception function
def rastrigin(genome):
   n = len(genome)
   total = 0
   for i in xrange(n):
      total += genome[i]**2 - 10*math.cos(2*math.pi*genome[i])
   return (10*n) + total

def run_main():
   # Genome instance
   genome = G1DList.G1DList(20)
   genome.setParams(rangemin=-5.2, rangemax=5.30, bestrawscore=0.00, rounddecimal=2)
   genome.initializator.set(Initializators.G1DListInitializatorReal)
   genome.mutator.set(Mutators.G1DListMutatorRealGaussian)

   genome.evaluator.set(rastrigin)

   # Genetic Algorithm Instance
   ga = GSimpleGA.GSimpleGA(genome)
   ga.terminationCriteria.set(GSimpleGA.RawScoreCriteria)
   ga.setMinimax(Consts.minimaxType["minimize"])
   ga.setGenerations(3000)
   ga.setCrossoverRate(0.8)
   ga.setPopulationSize(100)
   ga.setMutationRate(0.06)
   #Base de datos
   sqlite_adapter = DBAdapters.DBSQLite(identify="ex2", resetDB=True)
   ga.setDBAdapter(sqlite_adapter)

   ga.evolve(freq_stats=50)

   best = ga.bestIndividual()
   print best

if __name__ == "__main__":
   run_main()