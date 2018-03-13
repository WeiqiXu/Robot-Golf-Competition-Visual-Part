"""
class and derived class
"""

import numpy as np

class Base1(object):
	def __init__(self, a1, a2):
		self._a1 = a1
		self._a2 = a2
		
	def printBase1(self):
		print "a1 = ", self._a1
		print "a2 = ", self._a2
		
		
class DeriveB1(Base1):
	def __init__(self, a1, a2, a3):
		super(DeriveB1, self).__init__(a1, a2)
		self._a3 = a3
		
	def printDerive1(self):
		print "a1 = ", self._a1
		print "a2 = ", self._a2
		print "a3 = ", self._a3
		
class DeriveB2(DeriveB1):
	def __init__(self, a1, a2, a3, a4):
		super(DeriveB2, self).__init__(a1, a2, a3)
		self._a4 = a4
		
	def printDerive2(self):
		print "a1 = ", self._a1
		print "a2 = ", self._a2
		print "a3 = ", self._a3
		print "a4 = ", self._a4


derive1 = DeriveB2(1,2,3,4)
derive1.printBase1()
print "--------------"
derive1.printDerive1()
print "--------------"
derive1.printDerive2()
