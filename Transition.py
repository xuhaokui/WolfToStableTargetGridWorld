#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-14 13:54:28
# @Author  : xuhaokui (haokuixu.psy@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import numpy as np 
import itertools as it 

def createStateList(worldRange):
	stateList=[(stateX,stateY) for stateX in range(worldRange[2]+1) for stateY in range(worldRange[3]+1)]
	return stateList

class TransitionFromStateAndAction():
	def __init__(self,worldRange):
		self.worldRange=worldRange
	def __call__(self,stateFrom,action):
		state = np.add(stateFrom,action)
		if state[0]<self.worldRange[0] or state[0]>self.worldRange[2] or state[1]<self.worldRange[1] or state[1]>self.worldRange[3]:
			state = stateFrom
		return tuple(state)

class CreateTransitionProbabilityDict():
	def __init__(self,transitionFunction):
		self.transitionFunction=transitionFunction
	def __call__(self,stateList,actionList):
		transitionProbabilityDict={currentState:{action:{self.transitionFunction(currentState,action): 1} for action in actionList} for currentState in stateList}
		return transitionProbabilityDict

if __name__=="__main__":
	worldRange=[0,0,21,21]
	actionList=[(0,1),(0,-1),(1,0),(-1,0)]
	stateList=createStateList(worldRange)
	print(stateList)

	transitionFunction=TransitionFromStateAndAction(worldRange)
	createTransitionProbabilityDict=CreateTransitionProbabilityDict(transitionFunction)

	state=transitionFunction((0,0),(0,-1))
	transitionProbabilityDict=createTransitionProbabilityDict(stateList, actionList)
	print(transitionProbabilityDict)
