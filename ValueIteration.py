#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-14 13:31:43
# @Author  : xuhaokui (haokuixu.psy@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import numpy as np 
import numba
from numba import jit

def initialValueDict(stateList):
	valueDict={state:np.random.normal() for state in stateList}
	return valueDict

# @jit(nopython=True)
class ValueIteration():
	def __init__(self,stateList,actionList,decayRate,convergeThreshold,maxIterationStep):
		self.stateList=stateList
		self.actionList=actionList
		self.decayRate=decayRate
		self.convergeThreshold=convergeThreshold
		self.maxIterationStep=maxIterationStep
	def __call__(self,transitionProbabilityDict,rewardDict):
		valueDict=initialValueDict(self.stateList)
		# print(valueDict)
		for iterStep in range(self.maxIterationStep):
			valueDictOld = valueDict.copy()
			for stateCurrent in self.stateList:
				valueDict[stateCurrent] = np.max([np.sum([transitionProbabilityDict[(stateFuture,action,stateCurrent)]*(rewardDict[(stateCurrent,action)] + self.decayRate * valueDictOld[stateFuture]) 
					for stateFuture in self.stateList]) for action in self.actionList])
			# print(iterStep,valueDict)
			if np.all(np.array([valueDict[state] - valueDictOld[state] for state in self.stateList])<self.convergeThreshold):
				break
		return valueDict

class PolicyFromValue():
	def __init__(self,stateList,actionList,decayRate):
		self.stateList=stateList
		self.actionList=actionList
		self.decayRate=decayRate
	def __call__(self,transitionProbabilityDict,rewardDict,valueDict):
		policyDict=dict()
		for state in self.stateList:
			policyDict[state]=self.actionList[np.argmax([np.sum([transitionProbabilityDict[(stateFuture,action,state)]*(rewardDict[(state,action)] + self.decayRate * valueDict[stateFuture]) 
				for stateFuture in self.stateList]) for action in self.actionList])]
		return policyDict

if __name__=="__main__":
	import Transition
	import Reward
	worldRange=[0,0,2,2]
	actionList=[(0,1),(0,-1),(1,0),(-1,0)]
	targetState=(1,1)
	targetReward=10
	decayRate=0.9
	convergeThreshold=0.001
	maxIterationStep=100
	stateList=Transition.createStateList(worldRange)

	transitionFunction=Transition.TransitionFromStateAndAction(worldRange)
	createTransitionProbabilityDict=Transition.CreateTransitionProbabilityDict(transitionFunction)
	createRewardDict = Reward.SingleTargetRewardDict(stateList, actionList, targetReward)
	runValueIteration=ValueIteration(stateList, actionList, decayRate, convergeThreshold, maxIterationStep)
	computePolicyFromValue=PolicyFromValue(stateList, actionList, decayRate)

	transitionProbabilityDict=createTransitionProbabilityDict(stateList, actionList)
	rewardDict=createRewardDict(targetState)
	valueDict=runValueIteration(transitionProbabilityDict, rewardDict)
	# print(valueDict)

	policyDict=computePolicyFromValue(transitionProbabilityDict, rewardDict, valueDict)
	print(policyDict)



