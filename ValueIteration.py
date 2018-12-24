#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-14 13:31:43
# @Author  : xuhaokui (haokuixu.psy@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import numpy as np 
import time
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
			time0=time.time()
			for stateCurrent in self.stateList:
				valueDict[stateCurrent] = np.max([np.sum([p*(rewardDict[(stateCurrent,action)] + self.decayRate * valueDictOld[stateFuture]) 
					for (stateFuture,p) in transitionProbabilityDict[stateCurrent][action].items()]) for action in self.actionList])
			# print(iterStep,valueDict)
			if np.all(np.array([valueDict[state] - valueDictOld[state] for state in self.stateList])<self.convergeThreshold):
				break
			# print(time.time()-time0)
		return valueDict

class PolicyFromValue():
	def __init__(self,stateList,actionList,decayRate):
		self.stateList=stateList
		self.actionList=actionList
		self.decayRate=decayRate
	def __call__(self,transitionProbabilityDict,rewardDict,valueDict):
		policyDict=dict()
		for state in self.stateList:
			policyDict[state]={action:np.sum([p*(rewardDict[(state,action)] + self.decayRate * valueDict[stateFuture]) 
				for (stateFuture,p) in transitionProbabilityDict[state][action].items()]) for action in self.actionList}
			policyDict[state]={action:np.divide(policyDict[state][action],np.sum(list(policyDict[state].values()))) for action in self.actionList}
		return policyDict

if __name__=="__main__":
	import Transition
	import Reward
	worldRange=[0,0,15,15]
	actionList=[(0,1),(0,-1),(1,0),(-1,0)]
	targetState=((3,12),(7,7))
	targetReward=10
	decayRate=0.9
	convergeThreshold=0.001
	maxIterationStep=100
	stateList=Transition.createStateList(worldRange)

	transitionFunction=Transition.TransitionFromStateAndAction(worldRange)
	createTransitionProbabilityDict=Transition.CreateTransitionProbabilityDict(transitionFunction)
	createRewardDict = Reward.MultiTargetsRewardDict(stateList, actionList, targetReward)
	runValueIteration=ValueIteration(stateList, actionList, decayRate, convergeThreshold, maxIterationStep)
	computePolicyFromValue=PolicyFromValue(stateList, actionList, decayRate)

	time1=time.time()
	transitionProbabilityDict=createTransitionProbabilityDict(stateList, actionList)
	print('prepare transitionDict',time.time()-time1)
	rewardDict=createRewardDict(targetState)
	print('finish prepare')
	valueDict=runValueIteration(transitionProbabilityDict, rewardDict)
	print('finish valueIteration',time.time()-time1)
	print(valueDict)

	policyDict=computePolicyFromValue(transitionProbabilityDict, rewardDict, valueDict)
	print(policyDict[6,4])
	print(policyDict[5,4])
	print(policyDict[5,3])



