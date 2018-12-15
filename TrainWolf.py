#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-14 13:30:21
# @Author  : xuhaokui (haokuixu.psy@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import numpy as np 
import time
import Transition
import Reward
import ValueIteration

class TrainWolfPolicy():
	def __init__(self,stateList,transitionProbabilityDict,createRewardDict,runValueIteration,createPolicyFromValue):
		self.stateList=stateList
		self.transitionProbabilityDict=transitionProbabilityDict
		self.createRewardDict=createRewardDict
		self.runValueIteration=runValueIteration
		self.createPolicyFromValue=createPolicyFromValue
	def __call__(self):
		'''action=policy[sheepPosition][wolfPosition]'''
		wolfPolicy=dict()
		time1=time.time()
		for targetState in self.stateList:
			print(time.time()-time1)
			print(targetState)
			time1=time.time()
			rewardDict=self.createRewardDict(targetState)
			singleTargetStateValueDict=self.runValueIteration(self.transitionProbabilityDict,rewardDict)
			singleTargetStatePolicy=self.createPolicyFromValue(self.transitionProbabilityDict,rewardDict,singleTargetStateValueDict)
			wolfPolicy[targetState]=singleTargetStatePolicy
		return wolfPolicy

if __name__=="__main__":
	time0=time.time()
	worldRange=[0,0,10,10]
	actionList=[(0,1),(0,-1),(1,0),(-1,0)]
	targetReward=10
	decayRate=0.9
	convergeThreshold=0.001
	maxIterationStep=100
	stateList=Transition.createStateList(worldRange)

	print('finish set parameter',time.time()-time0)
	transitionFunction=Transition.TransitionFromStateAndAction(worldRange)
	createTransitionProbabilityDict=Transition.CreateTransitionProbabilityDict(transitionFunction)
	transitionProbabilityDict=createTransitionProbabilityDict(stateList, actionList)
	createRewardDict=Reward.RewardDict(stateList, actionList, targetReward)
	runValueIteration=ValueIteration.ValueIteration(stateList, actionList, decayRate, convergeThreshold, maxIterationStep)
	createPolicyFromValue=ValueIteration.PolicyFromValue(stateList, actionList, decayRate)

	print('finish set function',time.time()-time0)
	trainWolfPolicy = TrainWolfPolicy(stateList,transitionProbabilityDict,createRewardDict,runValueIteration,createPolicyFromValue)
	wolfPolicy=trainWolfPolicy()
	print(wolfPolicy)
	print('finish train policy',time.time()-time0)





