#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-14 13:30:21
# @Author  : xuhaokui (haokuixu.psy@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import numpy as np 
import itertools as it 
import klepto
import time
import Transition
import Reward
import ValueIteration
import QLearning
import Writer

class TrainWolfPolicyValueIteration():
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
		for targetState in it.product(self.stateList,self.stateList):
			transTargetState = (targetState[1],targetState[0])
			if transTargetState in wolfPolicy.keys():
				continue
			print(time.time()-time1)
			print(targetState)
			time1=time.time()
			rewardDict=self.createRewardDict(list(targetState))
			singleTargetStateValueDict=self.runValueIteration(self.transitionProbabilityDict,rewardDict)
			singleTargetStatePolicy=self.createPolicyFromValue(self.transitionProbabilityDict,rewardDict,singleTargetStateValueDict)
			wolfPolicy[targetState]=singleTargetStatePolicy
		return wolfPolicy

class TrainWolfPolicyQLearning():
	def __init__(self,stateList,createRewardDict,runQLearning):
		self.stateList=stateList
		self.createRewardDict=createRewardDict
		self.runQLearning=runQLearning
	def __call__(self):
		wolfPolicy=dict()
		time1=time.time()
		for targetState in it.product(self.stateList,self.stateList):
			print(time.time()-time1)
			print(targetState)
			time1=time.time()
			rewardDict=self.createRewardDict(list(targetState))
			[QDict,policyDict]=self.runQLearning(rewardDict,list(targetState))
			wolfPolicy[targetState]=policyDict.copy()
		return wolfPolicy

if __name__=="__main__":
	time0=time.time()
	worldRange=[0,0,15,15]
	actionList=[(0,1),(0,-1),(1,0),(-1,0)]
	targetReward=10
	decayRate=0.9
	convergeThreshold=0.001
	maxIterationStep=100
	stateList=Transition.createStateList(worldRange)
	savePolicyFilename='SingleWolfTwoSheepsGrid15'
	alpha=1
	gamma=0.9
	epsilon=0.1
	segmentTotalNumber=1000

	print('finish setting parameter',time.time()-time0)
	transitionFunction=Transition.TransitionFromStateAndAction(worldRange)
	createTransitionProbabilityDict=Transition.CreateTransitionProbabilityDict(transitionFunction)
	transitionFromStateAndAction=Transition.TransitionFromStateAndAction(worldRange)
	transitionProbabilityDict=createTransitionProbabilityDict(stateList, actionList)
	createRewardDict=Reward.MultiTargetsRewardDict(stateList, actionList, targetReward)
	runValueIteration=ValueIteration.ValueIteration(stateList, actionList, decayRate, convergeThreshold, maxIterationStep)
	createPolicyFromValue=ValueIteration.PolicyFromValue(stateList, actionList, decayRate)
	runQLearning=QLearning.QLearning(alpha, gamma, epsilon, segmentTotalNumber, stateList, actionList, transitionFromStateAndAction)

	print('finish setting function',time.time()-time0)
	trainWolfPolicy = TrainWolfPolicyValueIteration(stateList,transitionProbabilityDict,createRewardDict,runValueIteration,createPolicyFromValue)
	# trainWolfPolicy = TrainWolfPolicyQLearning(stateList, createRewardDict, runQLearning)
	wolfPolicy=trainWolfPolicy()
	# print(wolfPolicy)
	print('finish training policy',time.time()-time0)

	print('begin saving policy, please wait')
	Writer.savePolicyToPkl(wolfPolicy, savePolicyFilename)
	# Writer.savePolicyToNpy(wolfPolicy, savePolicyFilename)
	# Writer.savePolicyToJson(wolfPolicy, savePolicyFilename)
	print('finish saving policy, mission complete', time.time()-time0)

	# loadWolfPolicy=klepto.archives.file_archive(savePolicyFilename+'.json')
	# print(loadWolfPolicy.archive[((1,0),(0,1))])







