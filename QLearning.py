#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-17 20:00:48
# @Author  : xuhaokui (haokuixu.psy@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import numpy as np 
import random

def initialQDict(stateList,actionList,targetStateList):
	QDict = {state:{action:0 if state in targetStateList else 1 for action in actionList} for state in stateList}
	return QDict

def initialPolicyDict(stateList,actionList):
	policyDict = {state:{action:1.0/len(actionList) for action in actionList} for state in stateList}
	return policyDict

def updatePolicy(policyDict,QDict,state):
	QValueDictForState=QDict[state]
	policyDict[state]={k:np.divide(v,np.sum(list(QValueDictForState.values()))) for k,v in QValueDictForState.items()}
	return policyDict

class QLearning():
	def __init__(self,alpha,gamma,epsilon,segmentTotalNumber,stateList,actionList,transitionFromStateAndAction):
		self.alpha=alpha
		self.gamma=gamma
		self.epsilon=epsilon
		self.segmentTotalNumber=segmentTotalNumber
		self.stateList=stateList
		self.actionList=actionList
		self.transitionFromStateAndAction=transitionFromStateAndAction
	def __call__(self,rewardDict,targetStateList):
		QDict=initialQDict(self.stateList, self.actionList, targetStateList)
		policyDict=initialPolicyDict(self.stateList, self.actionList)
		for segment in range(self.segmentTotalNumber):
			state = random.choice(self.stateList)
			while state in targetStateList:
				state = random.choice(self.stateList)
			keepTraining=True
			while keepTraining:
				if np.random.uniform(0, 1)<self.epsilon:
					action=random.choice(self.actionList)
				else:
					actionMaxList=[action for action in self.actionList if policyDict[state][action]==np.max(list(policyDict[state].values()))]
					action=random.choice(actionMaxList)
				nextState=tuple(self.transitionFromStateAndAction(state,action))
				nextQValueMax=np.max(list(QDict[nextState].values()))
				QDict[state][action]=QDict[state][action]+self.alpha*(rewardDict[(nextState,action)] + self.gamma*nextQValueMax - QDict[state][action])
				policyDict=updatePolicy(policyDict, QDict, state)
				# print('state',state)
				# print('action',action)
				# print(QDict)
				state=nextState
				if state in targetStateList:
					keepTraining=False
		return QDict,policyDict

if __name__=="__main__":
	import Transition
	import Reward
	worldRange=[0,0,2,2]
	actionList=[(0,1),(0,-1),(1,0),(-1,0),(0,0)]
	targetStateList=[(1,1)]
	targetReward=10
	stateList=Transition.createStateList(worldRange)
	alpha=1
	gamma=0.9
	epsilon=0.1
	segmentTotalNumber=1000

	transitionFromStateAndAction=Transition.TransitionFromStateAndAction(worldRange)
	createRewardDict=Reward.MultiTargetsRewardDict(stateList, actionList, targetReward)
	runQLearning=QLearning(alpha, gamma, epsilon, segmentTotalNumber, stateList, actionList, transitionFromStateAndAction)

	rewardDict=createRewardDict(targetStateList)
	[QDict,policyDict]=runQLearning(rewardDict,targetStateList)

	print(QDict)
	print(policyDict)

