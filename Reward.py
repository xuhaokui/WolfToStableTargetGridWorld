#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-14 14:26:56
# @Author  : xuhaokui (haokuixu.psy@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import numpy as np 
import itertools as it 

class RewardDict():
	def __init__(self,stateList,actionList,targetReward):
		self.stateList=stateList
		self.actionList=actionList
		self.targetReward=targetReward
	def __call__(self,targetState):
		keyList=list(it.product(self.stateList,self.actionList))
		rewardDict={key:self.targetReward if np.all(key[0]==targetState) else 0 for key in keyList}
		return rewardDict

if __name__=="__main__":
	worldRange=[0,0,3,3]
	actionList=[(0,1),(0,-1),(1,0),(-1,0)]
	targetState=(2,2)
	targetReward=10
	import Transition
	stateList=Transition.createStateList(worldRange)

	createRewardDict = RewardDict(stateList, actionList, targetReward)

	rewardDict=createRewardDict(targetState)
	print(rewardDict)