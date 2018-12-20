#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-14 17:25:44
# @Author  : xuhaokui (haokuixu.psy@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import pickle
import numpy as np 
import os
import klepto 

def savePolicyToPkl(policy,filename):
	filename=filename+'.pkl'
	output=open(filename,'wb')
	pickle.dump(policy, output)
	output.close()
	return

def savePolicyToNpy(policy,filename):
	filename=filename+'.npy'
	np.save(filename,policy)
	return

def savePolicyToJson(policy,filename):
	filename=filename+'.json'
	saveDict=klepto.archives.file_archive(filename,policy)
	saveDict.dump()
	return
