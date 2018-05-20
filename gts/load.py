#coding = utf-8

#load data

import os    #python中的os模块用于处理文件和目录
import skimage               #python中的skimage模块用于图像处理
import numpy as np           #python中的numpy模块用于科学计算
from skimage import data,transform
from skimage.color import rgb2gray     #rgb2gray将图片转化为灰度

class Load(object):
	def getClassImg(self):
		'''
			get one image from each calsss
		'''
		img = []
		directories=[d for d in os.listdir(self._train_data_directory) if os.path.isdir(os.path.join(self._train_data_directory,d))]
		
		for d in directories:
			#每一类的路径
			label_directory=os.path.join(self._train_data_directory,d)
			file_names=[os.path.join(label_directory,f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
			#file_names is every photo which is end with ".ppm"
			for f in file_names:
				img.append(skimage.data.imread(f)) 
				break
		return img
	def getLable(self,d):
		a = []
		for i in range(0,62):
			a.append(0)
		if(d<len(a)):
			a[d] = 1
		return a
		
	def load_data(self,data_directory):
		'''
			load all image under the data_directory
		'''
		directories=[d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory,d))]
		#d is every classification file
		labels=[]
		images=[]
		
		for d in directories:
			#每一类的路径
			label_directory=os.path.join(data_directory,d)
			file_names=[os.path.join(label_directory,f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
			#file_names is every photo which is end with ".ppm"
			for f in file_names:
				images.append(skimage.data.imread(f))   #read image
				labels.append(self.getLable(int(d)))                   #read label
				self._count += 1
		return images,labels
	
	#images and labels are list

	def __init__(self,path,shell= True):
		self._count = 0
		self._ROOT_PATH=path
		self._train_data_directory=os.path.join(self._ROOT_PATH,"Training")
		self._test_data_directory=os.path.join(self._ROOT_PATH,"Testing")
		
		self.images,self.labels=self.load_data(self._train_data_directory)
		# Rescale the images in the `images` array
		self.images32 = [transform.resize(image, (32, 32)) for image in self.images]
		# Convert `images32` to an array
		self.images32 = np.array(self.images32)
		# Convert `images32` to grayscale
		self.images32 = rgb2gray(self.images32)
		
		self.test_images,self.test_labels = self.load_data(self._test_data_directory)
		# Rescale the images in the `images` array
		self.test_images32 = [transform.resize(image, (32, 32)) for image in self.test_images]
		# Convert `images32` to an array
		self.test_images32 = np.array(self.test_images32)
		# Convert `images32` to grayscale
		self.test_images32 = rgb2gray(self.test_images32)