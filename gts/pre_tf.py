#coding = utf-8
import matplotlib.pyplot as plt      #python中强大的画图模块
from load import*                    #导入和预处理代码写于load.py中，需要用到其中加载和处理后的images28

traffic_signs = [300, 2250, 3650, 4000]      #随机选取

for i in range(len(traffic_signs)):     #i from 0 to 3
	plt.subplot(1, 4, i + 1)
	plt.axis('off')
	plt.imshow(images32[traffic_signs[i]], cmap="gray")
	#你确实必须指定颜色图(即 cmap)，并将其设置为 gray 以给出灰度图像的图表。
	# 这是因为 imshow() 默认使用一种类似热力图的颜色图。
	plt.subplots_adjust(wspace=0.5)       #调整各个图之间的间距

# Show the plot
plt.show()