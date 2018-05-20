#coding = utf-8
class Test(object):
	
	def tem(self,b):
		a = []
		for i in range(0,10):
			a.append(0)
		if(b<len(a)):
			a[b]= 1
		return a
		
	def main(self):
		b = []
		for i in range(0,10):
			b.append(self.tem(i))
	
		for i in range(0,10):
			print(b[i])
if __name__ == "__main__":
	t = Test()
	print(__name__)
	t.main()
	
