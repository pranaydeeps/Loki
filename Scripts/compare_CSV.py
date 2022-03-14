import sys

def main():
	csv1=open(sys.argv[1],'r')
	csvLines=csv1.readlines()
	csv1.close()

	csv2=open(sys.argv[2],'r')
	csvLines2=csv2.readlines()
	csv2.close()



	f= open("comp.txt","w+")

	for i, line in enumerate(csvLines):
		fields = line.split(",")
		#for line2 in csvLines2:
		fields2 = csvLines2[i].split(",")
		if fields[-1] != fields2[-1]:
			print ("different", i, fields[-1], fields2[-1])
		else:
			continue
			# print("same: ", i, fields[-1], fields2[-1])
	
if __name__=="__main__":
	main()
