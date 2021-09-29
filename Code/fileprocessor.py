
def writetofile(filename,Y):
    f = open(filename,"w")
    for data in Y:
        temp_string = ""
        for i in range(len(data)-1):
            temp_string += str(data[i]) + " "
        temp_string += str(data[-1]) +"\n"
        f.write(temp_string)
    f.close()
    return

def readfromfile(filename):
    Y = []
    f = open(filename,"r")
    for line in f:
        data = line.split()
        Y.append(data)
    f.close()
    return Y

def writeErrResult(filename,Err):
	f = open(filename,"w")
	for data in Err:
		f.write(str(data)+"\n")
	f.close()
	return