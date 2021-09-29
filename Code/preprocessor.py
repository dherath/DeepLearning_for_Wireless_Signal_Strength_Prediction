import numpy as np

#-----------------------------------------
# helper functions for getData()
#-----------------------------------------

def openFile(filename):
    data = []
    with open(filename) as f:
        for line in f:
            words = line.split()
            data.append(words[0])
    return data


def sampleData(dataset,x_length,y_length):
    x_data_limit = len(dataset) - (x_length+y_length)
    X = []
    Y = []
    for i in range(x_data_limit):
        # for the inputs
        temp_x = []
        for j in range(x_length):
            temp_x.append(dataset[i+j])
        X.append(temp_x)
        # for the outputs
        temp_y = []
        for j in range(y_length):
            temp_y.append(dataset[i+x_length+j])
        Y.append(temp_y)
    return X,Y
        

    
#-----------------------------------------
# main method to obtain data
#-----------------------------------------

# obtains the datasets -> used for the RNN model
# filename : the string name for the file
# x_length : length of the input(timesteps of the past)
# y_length : length of output(timesteps into future)
# percentage : the percentage of data to use for training and testing

def getData(filename,x_length,y_length,percentage):
    data = openFile(filename) # open the file and get data

    #-- seperate training and testing --------
    train_size = int(percentage*len(data))
    
    train_data = data[1:train_size]
    test_data = data[train_size+1:-1]

    X_Train,Y_Train = sampleData(train_data,x_length,y_length)
    X_Test,Y_Test = sampleData(test_data,x_length,y_length)

    return X_Train,Y_Train,X_Test,Y_Test
