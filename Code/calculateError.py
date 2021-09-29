

#--------------------------------------------
# helper functions
#--------------------------------------------

# combining the data into placeholders
def getPlaceholders(org_data,comp_data):
    y1 = [] # correct values
    y2 = [] # computed values
    for i in range(len(org_data[0])):
        temp1 = []
        temp2 = []
        for j in range(len(org_data)):
            temp1.append(org_data[j][i])
            temp2.append(comp_data[j][i])
        y1.append(temp1)
        y2.append(temp2)
    return y1,y2

#--------------------------------------------
# error calculation
#--------------------------------------------

# MSE : Mean Squared Error
def MSE(org_data,comp_data):
    try:
        if len(org_data) != len(comp_data):
            raise ValueError("length of original Y and computed Y does not match")
        y_org_sample, y_calc_sample = getPlaceholders(org_data,comp_data)
        mse = []
        for n in range(len(y_org_sample)):
            y_org = y_org_sample[n]
            y_calc = y_calc_sample[n]
            sum_value = 0
            for i in range(len(y_org)):
               diff = float(float(y_org[i])-float(y_calc[i]))
               sqrd_diff = diff ** 2
               sum_value += sqrd_diff
            mse.append(float(sum_value/len(y_org)))
        return mse
    except ValueError as err:
        print "Error: ",err

# RMSE : Root Mean Squared Error
def RMSE(org_data,comp_data):
    mse = MSE(org_data,comp_data)
    rmse = []
    for data in mse:
        rmse.append(float(data ** 0.5))
    return rmse

# MAE : mean Absolute Error
def MAE(org_data,comp_data):
    try:
        if len(org_data) != len(comp_data):
            raise ValueError("length of original Y and computed Y does not match")
        y_org_sample, y_calc_sample = getPlaceholders(org_data,comp_data)
        mae = []
        for n in range(len(y_org_sample)):
            y_org = y_org_sample[n]
            y_calc = y_calc_sample[n]
            sum_value = 0
            for i in range(len(y_org)):
                diff = abs(float(y_org[i])-float(y_calc[i]))
                sum_value += diff
            mae.append(float(sum_value/len(y_org)))
        return mae
    except ValueError as err:
        print "Error: ",err
    