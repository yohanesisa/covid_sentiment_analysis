Cs          = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]
tols        = [0.0001,0.001,0.01,0.1,1]
max_passes  = 5

gammas      = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]

aa          = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]
rr          = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]

total_model_linear = len(Cs)*len(tols)*3
total_model_polynomial = len(Cs)*len(tols)*3
total_model_rbf = len(Cs)*len(tols)*len(gammas)*3
total_model_sigmoid = len(Cs)*len(tols)*len(aa)*len(rr)*3