Cs          = [0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4]
tols        = [0.0001,0.001,0.01,0.1,1]
max_passes  = 5

gammas      = [0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4]

aa          = [0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4]
rr          = [0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4]

total_model_linear = len(Cs)*len(tols)*3
total_model_polynomial = len(Cs)*len(tols)*3
total_model_rbf = len(Cs)*len(tols)*len(gammas)*3
total_model_sigmoid = len(Cs)*len(tols)*len(aa)*len(rr)*3