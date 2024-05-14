
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
import tensorflow.keras as keras
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
import shap

import aumann_shapley_mv_torch as aumann

import torch
import torch.nn as nn

from Dynamask.attribution.mask import Mask
from Dynamask.attribution.perturbation import FadeMovingAverage
from Dynamask.utils.losses import mse
from captum.attr import (
    GradientShap,
    DeepLift,
    IntegratedGradients,
)

def generate_synthetic_data(tag):
	X = []
	if tag == 'mean':
		## Feature 1
		X_synthetic = [] 
		Y = []
		for _ in range(0,50):
		    x1 = np.random.normal(5, 0.2, 100)
		    x2 = np.random.normal(10, 0.03, 100)
		    x3 = np.random.normal(10, 0.1, 100)
		    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
		    X_synthetic.append(np.asarray(t))
		    Y.append(0)
		for _ in range(0,50):
		    x1 = np.random.normal(10, 0.2, 100)
		    x2 = np.random.normal(10, 0.03, 100)
		    x3 = np.random.normal(10, 0.1, 100)
		    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
		    X_synthetic.append(np.asarray(t))
		    Y.append(1)
		X.append(X_synthetic)
	elif tag == 'variance':
		## Feature 1
		X_synthetic = [] 
		Y = []
		for _ in range(0,50):
		    x1 = np.random.normal(10, 2, 100)
		    x2 = np.random.normal(10, 0.03, 100)
		    x3 = np.random.normal(10, 0.1, 100)
		    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
		    X_synthetic.append(np.asarray(t))
		    Y.append(0)
		for _ in range(0,50):
		    x1 = np.random.normal(10, 0.2, 100)
		    x2 = np.random.normal(10, 0.03, 100)
		    x3 = np.random.normal(10, 0.1, 100)
		    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
		    X_synthetic.append(np.asarray(t))
		    Y.append(1)
		X.append(X_synthetic)
	elif tag == 'scale':
		X_synthetic = [] 
		Y = []
		for _ in range(0,50):
		    x1 = np.random.normal(0.2, 0.02, 100)
		    x2 = np.random.normal(0.1, 0.003, 100)
		    x3 = np.random.normal(0.1, 0.01, 100)
		    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
		    X_synthetic.append(np.asarray(t))
		    Y.append(0)
		for _ in range(0,50):
		    x1 = np.random.normal(0.1, 0.02, 100)
		    x2 = np.random.normal(0.1, 0.003, 100)
		    x3 = np.random.normal(0.1, 0.01, 100)
		    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
		    X_synthetic.append(np.asarray(t))
		    Y.append(1)
		X.append(X_synthetic)

	## Feature 2
	X_synthetic = [] 
	for _ in range(0,50):
	    x1 = np.random.normal(10, 0.2, 100)
	    x2 = np.random.normal(10, 0.03, 100)
	    x3 = np.random.normal(10, 0.1, 100)
	    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
	    X_synthetic.append(np.asarray(t))
	for _ in range(0,50):
	    x1 = np.random.normal(10, 0.2, 100)
	    x2 = np.random.normal(10, 0.03, 100)
	    x3 = np.random.normal(10, 0.1, 100)
	    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
	    X_synthetic.append(np.asarray(t))
	X.append(X_synthetic)
	## Feature 3
	X_synthetic = [] 
	for _ in range(0,50):
	    x1 = np.random.normal(10, 0.2, 100)
	    x2 = np.random.normal(10, 0.03, 100)
	    x3 = np.random.normal(10, 0.1, 100)
	    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
	    X_synthetic.append(np.asarray(t))
	for _ in range(0,50):
	    x1 = np.random.normal(10, 0.2, 100)
	    x2 = np.random.normal(10, 0.03, 100)
	    x3 = np.random.normal(10, 0.1, 100)
	    t = list(x1) + list(x2) + list(x3) + list(x2) + list(x1)
	    X_synthetic.append(np.asarray(t))
	X.append(X_synthetic)

	X = np.asarray(X).swapaxes(0,1)
	Y = np.asarray(Y)

	return X,Y



def evaluate_faithfulness(x_test,y_test,model):
	feature_number = x_test.shape[1]
	ts_length = x_test.shape[2]

	ts_aumann = []
	ts_IG = []
	ts_DL = []
	ts_GS = []
	ts_MASK = []

	def f(x):
	    x = x.unsqueeze(0)
	    #x = x.transpose(1, 2)
	    out = model(x)
	    out = out[:, -1]
	    out = torch.nn.Softmax()(out)
	    return out

	predictions = []
	observations = []
	ids = []
	for i in range(len(x_test)):
	    sample_index = i
	    baseline = aumann.generate_baseline(feature_number,ts_length,x_test,y_test[sample_index],model)
	    baseline_torch = torch.from_numpy(baseline.reshape(1,feature_number,ts_length)).type(torch.FloatTensor)
	    sample = torch.from_numpy(x_test[sample_index].reshape(1,feature_number,ts_length)).type(torch.FloatTensor)
	    out_probs =  model(sample).detach().numpy()[0]
	    baseline_probs = model(baseline_torch).detach().numpy()
	    
	    print(baseline_probs)
	    print(out_probs)
	    print(y_test[sample_index])
	    
	    grads = aumann.aumann_shapley(x_test, y_test, sample_index, baseline, model)
	    perturbed_predictions = aumann.evaluate_faithfulness(feature_number,ts_length, x_test, sample_index, y_test[sample_index], np.asarray(grads), model)
	    predictions.append(perturbed_predictions)
	    observations.append(np.arange(feature_number*ts_length))
	    ids.append(["Aumann" for _ in range(feature_number*ts_length)])
	    
	    ig = IntegratedGradients(model)
	    attributions, delta = ig.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    perturbed_predictions = aumann.evaluate_faithfulness(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    predictions.append(perturbed_predictions)
	    observations.append(np.arange(feature_number*ts_length))
	    ids.append(["IG" for _ in range(feature_number*ts_length)])

	    dl = DeepLift(model)
	    attributions, delta = dl.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    perturbed_predictions = aumann.evaluate_faithfulness(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    predictions.append(perturbed_predictions)
	    observations.append(np.arange(feature_number*ts_length))
	    ids.append(["DL" for _ in range(feature_number*ts_length)])


	    gs = GradientShap(model)
	    attributions, delta = gs.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    perturbed_predictions = aumann.evaluate_faithfulness(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    predictions.append(perturbed_predictions)
	    observations.append(np.arange(feature_number*ts_length))
	    ids.append(["GS" for _ in range(feature_number*ts_length)])


	    model = model
	    new_sample = torch.from_numpy(x_test[sample_index].reshape(feature_number,ts_length)).type(torch.FloatTensor)
	    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	    pert = FadeMovingAverage(device)  # This is the perturbation operator
	    mask_saliency = torch.zeros(size=x_test.shape, dtype=torch.float32) 
	    mask = Mask(pert, device, task="classification", verbose=False, deletion_mode=True)
	    mask.fit(
	        X=new_sample,
	        f=f,
	        loss_function=mse,
	        keep_ratio=0.1,
	        target= y_test[i],
	        learning_rate=1.0,
	        size_reg_factor_init=0.1,
	        size_reg_factor_dilation=10000,
	        initial_mask_coeff=0.5,
	        n_epoch=1000,
	        momentum=1.0,
	        time_reg_factor=0,
	    )
	    attributions = mask.mask_tensor.clone().detach().cpu().numpy().reshape(ts_length,feature_number,1)
	   
	    perturbed_predictions = aumann.evaluate_faithfulness(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    predictions.append(perturbed_predictions)
	    observations.append(np.arange(feature_number*ts_length))
	    ids.append(["MASK" for _ in range(feature_number*ts_length)])


	    df_results = pd.DataFrame()
	    df_results['predictions'] = np.asarray(predictions).flatten()
	    df_results['observations'] = np.asarray(observations).flatten()
	    df_results['method'] = np.asarray(ids).flatten()

	return df_results



def evaluate_hits(x_test, y_test, model):
	feature_number = x_test.shape[1]
	ts_length = x_test.shape[2]

	ts_aumann = []
	ts_IG = []
	ts_DL = []
	ts_GS = []
	ts_MASK = []

	def f(x):
	    x = x.unsqueeze(0)
	    #x = x.transpose(1, 2)
	    out = model(x)
	    out = out[:, -1]
	    out = torch.nn.Softmax()(out)
	    return out

	for i in range(0,len(x_test)):
	    sample_index = i
	    baseline = aumann.generate_baseline_original(feature_number,ts_length,x_test)
	    baseline_torch = torch.from_numpy(baseline.reshape(1,feature_number,ts_length)).type(torch.FloatTensor)
	    sample = torch.from_numpy(x_test[sample_index].reshape(1,feature_number,ts_length)).type(torch.FloatTensor)
	    out_probs =  model(sample).detach().numpy()[0]
	    baseline_probs = model(baseline_torch).detach().numpy()
	    
	    print(baseline_probs)
	    print(out_probs)
	    print(y_test[sample_index])
	    
	    grads = aumann.aumann_shapley(x_test, y_test, sample_index, baseline, model)
	    print(np.asarray(grads).shape)
	    ordered_perturbed_df = aumann.generate_order(feature_number,ts_length, x_test, sample_index, y_test[sample_index], np.asarray(grads), model)
	    #ts_aumann.append(ordered_perturbed_ts)
	    hit_count = 0
	    for item in ordered_perturbed_df[len(ordered_perturbed_df)-200:-1].values:
	    	if item[1] == 0:
	    		if item[2] <= 100 or item[2] >= 400:
	    			hit_count += 1

	    ts_aumann.append(hit_count)
	    
	    ig = IntegratedGradients(model)
	    attributions, delta = ig.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    ordered_perturbed_df = aumann.generate_order(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    hit_count = 0
	    for item in ordered_perturbed_df[len(ordered_perturbed_df)-200:-1].values:
	    	if item[1] == 0:
	    		if item[2] < 100 or item[2] > 400:
	    			hit_count += 1
	    ts_IG.append(hit_count)

	    dl = DeepLift(model)
	    attributions, delta = dl.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    ordered_perturbed_df = aumann.generate_order(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    hit_count = 0
	    for item in ordered_perturbed_df[len(ordered_perturbed_df)-200:-1].values:
	    	if item[1] == 0:
	    		if item[2] < 100 or item[2] > 400:
	    			hit_count += 1
	    ts_DL.append(hit_count)

	    gs = GradientShap(model)
	    attributions, delta = gs.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    ordered_perturbed_df = aumann.generate_order(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    hit_count = 0
	    for item in ordered_perturbed_df[len(ordered_perturbed_df)-200:-1].values:
	    	if item[1] == 0:
	    		if item[2] < 100 or item[2] > 400:
	    			hit_count += 1
	    ts_GS.append(hit_count)
	    
	    model = model
	    new_sample = torch.from_numpy(x_test[sample_index].reshape(feature_number,ts_length)).type(torch.FloatTensor)
	    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	    pert = FadeMovingAverage(device)  # This is the perturbation operator
	    mask_saliency = torch.zeros(size=x_test.shape, dtype=torch.float32) 
	    mask = Mask(pert, device, task="classification", verbose=False, deletion_mode=True)
	    mask.fit(
	        X=new_sample,
	        f=f,
	        loss_function=mse,
	        keep_ratio=0.1,
	        target= y_test[i],
	        learning_rate=1.0,
	        size_reg_factor_init=0.1,
	        size_reg_factor_dilation=10000,
	        initial_mask_coeff=0.5,
	        n_epoch=1000,
	        momentum=1.0,
	        time_reg_factor=0,
	    )
	    attributions = mask.mask_tensor.clone().detach().cpu().numpy().reshape(ts_length,feature_number,1)
	   
	    ordered_perturbed_df = aumann.generate_order(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    hit_count = 0
	    for item in ordered_perturbed_df[len(ordered_perturbed_df)-200:-1].values:
	    	if item[1] == 0:
	    		if item[2] < 100 or item[2] > 400:
	    			hit_count += 1
	    ts_MASK.append(hit_count)

	return [ts_aumann,ts_IG,ts_DL,ts_GS,ts_MASK]

def evaluate_accuracy(x_test, y_test, model):

	feature_number = x_test.shape[1]
	ts_length = x_test.shape[2]

	ts_aumann = []
	ts_IG = []
	ts_DL = []
	ts_GS = []
	ts_MASK = []

	def f(x):
	    x = x.unsqueeze(0)
	    #x = x.transpose(1, 2)
	    out = model(x)
	    out = out[:, -1]
	    out = torch.nn.Softmax()(out)
	    return out

	for i in range(0,len(x_test)):
	    sample_index = i
	    baseline = aumann.generate_baseline_original(feature_number,ts_length,x_test)
	    baseline_torch = torch.from_numpy(baseline.reshape(1,feature_number,ts_length)).type(torch.FloatTensor)
	    sample = torch.from_numpy(x_test[sample_index].reshape(1,feature_number,ts_length)).type(torch.FloatTensor)
	    out_probs =  model(sample).detach().numpy()[0]
	    baseline_probs = model(baseline_torch).detach().numpy()
	    
	    print(baseline_probs)
	    print(out_probs)
	    print(y_test[sample_index])
	    
	    grads = aumann.aumann_shapley(x_test, y_test, sample_index, baseline, model)
	    ordered_perturbed_ts = aumann.generate_ordered_perturbed(feature_number,ts_length, x_test, sample_index, y_test[sample_index], np.asarray(grads), model)
	    ts_aumann.append(ordered_perturbed_ts)  
	    
	    ig = IntegratedGradients(model)
	    attributions, delta = ig.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    ts_IG.append(aumann.generate_ordered_perturbed(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model))

	    dl = DeepLift(model)
	    attributions, delta = dl.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    ts_DL.append(aumann.generate_ordered_perturbed(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model))

	    gs = GradientShap(model)
	    attributions, delta = gs.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    ts_GS.append(aumann.generate_ordered_perturbed(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model))
	    
	    
	    model = model
	    new_sample = torch.from_numpy(x_test[sample_index].reshape(feature_number,ts_length)).type(torch.FloatTensor)
	    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	    pert = FadeMovingAverage(device)  # This is the perturbation operator
	    mask_saliency = torch.zeros(size=x_test.shape, dtype=torch.float32) 
	    mask = Mask(pert, device, task="classification", verbose=False, deletion_mode=True)
	    mask.fit(
	        X=new_sample,
	        f=f,
	        loss_function=mse,
	        keep_ratio=0.1,
	        target= y_test[i],
	        learning_rate=1.0,
	        size_reg_factor_init=0.1,
	        size_reg_factor_dilation=10000,
	        initial_mask_coeff=0.5,
	        n_epoch=1000,
	        momentum=1.0,
	        time_reg_factor=0,
	    )
	    attributions = mask.mask_tensor.clone().detach().cpu().numpy().reshape(ts_length,feature_number,1)
	    print(attributions.shape)
	   
	    ordered_perturbed_ts = aumann.generate_ordered_perturbed(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    print(ordered_perturbed_ts)
	    ts_MASK.append(ordered_perturbed_ts)

	accuracies = []
	observations = []
	ids = []

	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_aumann[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    	
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("Aumann")
	    print(i)
	print("Done Aumann")
	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_IG[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("IG")
	    print(i)
	print("Done IG")
	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_DL[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("DL")
	    print(i)
	print("Done DL")
	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_GS[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("GS")
	    print(i)
	print("Done GS")
	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_MASK[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("MASK")
	    print(i)
	print("Done MASK") 

	accuracies = np.asarray(accuracies).flatten()
	observations = np.asarray(observations).flatten()
	ids = np.asarray(ids).flatten()

	df_results = pd.DataFrame()
	df_results['accuracies'] = accuracies
	df_results['observations'] = observations
	df_results['method'] = ids

	return df_results

def evaluate_accuracy_sepsis(x_test, y_test, model):

	feature_number = x_test.shape[1]
	ts_length = x_test.shape[2]

	ts_aumann = []
	ts_IG = []
	ts_DL = []
	ts_GS = []
	ts_MASK = []

	def f(x):
	    x = x.unsqueeze(0)
	    #x = x.transpose(1, 2)
	    out = model(x)
	    out = out[:, -1]
	    out = torch.nn.Softmax()(out)
	    return out

	for i in range(0,len(x_test)):
	    sample_index = i
	    baseline = aumann.generate_baseline_mimic(feature_number,ts_length,x_test,y_test[sample_index])
	    baseline_torch = torch.from_numpy(baseline.reshape(1,feature_number,ts_length)).type(torch.FloatTensor)
	    sample = torch.from_numpy(x_test[sample_index].reshape(1,feature_number,ts_length)).type(torch.FloatTensor)
	    out_probs =  model(sample).detach().numpy()[0]
	    baseline_probs = model(baseline_torch).detach().numpy()
	    

	    print(baseline_probs)
	    print(out_probs)
	    
	    grads = aumann.aumann_shapley(x_test, y_test, sample_index, baseline, model)
	    ordered_perturbed_ts = aumann.generate_ordered_perturbed_mimic(feature_number,ts_length, x_test, sample_index, y_test[sample_index], np.asarray(grads), model)
	    ts_aumann.append(ordered_perturbed_ts)  
	    
	    ig = IntegratedGradients(model)
	    attributions, delta = ig.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    ts_IG.append(aumann.generate_ordered_perturbed_mimic(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model))

	    dl = DeepLift(model)
	    attributions, delta = dl.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    ts_DL.append(aumann.generate_ordered_perturbed_mimic(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model))

	    gs = GradientShap(model)
	    attributions, delta = gs.attribute(sample, baseline_torch, target=int(y_test[sample_index]), return_convergence_delta=True)
	    attributions = attributions.detach().numpy().reshape(ts_length,feature_number,1)
	    ts_GS.append(aumann.generate_ordered_perturbed_mimic(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model))
	    
	    
	    model = model
	    new_sample = torch.from_numpy(x_test[sample_index].reshape(feature_number,ts_length)).type(torch.FloatTensor)
	    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	    pert = FadeMovingAverage(device)  # This is the perturbation operator
	    mask_saliency = torch.zeros(size=x_test.shape, dtype=torch.float32) 
	    mask = Mask(pert, device, task="classification", verbose=False, deletion_mode=True)
	    mask.fit(
	        X=new_sample,
	        f=f,
	        loss_function=mse,
	        keep_ratio=0.1,
	        target= y_test[i],
	        learning_rate=1.0,
	        size_reg_factor_init=0.1,
	        size_reg_factor_dilation=10000,
	        initial_mask_coeff=0.5,
	        n_epoch=1000,
	        momentum=1.0,
	        time_reg_factor=0,
	    )
	    attributions = mask.mask_tensor.clone().detach().cpu().numpy().reshape(ts_length,feature_number,1)
	   
	    ordered_perturbed_ts = aumann.generate_ordered_perturbed_mimic(feature_number, ts_length, x_test, sample_index, y_test[sample_index], attributions, model)
	    ts_MASK.append(ordered_perturbed_ts)

	accuracies = []
	observations = []
	ids = []

	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_aumann[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    	
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("Aumann")
	    print(i)
	print("Done Aumann")
	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_IG[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("IG")
	    print(i)
	print("Done IG")
	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_DL[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("DL")
	    print(i)
	print("Done DL")
	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_GS[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("GS")
	    print(i)
	print("Done GS")
	for i in range(0,int(feature_number)*int(ts_length)):
	    out_classes = []
	    for j in range(0,len(x_test)):
	    
	        out_probs = model(ts_MASK[j][i]).detach().numpy()
	        out_classes.append(np.argmax(out_probs, axis=1))
	    
	    out_classes = np.asarray(out_classes).flatten()
	    accuracy = sum(out_classes == y_test) / len(y_test)
	    accuracies.append(accuracy)
	    observations.append(i)
	    ids.append("MASK")
	    print(i)
	print("Done MASK") 

	accuracies = np.asarray(accuracies).flatten()
	observations = np.asarray(observations).flatten()
	ids = np.asarray(ids).flatten()

	df_results = pd.DataFrame()
	df_results['accuracies'] = accuracies
	df_results['observations'] = observations
	df_results['method'] = ids

	return df_results
	    