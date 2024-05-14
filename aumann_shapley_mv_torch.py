

from pyearth import Earth
import numpy as np
import pandas as pd
import torch

def generate_baseline_original(n_features, ts_length, x_test):
	dim_means = []
	for j in range(0,n_features):
		dim_means.append([x_test[:,j,:].mean() for _ in range(0,ts_length)])
	x_baseline = np.concatenate(dim_means).reshape(n_features,ts_length)
	return x_baseline

def generate_baseline_mimic(n_features, ts_length, x_test,sample_index):
	dim_means = []
	for j in range(0,n_features):
		dim_means.append([x_test[sample_index:,j,:].min() for _ in range(0,ts_length)])
	x_baseline = np.concatenate(dim_means).reshape(n_features,ts_length)
	return x_baseline

def generate_baseline(n_features, ts_length, x_test, sample_label, model):
	dim_means = []

	x_test_tensor = torch.from_numpy(x_test).type(torch.FloatTensor)
	results = model(x_test_tensor).detach().numpy()
	baseline_indexes = np.argmin(results,axis=0)
	x_baseline = x_test[baseline_indexes[sample_label]]
	return x_baseline


def generate_baseline_zeros(n_features, ts_length, x_test):
	dim_means = []
	for j in range(0,n_features):
		dim_means.append([0 for _ in range(0,ts_length)])
	x_baseline = np.concatenate(dim_means).reshape(n_features,ts_length)
	return x_baseline

def compute_gradients(initial_value, final_value,  feature_number, interpolate_model, steps=100):
    scaled_inputs = np.asarray([initial_value + (float(i)/steps)*(final_value-initial_value) for i in range(0, steps+1)])
    grads = interpolate_model.predict_deriv(scaled_inputs.reshape(scaled_inputs.shape[0],scaled_inputs.shape[1]))
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.average(grads, axis=0)
    integrated_gradients = (final_value-initial_value).reshape(feature_number,1)*avg_grads
    return integrated_gradients

def interpolation(x_test, y_test, sample_index, x_baseline, model):
	feature_number = x_test.shape[1]
	ts_length = x_test.shape[2]
	x_test_baseline = []
	for i in range(0,x_test.shape[2]):
	    x_baseline_copy = x_baseline.copy()
	    x_baseline_copy[:,:i] = x_test[sample_index,:,:i]
	    x_test_baseline.append(x_baseline_copy)

	x_test_baseline = np.asarray(x_test_baseline)
	sample_values = np.asarray(x_test_baseline)
	sample_label = y_test[sample_index]
	inputs = [torch.from_numpy(x.reshape(1,feature_number,ts_length)).type(torch.FloatTensor) for x in sample_values]
	sample_predictions = [model(x).detach().numpy()[0][sample_label]for x in inputs]
	interpolate_model = Earth(enable_pruning=False)
	interpolate_model.fit(x_test[sample_index].reshape(x_test[sample_index].shape[1],x_test[sample_index].shape[0]), np.asarray(sample_predictions))
	return interpolate_model


def aumann_shapley(x_test, y_test, sample_index, x_baseline, model):
	feature_number = x_test.shape[1]
	interpolate_model = interpolation(x_test, y_test, sample_index, x_baseline, model)
	gradients = [x_baseline[:,0].reshape(feature_number,1)]
	for i in range(0,x_test.shape[2]-1):
	    initial_value = x_test[sample_index][:,i]
	    final_value = x_test[sample_index][:,i+1]
	    gradients.append(compute_gradients(initial_value, final_value, feature_number, interpolate_model))
	return gradients



def evaluate_faithfulness(n_features, ts_length, x_test, sample_index, sample_label, grads, net):
	flattened_grads = []
	features = []
	f_number = list(np.arange(0,n_features))
	timesteps = []
	j = 0
	for timestep in grads:
	    for feature in timestep:
	        flattened_grads.append(np.abs(feature[0]))
	    for i in f_number:
	        features.append(i)
	        timesteps.append(j)
	    j = j + 1
	    
    
	df_values = pd.DataFrame()
	df_values['importances'] = flattened_grads
	df_values['feature'] = features
	df_values['timesteps'] = timesteps

	df_sorted_values = df_values.sort_values(['importances'])
	x_perturbed = x_test[sample_index].copy()
	baseline = generate_baseline_mimic(n_features,ts_length,x_test,sample_index)
	perturbed_predictions = []
	for obs in df_sorted_values[::-1].values:
	    x_perturbed[int(obs[1]),int(obs[2])] = baseline[int(obs[1]),int(obs[2])]
	    x_perturbed_torch = torch.from_numpy(x_perturbed.reshape(1,n_features,ts_length)).type(torch.FloatTensor)
	    perturbed_predictions.append(net(x_perturbed_torch).detach().numpy()[0][sample_label])
	
	return perturbed_predictions  
    

def generate_order(n_features, ts_length, x_test, sample_index, sample_label, grads, net):
	flattened_grads = []
	features = []
	f_number = list(np.arange(0,n_features))
	timesteps = []
	j = 0
	for timestep in grads:
	    for feature in timestep:
	        flattened_grads.append(np.abs(feature[0]))
	    for i in f_number:
	        features.append(i)
	        timesteps.append(j)
	    j = j + 1
	    

	df_values = pd.DataFrame()
	df_values['importances'] = flattened_grads
	df_values['feature'] = features
	df_values['timesteps'] = timesteps

	df_sorted_values = df_values.sort_values(['importances'])

	return df_sorted_values  

def generate_ordered_perturbed(n_features, ts_length, x_test, sample_index, sample_label, grads, net):
	flattened_grads = []
	features = []
	f_number = list(np.arange(0,n_features))
	timesteps = []
	j = 0
	for timestep in grads:
	    for feature in timestep:
	        flattened_grads.append(np.abs(feature[0]))
	    for i in f_number:
	        features.append(i)
	        timesteps.append(j)
	    j = j + 1
	    
    
	df_values = pd.DataFrame()
	df_values['importances'] = flattened_grads
	df_values['feature'] = features
	df_values['timesteps'] = timesteps

	df_sorted_values = df_values.sort_values(['importances'])
	x_perturbed = x_test[sample_index].copy()
	baseline = generate_baseline_original(n_features,ts_length,x_test)
	perturbed_ts = []
	for obs in df_sorted_values[::-1].values:
	    x_perturbed[int(obs[1]),int(obs[2])] = baseline[int(obs[1]),int(obs[2])]
	    perturbed_ts.append(torch.from_numpy(x_perturbed.reshape(1,n_features,ts_length)).type(torch.FloatTensor))

	return np.asarray(perturbed_ts)  

def generate_ordered_perturbed_mimic(n_features, ts_length, x_test, sample_index, sample_label, grads, net):
	flattened_grads = []
	features = []
	f_number = list(np.arange(0,n_features))
	timesteps = []
	j = 0
	for timestep in grads:
	    for feature in timestep:
	        flattened_grads.append(np.abs(feature[0]))
	    for i in f_number:
	        features.append(i)
	        timesteps.append(j)
	    j = j + 1
	    
    
	df_values = pd.DataFrame()
	df_values['importances'] = flattened_grads
	df_values['feature'] = features
	df_values['timesteps'] = timesteps

	df_sorted_values = df_values.sort_values(['importances'])
	x_perturbed = x_test[sample_index].copy()
	baseline = generate_baseline_zeros(n_features,ts_length,x_test)
	perturbed_ts = []
	for obs in df_sorted_values[::-1].values:
	    x_perturbed[int(obs[1]),int(obs[2])] = baseline[int(obs[1]),int(obs[2])]
	    perturbed_ts.append(torch.from_numpy(x_perturbed.reshape(1,n_features,ts_length)).type(torch.FloatTensor))

	return np.asarray(perturbed_ts)  