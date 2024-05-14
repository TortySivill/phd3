

from pyearth import Earth
import numpy as np
import pandas as pd

def generate_baseline(n_features, ts_length, x_test):
	dim_means = []
	for j in range(0,n_features):
		dim_means.append([x_test[:,j,:,:].mean() for _ in range(0,ts_length)])

	x_baseline = np.concatenate(dim_means).reshape(n_features,ts_length,1)
	return x_baseline

def compute_gradients(initial_value, final_value, interpolate_model, steps=100):
    scaled_inputs = np.asarray([initial_value + (float(i)/steps)*(final_value-initial_value) for i in range(0, steps+1)])
    grads = interpolate_model.predict_deriv(scaled_inputs.reshape(scaled_inputs.shape[0],scaled_inputs.shape[1]))
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.average(grads, axis=0)
    print(final_value.shape)
    print("here")
    print(initial_value.shape)
    integrated_gradients = (final_value-initial_value)*avg_grads
    return integrated_gradients

def interpolation(x_test, y_test, sample_index, x_baseline, model):
    x_test_baseline = []
    for i in range(0,x_test.shape[2]):
        x_baseline_copy = x_baseline.copy()
        x_baseline_copy[:,:i,:] = x_test[sample_index,:,:i,:]
        x_test_baseline.append(x_baseline_copy)

    x_test_baseline = np.asarray(x_test_baseline)
    sample_values = np.asarray(x_test_baseline)
    sample_label = y_test[sample_index]
    sample_predictions = [model.predict(x.reshape(1,x.shape[0],x.shape[1],1))[0][sample_label] for x in sample_values]
    interpolate_model = Earth(enable_pruning=False)
    interpolate_model.fit(x_test[sample_index].reshape(x_test[sample_index].shape[1],x_test[sample_index].shape[0]), np.asarray(sample_predictions))
    return interpolate_model


def aumann_shapley(x_test, y_test, sample_index, x_baseline, model):
    interpolate_model = interpolation(x_test, y_test, sample_index, x_baseline, model)
    gradients = []
    for i in range(0,x_test.shape[2]-1):
        initial_value = x_test[sample_index][:,i,:]
        final_value = x_test[sample_index][:,i+1,:]
        gradients.append(compute_gradients(initial_value, final_value, interpolate_model))
    return gradients



def evaluate_faithfulness(n_features, ts_length, x_test, sample_index, grads, model):
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
	dim_means = generate_baseline(n_features, ts_length, x_test)
	perturbed_predictions = []
	for obs in df_sorted_values[::-1].values:
	    x_perturbed[int(obs[1])][int(obs[2])][0] = dim_means[int(obs[1])][0]
	    perturbed_predictions.append( model.predict(x_perturbed.reshape(1, n_features, ts_length, 1))[0][1])

	return perturbed_predictions 