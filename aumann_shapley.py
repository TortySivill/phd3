

import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
import tensorflow.keras as keras
import matplotlib.pyplot as pyplot
from pyearth import Earth


def compute_gradients_static(initial_value, final_value, interpolate_model, steps=50):
    scaled_inputs = np.asarray([initial_value + (float(i)/steps)*(final_value-initial_value) for i in range(0, steps+1)])
    grads = interpolate_model.predict_deriv(scaled_inputs.reshape(scaled_inputs.shape[0],scaled_inputs.shape[1]))
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.average(grads, axis=0)
    avg_grads = avg_grads.flatten()
    integrated_gradients = (final_value-initial_value)*avg_grads  # shape: <inp.shape>
    return integrated_gradients 


def interpolation_static(sample_values, sample_label, model_static):
    sample_predictions = np.asarray([model_static.predict_proba(x.reshape(1,x.shape[0]))[0][sample_label] for x in sample_values]) 
    interpolate_model = Earth()
    interpolate_model.fit(sample_values, sample_predictions)
    return interpolate_model

def aumann_shapley(x_test, y_test, model_static, sample_index, initial_index, final_index):
    sample_values = np.asarray([x_test[sample_index,:,i,:] for i in range(0,x_test.shape[2])])
    sample_values = sample_values.reshape(sample_values.shape[0],sample_values.shape[1])
    sample_label = y_test[sample_index]
    initial_value = x_test[sample_index,:,initial_index,:]
    final_value = x_test[sample_index,:,final_index,:]
    interpolate_model = interpolation_static(sample_values, sample_label, model_static)
    grads = compute_gradients_static(initial_value,final_value,interpolate_model)
    return grads





def interpolation_static_concept(sample_values, sample_label, model_static):
    sample_predictions = np.asarray([model_static.predict_proba(x.reshape(1,x.shape[0]))[0][sample_label] for x in sample_values]) 
    interpolate_model = Earth()
    interpolate_model.fit(sample_values, sample_predictions)
    #print(interpolate_model.summary)
    return interpolate_model

def aumann_shapley_concept(x_test, model_static, sample_label, initial_value, final_value):
    sample_values = np.asarray([x_test[:,i] for i in range(0,x_test.shape[1])])
    sample_values = sample_values.reshape(sample_values.shape[0],sample_values.shape[1])

    interpolate_model = interpolation_static_concept(sample_values.swapaxes(0,1), sample_label, model_static)
    grads = compute_gradients_static(initial_value,final_value,interpolate_model)
    return grads