#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>

static const int trace = 0;

#include "utility.h"
#include "brain.h"
#include "brain_training.h"

#define kEta 0.25 /* 0.1..0.25..1.0 learing rate */
#define kAlpha 0.9 /* 0.1..0.9..1.0? moving rate */

#ifdef kBrain_Training

static void brain_backpropagate_neuron_out(brain_p brain_matter, neuron_p neuron, float error, float div_xo)
{
	for(int i = 0; i < kNet_Matrix_Width; i++)
	{
		float x = neuron->x[0][i];
		float w = neuron->w[0][i];

		float delta_w = kEta * error * x * div_xo;
		
//		printf("((%2.6f, %2.6f) %2.6f) ", w, w + delta_w, delta_w);
		
		neuron->w[0][i] += delta_w;
	}
//	printf("\n");
}

static void brain_backpropagate_layer(brain_p brain_matter, llayer_p layer)
{
	neuron_t neuron;
	
	neuron.x = layer->l0x;
	neuron.bias = kBias;
	
	T("x0=%p, x1=%p, w0=%p\n", layer->l0x, layer->l1x, layer->lw0);

	for(int node = 0; node < kNet_Matrix_Width; node++) {
		neuron.w = (weight_layer_node_p)&layer->lw0[0][node];

		float error = brain_matter->delta_error[0][node];

		float x1 = layer->l1x[0][node];
		float div_xo = brain_matter->activation_derivative(x1);

		brain_backpropagate_neuron_out(brain_matter, &neuron, error, div_xo);
	}

}

static void brain_backpropagate_network(brain_p brain_matter)
{
	llayer_t layer;
	
	T("\n");

	for(int i = 0; i < kNet_Matrix_Layers; i++) {
		int ii = kNet_Matrix_Layers - i;
		layer.l0x = (hidden_layer_p)&brain_matter->hidden_network[0][ii];
		layer.l1x = (hidden_layer_p)&brain_matter->hidden_network[0][ii + 1];
		layer.lw0 = (weight_layer_p)&brain_matter->weight_network[0][ii];
		
		brain_backpropagate_layer(brain_matter, &layer);
	}
}

#endif /* kBrain_Training */

int brain_train_output(
	brain_p brain_matter,
	float *target,
	float (*output)[kNet_Matrix_Width],
	int count)
{
#ifdef kBrain_Training
	hidden_layer_p delta_error = brain_matter->delta_error; 
	T("delta_error=%p-->%p\n", delta_error, delta_error[0]);

	if(!delta_error)
		return(-1);

	hidden_layer_p output_layer =
		(hidden_layer_p)&brain_matter->hidden_network[0][kNet_Matrix_Layers];

	T("output_layer=%p\n",	output_layer);

	float net_error = 0;

	for(int i = 0; i < kNet_Matrix_Width; i++)
	{
		T("output_node_index=%04u\n", i);

		float out = (output[0][i] = output_layer[0][i]);
		float error = ((i < count) ? target[i] : 0);
		error -= out;
		
//		printf("((%02i, %2.6f) %2.6f) ", target[i], out, to_error);
		delta_error[0][i] = error;
		net_error += fabs(error) / kNet_Matrix_Width;
	}
	
//	printf("\n");

	float tsqr_error, tsqr_error_in = brain_matter->net_error;
	
	tsqr_error = tsqr_error_in + powf(net_error, 2.0);
	if(tsqr_error_in)
		tsqr_error *= 0.75;
	
	brain_matter->net_error = tsqr_error;
	
//	printf("net_error=%2.06f, sqrt_error=%2.06f\n", net_error, tsqr_error);
	
	brain_backpropagate_network(brain_matter);

	return(0);
#endif /* kBrain_Training */
	return(-1);
}
