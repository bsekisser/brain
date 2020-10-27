#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>

static const int trace = 0;

#include "utility.h"

#include "brain.h"
#include "brain_activations.h"
#include "brain_training.h"

#define kWeights_Network_Path "weights_network.bin"

#define kWeight_Matrix_Floated_Size (sizeof(float) * kWeight_Matrix_Elements)
#define kHidden_Matrix_Floated_Size (sizeof(float) * kHidden_Matrix_Elements)

#define kHidden_Layer_Floated_Size (sizeof(float) * kNet_Matrix_Width)

#define _T(_w) _w

static float brain_propagate_neuron_out(brain_p brain_matter, neuron_p neuron)
{
	float sum = neuron->bias;

//	float ppv = 1 / kNet_Matrix_Width;

	for(int i = 0; i < kNet_Matrix_Width; i++)
//		sum += ppv * neuron->x[0][i] * neuron->w[0][i];
		sum += ((neuron->x[0][i] * neuron->w[0][i]) / kNet_Matrix_Width);

	sum = brain_matter->activation(sum);

	return(sum);
}

static void brain_propagate_layer(brain_p brain_matter, llayer_p layer)
{
	neuron_t neuron;
	float out;
	
	neuron.x = layer->l0x;
	neuron.bias = kBias;
	
	T("x0=%p, x1=%p, w0=%p\n", layer->l0x, layer->l1x, layer->lw0);

	for(int node = 0; node < kNet_Matrix_Width; node++) {
		neuron.w = (weight_layer_node_p)&layer->lw0[0][node];
	
		out = brain_propagate_neuron_out(brain_matter, &neuron);
		layer->l1x[0][node] = out;
	}
}

void brain_propagate_network(
	brain_p brain_matter,
	float (*output)[kNet_Matrix_Width])
{
	llayer_t layer;

	T("\n");
	
	for(int i = 0; i < kNet_Matrix_Layers; i++) {
		layer.l0x = &brain_matter->hidden_network[0][i];
		layer.l1x = (hidden_layer_p)&brain_matter->hidden_network[0][i + 1];
		layer.lw0 = (weight_layer_p)&brain_matter->weight_network[0][i];
		
		brain_propagate_layer(brain_matter, &layer);
	}

	if(output)
	{
		for(int i = 0; i < kNet_Matrix_Width; i++)
		{
			output[0][i] = 
				brain_matter->hidden_network[0][kNet_Matrix_Layers - 1][i];
		}
	}
}

void brain_load_inputs(brain_p brain_matter, int *inputs, int count)
{
	hidden_layer_p input_layer =
		(hidden_layer_p)&brain_matter->hidden_network[0][0];

	T("input_layer=%p\n",	input_layer);

	float value;

	for(int i = 0; i < kNet_Matrix_Width; i++) {
		T("input_node_index=%04u\n", i);

		value = (float)((i < count) ? inputs[i] : 0);
		
		input_layer[0][i] = value;
	}
}

static int load_weights(weight_network_p weights_network, char *weights_network_path)
{
	FILE *wnf = fopen(weights_network_path, "r");

	T("file=%p, weights_network=%p\n", wnf, weights_network);

	size_t size = 0;
	
	int err = -(wnf == 0);
	
	if(!err) {
		size_t nemb = 1;
		uint32_t matrix_size = 0;
		size = fread(&matrix_size, sizeof(uint32_t), nemb, wnf);
		err = -(size < nemb);
		if(err)
			T_ERR_ERRNO_MSG();
		if(!err && (matrix_size < kWeight_Matrix_Floated_Size))
		{
			T_LOG("matrix_size(%08u) < kWeight_Matrix_Floated_Size(%08u)\n",
				matrix_size, kWeight_Matrix_Floated_Size);
		}
	}
	
	if(!err) {
		size_t nemb = 1;
		uint8_t float_size = 0;
		size = fread(&float_size, sizeof(uint8_t), nemb, wnf);
		err = -(size < nemb);
		if(err)
			T_ERR_ERRNO_MSG();
		if(!err && (float_size < sizeof(float)))
		{
			T_LOG("float_size(%02u) < sizeof(float)(%02u)\n",
				float_size, sizeof(float));
		}
	}
	
	if(!err) {
		size_t nemb = kWeight_Matrix_Elements;
		size = fread(weights_network, sizeof(float), nemb, wnf);
		err = -(size < nemb);
		if(err)
			T_LOG("Failed to read weight_network, loaded %08u of %08u.\n",
				size, nemb);
	}
	
	T("file=%p, nemb=%08i, err=%02u, errno=%04u -- %m\n", wnf, size, err, errno);

	if(wnf)
		fclose(wnf);

	return(err);
}

static void save_weights(weight_network_p weights_network, char *weights_network_path)
{
	FILE *wnf = fopen(weights_network_path, "w");

	T("file=%p, weights_network=%p\n", wnf, weights_network);
	
	size_t size = 0;
	
	int err = -(wnf == 0);

	if(!err) {
		size_t nemb = 1;
		uint32_t matrix_size = kWeight_Matrix_Floated_Size;
		size = fwrite(&matrix_size, sizeof(uint32_t), nemb, wnf);
		err = -(size < nemb);
		if(err)
			T_ERR_ERRNO_MSG();
	}

	if(!err) {
		size_t nemb = 1;
		uint8_t float_size = sizeof(float);
		size = fwrite(&float_size, sizeof(uint8_t), nemb, wnf);
		err = -(size < nemb);
		if(err)
			T_ERR_ERRNO_MSG();
	}

	if(!err) {
		size_t nemb = kWeight_Matrix_Elements;
		size = fwrite(weights_network, sizeof(float), nemb, wnf);
		err = -(size < nemb);
		if(err)
			T_ERR_ERRNO_MSG();
	}

	T("file=%p, nemb=%08i, err=%02u, errno=%04u -- %m\n", wnf, size, err, errno);
	
	if(wnf)
		fclose(wnf);
}

int brain_init(brain_p brain_matter)
{
	weight_network_p weights_network;
	
	weights_network = malloc(kWeight_Matrix_Floated_Size);
	if(!weights_network)
		goto error_fail_alloc_weights;

	hidden_network_p hidden_network;
	hidden_network = malloc(kHidden_Matrix_Floated_Size);
	if(!hidden_network)
		goto error_fail_alloc_hidden;

	brain_matter->weight_network = weights_network;
	brain_matter->hidden_network = hidden_network;

#ifdef kBrain_Training
	hidden_layer_p delta_error = 0;
	delta_error = malloc(kHidden_Layer_Floated_Size);
	if(!delta_error)
		goto error_fail_alloc_delta_error;

	brain_matter->delta_error = delta_error;
	brain_matter->net_error = 0;
#endif

#if 0
	brain_matter->activation = relu_activation;
	brain_matter->activation_derivative = relu_derivative;
#else
	brain_matter->activation = sigmoid_activation;
	brain_matter->activation_derivative = sigmoid_derivative;
#endif

	T("brain=%p, weights_network=%p\n", brain_matter, weights_network);

	int	err = load_weights(weights_network, kWeights_Network_Path);

	if(err && (errno == ENOENT))	{
		for(int x = 0; x < kNet_Matrix_Layers; x++)
			for(int y = 0; y < kNet_Matrix_Width; y++)
				for(int z = 0; z < kNet_Matrix_Width; z++)
					weights_network[0][x][y][z] = float_rand(0.0, 1.0);
		
		save_weights(weights_network, kWeights_Network_Path);
		
		err = 0;
	} if(err)
		goto error_fail_load_weights;

	return(err);

error_fail_load_weights:
error_fail_alloc_delta_error:
	free(hidden_network);
	
error_fail_alloc_hidden:
	free(weights_network);
	
error_fail_alloc_weights:
	T_ERR_ERRNO_MSG();
	return(-1);
}
