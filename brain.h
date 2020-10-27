#define kBrain_Training

#if 1
	#define kNet_Matrix_Width 64
	#define kNet_Matrix_Hidden_Layers 32
#else
	#define kNet_Matrix_Width 2
	#define kNet_Matrix_Hidden_Layers 1
#endif

/* 
 * layer usage
 * 
 * i -- inputs
 * w -- weights
 * h -- total hidden layers desired
 * w -- weights
 * o -- outputs
 *
 * (1) input_layer
 *   (1) input_weight_layer
 * (h) + hidden_layers			= 1 + (h)
 *   (w) hidden_weight_layer				= 1 + w
 * (1) + output_layer			= 2 + (h)
 * 
 * weight network layers = hidden layers + 1
 * value network layers = hidden layers + 2
 */

#define kWeight_Matrix_Layers (kNet_Matrix_Layers + 1)
#define kWeight_Matrix_Layer_Size (kNet_Matrix_Width * kNet_Matrix_Width)
#define kWeight_Matrix_Elements (kWeight_Matrix_Layer_Size * kWeight_Matrix_Layers)

#define kNet_Matrix_Layers (kNet_Matrix_Hidden_Layers + 2)
#define kHidden_Matrix_Elements (kNet_Matrix_Width * kNet_Matrix_Layers)

#define kBias 1.0

typedef float (*hidden_layer_p)[kNet_Matrix_Width];
typedef float (*weight_layer_p)[kNet_Matrix_Width][kNet_Matrix_Width];

typedef float (*weight_layer_node_p)[kNet_Matrix_Width];

/*
 * hidden layer usage
 * 
 * 0th layer -- inputs
 * kNet_Matrix_Hidden_Layers == total hidden layers desired
 * kNet_Matrix_Layers -- outputs
 * 
 */
 
typedef float (*hidden_network_p)[kNet_Matrix_Layers][kNet_Matrix_Width];
typedef float (*weight_network_p)[kWeight_Matrix_Layers][kNet_Matrix_Width][kNet_Matrix_Width];

typedef struct neuron_t *neuron_p;
typedef struct neuron_t {
	hidden_layer_p		x;
	weight_layer_node_p	w;
	float bias;
}neuron_t;

typedef struct llayer_t *llayer_p;
typedef struct llayer_t {
	hidden_layer_p l0x;
	hidden_layer_p l1x;
	weight_layer_p lw0;
//	bias_layer_p b;
}llayer_t;

typedef struct brain_t **brain_pp;
typedef struct brain_t *brain_p;
typedef struct brain_t {
		weight_network_p	weight_network;
		hidden_network_p	hidden_network;
		hidden_layer_p		delta_error;
		float				net_error;
		float (*activation)(float value);
		float (*activation_derivative)(float value);
}brain_t;

void brain_propagate_network(brain_p brain_matter, float (*output)[kNet_Matrix_Width]);
void brain_load_inputs(brain_p brain_matter, int (*inputs), int count);
int brain_init(brain_p brain_matter);
