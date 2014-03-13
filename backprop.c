/***********************************************************/
/* This code implements backprop learning for a simple     */
/* two-layer perceptron with sigmoid() activation function */
/* It is neither particularly efficient nor particularly   */
/* elegant.                                                */
/*                                                         */
/* You may use it as a working, though flawed, example     */
/* of how someone might implement backprop in a procedural */
/* language (in this case, ANSI C).                        */
/*                                                         */
/* Note that I've "hard coded" some items that an ideal    */
/* implemention would not have.  If you're going to base   */
/* any of your homework off of this, be warned ;)          */   
/* -jcg                                                    */
/***********************************************************/

    
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Yes... it is evil to #include a source file.  It's really
   lazy and opens one up to potential side effects and HUGE
   compile times if there's lots of that behavior about.
   Fix it if it bothers you :)
   
   Oh... and another thing.... the #included code is for a
   pseudo-random number generator.  It's generally not a 
   good idea to use the built-in random number generators 
   provided you in the standard libraries that come with your
   compiler.  You can google around for info if you like.
   The bottom line is, unless you know what you're doing,
   use a random number generator from a good source.
   As sources go, it doesn't get any better than Don Knuth...
   from whose work the code #included below derives */
   
#include "./rng/rng.c"

/* We're going to base our implementation around a combination 
   of structures that store information about individual neurons
   and matricies that store the weights between layers of 
   neurons.  This is similar to the vector/matrix notiation
   we've adopted in class and is fairly close to how we'd do
   it in Matlab and/or Octave -- except of course here we're 
   responsible for maintaing data structures and walking them
   with code loops */





/* The following is the neuron structure.  We're storing output, gradient
   (which corresponds to "little delta" in the book) and neuron
   bias. A more mature implementation might also store the 
   type of transfer function the neuron uses and allow different neurons
   to have different transfer functions.  In this code, I'm "hard coding" 
   sigmoid() transfer functions.  */
   
         
typedef struct neuron_s
   { double output;        /* sigmoid(receptive_field_of_neuron) */
     double gradient;      /* little delta (local gradient) for this 
	                          neuron, computed presuming sigmoid() 
							  transfer function */
     double bias;          /* In this implementation, we're NOT folding
	                          the bias into the weight vector.
							  We're leaving it separate for clarity of 
							  presentation */
   } neuron_t;
typedef neuron_t *neuron_ptr;


/* Now we give the definition of the structure holding a whole neural network */

typedef struct network_s
   { /* Following are the counts of the number of elements in each
        of THREE layers.  One layer of input units, one layer of
		hidden neurons, and one layer of output neurons */
		
     int input_node_count;      /* The number of input units */
     int hidden_neuron_count;   /* The number of hidden layer neurons */
     int output_neuron_count;   /* The number of output layer neurons */
     
	 /* Following are the actual elements in each layer.  We will 
	    represent the elements as arrays of the appropraite types.
		The input layer, for example, will be a simple array of 
		floating point numbers, because input units are just
		"placeholders" for elements of the input vector.  The hidden
		layer and the output layer ARE made up of neurons, however.
		In this scheme, we're starting our indexing at 0 as per the
		normal C language convention.  Remember, in this code,
		biases are NOT folded into the weight matricies, so
		0 indexed elements are actual inter-unit weights. */
		
     double    *input_layer;    /* pointer to a block of input layer elements */
     neuron_t  *hidden_layer;   /* pointer to a block of neurons */
     neuron_t  *output_layer;   /* pointer to a block of neurons */

     /* Following are arrays to store weights.  Technically, these elements
	    form matrices that have rows and columns equal in number to the
		number of elements in the layer the weights are coming from to 
		and the number of elements in the layer the weights are going to.
		I'm storing them in flattened form in single dimension arrays,
		however.  No big... we'll just compute positions in the one dimensional
		weight arrays from 2 dimensional weight indices.  It's all good. */

     double *first_layer_weights;  /* Weights FROM input layer TO hidden layer  */
     double *second_layer_weights; /* Weights FROM hidden layer TO output layer */
   } network_t;
typedef network_t  *network_ptr;


/***********************************************************/
/* math support routines                                   */
/***********************************************************/

double sigmoid(double x)

/* This function computes a simple sigmoid, its operation 
   should be fairly obvious.  Note that this implementation does
   not support the alpha shape parameter for sigmoid.  Add it
   if you think you need it. */

{ double exp();
  return 1.0 / (1.0 + exp(-x));
}

/***********************************************************/
/* memory allocation / and network support routines        */
/***********************************************************/

void malloc_network(int         input_nodes,
                    int         hidden_nodes,
                    int         output_nodes,
                    network_ptr *network)

/* This function does all the dynamic allocation needed for
   a two-layer perceptron as defined by struct network_s.
   Note that this routine just allocates memory.  It does
   not initialize the network with any preliminary values.
   This is done by init_network().  It does, however, 
   set up node and weight memory allowing for a network
   of <input_nodes> input nodes, <hidden_nodes> hidden 
   layer nodes, and <output_nodes> output neurons.

   NOTE: Though this should be obvious, this routine takes
   in a pointer to the pointer to the network structure.
   This is done so that this routine can change the user
   designated pointer to point to the newly allocated 
   network memory.  If this is mystifying, please consult
   a basic C language reference.  Alternatively switch 
   to a language that does not require pointer voodoo 
   to get things done.
*/

{  *network                        = (network_t*)malloc(sizeof(network_t));
  (*network)->input_node_count     = input_nodes;
  (*network)->hidden_neuron_count  = hidden_nodes;
  (*network)->output_neuron_count  = output_nodes;
  (*network)->input_layer          = (double*)malloc(sizeof(double)*input_nodes);
  (*network)->hidden_layer         = (neuron_t*)malloc(sizeof(neuron_t)*hidden_nodes);
  (*network)->output_layer         = (neuron_t*)malloc(sizeof(neuron_t)*output_nodes);
  (*network)->first_layer_weights  = (double*)malloc(sizeof(double)*input_nodes*hidden_nodes);
  (*network)->second_layer_weights = (double*)malloc(sizeof(double)*hidden_nodes*output_nodes);

}


void free_network(network_ptr *network)

/* This one should be pretty obvious.  It just unallocates 
   any dynamically allocated memory associated with the 
   given pointer.  Standard disclaimers apply.  Don't unallocate
   something you didn't allocate... etc. 

   NOTE: This routine, like malloc_network(), also takes in 
   a pointer to a pointer.  This is because this routine,
   after freeing any memory associated with the user designated
   pointer, sets it back to NULL.  Again, consut a C reference if
   this is mysterious.
*/

{ free((*network)->second_layer_weights);
  free((*network)->first_layer_weights);
  free((*network)->output_layer);
  free((*network)->hidden_layer);
  free((*network)->input_layer);
  free(*network);
  *network = NULL;
}


void init_network_weights(network_ptr network, double max_weight_max, RNG *generator)

/* This routine takes in a pointer to an allocated network, a maximum weight value,
   and a pointer to a random number generator struct (see rnd.c).  It walks the network
   and pre-initializes all weights and biases to values in the range of -max_weight_max
   to max_weight_max.  Be sure you pass in only network structs that have been 
   allocated via malloc_network().
   
   If this were a mature implementation, this routine would actually initialize
   the weights according to a heursitic that took into account the number of inputs
   going into each neuron AND presuming that input patters were preprocessed and
   pre-scaled into a known range of numerical values.  See the book and class
   notes for details.
   
*/

{ int count;
  for (count=0; count < (network->input_node_count * network->hidden_neuron_count); count++)
      network->first_layer_weights[count] = rng_uniform(generator, -max_weight_max, max_weight_max);
  for (count=0; count < (network->hidden_neuron_count * network->output_neuron_count); count++)
      network->second_layer_weights[count] = rng_uniform(generator, -max_weight_max, max_weight_max);
  for (count = 0; count < network->hidden_neuron_count; count++)
      network->hidden_layer[count].bias = rng_uniform(generator, -max_weight_max, max_weight_max);
  for (count = 0; count < network->output_neuron_count; count++)
      network->output_layer[count].bias = rng_uniform(generator, -max_weight_max, max_weight_max);
}


void print_network_parameters(network_ptr network)

/* This is just a utility / debugging routine that prints out all parameters
   of a the network pointed at by <network>.  Since this is a simple multi-layer 
   perceptron, it is completely defined by the settings of all the biases and 
   inter-layer weights.  This routine just... ummm... prints those out 
*/

{ int j,k;
  printf("Input Nodes:  %d\n", network->input_node_count);
  printf("Hidden Nodes: %d\n", network->hidden_neuron_count);
  printf("Output Nodes: %d\n", network->output_neuron_count);
  printf("\n\n");
  printf("Hidden Layer Neurons\n");
  for (j=0; j<network->hidden_neuron_count; j++)
      printf("     Hidden Neuron %d Bias: %f\n", j, network->hidden_layer[j].bias);
  printf("Hidden Layer Weights\n");
  for (j=0; j<network->hidden_neuron_count; j++)
      for (k=0; k<network->input_node_count; k++)
          printf("     hidden weight(%d, %d) = %f\n", j, k, network->first_layer_weights[j*network->input_node_count+k]);
  printf("Output Layer Neurons\n");
  for (j=0; j<network->output_neuron_count; j++)
      printf("     Output Neuron %d Bias: %f\n", j, network->output_layer[j].bias);
  printf("Output Layer Weights\n");
  for (j=0; j<network->output_neuron_count; j++)
      for (k=0; k<network->hidden_neuron_count; k++)
          printf("     output weight(%d, %d) = %f\n", j, k, network->second_layer_weights[j*network->hidden_neuron_count+k]);

}

void print_network_outputs(network_ptr network)  //no output currently

/* Another utility routine.  This one just prints out the outputs and gradients
   associated with every neuron in the network.  
*/

{ int j,k;
  printf("Input Nodes:  %d\n", network->input_node_count);
  printf("Hidden Nodes: %d\n", network->hidden_neuron_count);
  printf("Output Nodes: %d\n", network->output_neuron_count);
  printf("\n\n");

  printf("Hidden Layer Neurons\n");  
  for (j=0; j<network->hidden_neuron_count; j++)
      printf("     Hidden Neuron %d Output: %f    gradient %f\n", j, network->hidden_layer[j].output, network->hidden_layer[j].gradient);

  printf("Output Layer Neurons\n");
  for (j=0; j<network->output_neuron_count; j++)
      printf("     Output Neuron %d Output: %f    gradient %f\n", j, network->output_layer[j].output, network->output_layer[j].gradient);

}

/*************************************************************************/
/* Perceptron Computation Routines (I.E. The important stuff)            */
/*************************************************************************/

void compute_network_outputs(double *x, network_ptr network)

/* This routine takes in a vector of inputs (*x), copies them into the input
   units of the supplied network, and then propagates the inputs forward through
   the network (the feedforward phase).  The output of each neuron is
   placed in the 'output' field of each neuron.  Subsequent calls to this
   routine will, of course, overwrite those values with new outputs
   determined by the inputs provided on the new call 
*/

{ int j,k;
  double receptive_field;

  /* Apply inputs to network by copying them into the network's input 
     units.  Note, we're assuming here that the vector of inputs 
     provided as a parameter has the same length as the number
     of input units in the network.*/  

     for (j=0; j < network->input_node_count; j++)
         network->input_layer[j] = x[j];

  /* Ok... now compute the outputs of all the neurons in the hidden layer.  Note
     that weight references are being flattened into a single dimension array using
     row-major ordering. */

     for (j=0; j < network->hidden_neuron_count; j++)
         { receptive_field = 0.0;
           for (k=0; k < network->input_node_count; k++)
               receptive_field += network->input_layer[k] * network->first_layer_weights[j*network->input_node_count+k];
           receptive_field += network->hidden_layer[j].bias;
           network->hidden_layer[j].output = sigmoid(receptive_field);  // can be changed joshua
          }    
 
   /* ...and.... compute the outputs of all the neurons in the output layer */
 
     for (j=0; j < network->output_neuron_count; j++)
         { receptive_field = 0.0;
           for (k=0; k < network->hidden_neuron_count; k++)
               receptive_field += network->hidden_layer[k].output * network->second_layer_weights[j*network->hidden_neuron_count+k];
           receptive_field += network->output_layer[j].bias;
           network->output_layer[j].output = sigmoid(receptive_field);  // can be changed joshua
          }
}

void compute_network_gradients(double *d, network_ptr network)

/* This routine, given a vector of _desired_ outputs, will compute gradients for
   all nodes in the network pointed at by the network_ptr <network>.  This 
   routine corresponds to the "backward pass" through the network.
   Note that this routine makes use of the input vector and neural outputs
   _already_ stored in the network.  It presumes you made a call to
   compute_network_outputs() to fill those values in. 
*/

{  int j,k;
   double gradient_accumulator;

   /* First, let's do the easy part.  Compute the gradients for all output nodes */
      for (j=0; j<network->output_neuron_count; j++)
          network->output_layer[j].gradient =
                                           (d[j] - network->output_layer[j].output);

   /* Now, let's backpropagate the gradients into the hidden layer */
     for (k=0; k<network->hidden_neuron_count; k++)
         { gradient_accumulator = 0.0;
           for (j=0; j<network->output_neuron_count; j++)
               gradient_accumulator += (network->output_layer[j].gradient *
                                     network->second_layer_weights[j*network->hidden_neuron_count+k]);
           network->hidden_layer[k].gradient = network->hidden_layer[k].output *
                                            (1.0 - network->hidden_layer[k].output) *
                                            gradient_accumulator; //by  transfer fucntion
         }

}
 

void apply_delta_rule(network_ptr network, double learning_rate,double learning_momentum)  // adding learning momentum

 /* This routine will compute and apply delta_w's to all the weights and biases in the
     network.  It PRESUMES that you have already ran compute_network_outputs()
     and compute_network_gradients() to handle both the forward propagation of inputs
     and the back propagation of gradients.  This routine will use inputs and gradients
     alreadly stored in the network by those other two calls to do its work.
     Note, this code (obviously) presumes a set learning rate that is universal
     to all neurons.  You'll have to hack this a bit if you want/need something
     else 
*/

{ int j,k;

  for (j=0; j<network->output_neuron_count; j++)
      { for (k=0; k<network->hidden_neuron_count; k++)
             network->second_layer_weights[j*network->hidden_neuron_count+k] = learning_momentum*network->second_layer_weights[j*network->hidden_neuron_count+k]+ (learning_rate *
                                                                                 network->output_layer[j].gradient *
                                                                                 network->hidden_layer[k].output);
      
        network->output_layer[j].bias = learning_momentum*network->output_layer[j].bias+ (learning_rate * network->output_layer[j].gradient * network->hidden_layer[k].output);
      }

  for (j=0; j<network->hidden_neuron_count; j++)
      { for (k=0; k<network->input_node_count; k++)
             network->first_layer_weights[j*network->input_node_count+k] = learning_momentum*network->first_layer_weights[j*network->input_node_count+k]+(learning_rate *
                                                                                network->hidden_layer[j].gradient *
                                                                                network->input_layer[k]);
    
       network->hidden_layer[j].bias = learning_momentum*network->hidden_layer[j].bias+ (learning_rate * network->hidden_layer[j].gradient * network->input_layer[k] );
      }

}

int main() 
{ int c;
  int random_index;
  double Y;
  
  network_ptr my_network;  /* This is a pointer to the neural net structure that will
                              store our perceptron */

  RNG *rng_one;            /* This holds the current state of a random number generator.
                              see rnd.c if you interested */

  
  /* Note that below I'm not training the outputs to extreme values (0.0 or 1.0) 
     and I'm not inputting extreme values (0.0 or 1.0) on the input units.
     Anyone want to hazard a guess as to why? */
     
  /* Define Input Vectors (XOR inputs) */
  double X[4][2] = { {0.05, 0.05}, {0.05, 0.95}, {0.95, 0.05}, {0.95, 0.95} };  

  /* Define Corresponding Desired Outputs (XOR outputs) */
  double D[4]    = { 0.05, 0.95, 0.95, 0.05 };


  /* Set up random number generator */
   rng_one = rng_create();

   /* Set up hiden layer nodes number */
   int hiden_node_number = 20;
  
   /* Set up neural network */
   malloc_network(2, hiden_node_number, 1, &my_network); /* Two input nodes, two hidden nodes, one output node */

   /* Randomize initial weights */
   init_network_weights(my_network,0.01,rng_one);

   /* Ok... let's let this guy learn the XOR function.  We'll use random selection 
      of input/output pairs and update weights with each trial (non-epoch).  Also,
      we'll hard code for 50,000 presentations.  You might want to change the stopping
      condition to something that makes more sense, like achieving a target gradient */
   int counter = 0;
   int i=0;
   for (c=0; c<500000; c++)
      { random_index = (int)trunc(rng_uniform(rng_one,0.0, 3.9999)); /* select training pair */
        //random_index = c%4;
        //printf("%d\n",random_index);
        compute_network_outputs(X[random_index], my_network);         /* feed inputs forward  */
        compute_network_gradients(&D[random_index], my_network);      /* compute gradients       */
        apply_delta_rule(my_network, 1, 1);                             /* adjust weights       */
        for(i=0;i<my_network->hidden_neuron_count;i++){                
             if(my_network->hidden_layer[i].gradient-0<0.000000000000001 && my_network->hidden_layer[i].gradient-0>-0.000000000000001){ 
                    counter++;
             }
        }
        if(counter==10){          //all gradients of hiden layer are 0.0
             break;
        }
        counter=0;
      } 
      printf("Break at:%d\n",c);         

   
   /* Now, let's see how we did by checking all the inputs against what the trained network
      produces */
	  
   printf("\nNetwork Outputs for Each Training Pattern\n");
   printf("-----------------------------------------\n");
   
   for (c=0; c<4; c++)
       { 
         compute_network_outputs(X[c],my_network); /* Feed input pattern forward */
         Y = my_network->output_layer[0].output;   /* Get output of output layer node 0 */
         printf("X0 = %f      X1=%f      Y=%f\n", X[c][0], X[c][1], Y); /* Print outputs generated by trained network */
       }


   printf("\n");
   printf("Perceptron Architecture \n");
   printf("-----------------------------------------\n");
   
    print_network_parameters(my_network);
    print_network_outputs(my_network);
    free_network(&my_network);
    
    system("pause");

}

