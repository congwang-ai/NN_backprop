/******************************************************************************
 * rng.h
 *
 * Assorted routines for multiple, independent pseudo-random number
 * generation.
 *
 * Gregory R. Kramer
 * Department of Computer Science and Engineering
 * Wright State University
 * gkramer@cs.wright.edu
 *
 * Last Revised on 01/06/2003
 *
 * SPECIAL NOTES
 * -------------
 *
 * A good deal of the code in this file is derived from code found in
 * the book 'Numerical Receipies in C' It has been modified to hide
 * the details of random number generation from the user and to allow
 * the user to create more than one random number generator at once.
 *****************************************************************************/

#ifndef RNG_H
#define RNG_H

/******************************************************************************
 * Forward declaration of the RNG struct. The member variables of the
 * RNG struct are private and should never be accessed directly.
 *****************************************************************************/

typedef struct RNG RNG;



/******************************************************************************
 * Returns a pointer to a new RNG structure that is initally seeded to
 * the current time as returned by the time() function in time.h.
 *****************************************************************************/

RNG *rng_create();



/******************************************************************************
 * Frees the memory used by an RNG structure and assigns the pointer the
 * NULL value.
 *****************************************************************************/

void rng_destroy(RNG *rng);



/******************************************************************************
 * Returns a random value from a gaussian distribution with the
 * specified mean and variance.
 *****************************************************************************/

double rng_gaussian(RNG *rng, double mean, double variance);



/******************************************************************************
 * Seeds the RNG struct to the specified seed value.
 *****************************************************************************/

void rng_set_seed(RNG *rng, long seed);



/******************************************************************************
 * Returns a uniform random value in the range (min,max) exclusive of
 * the endpoints.
 *****************************************************************************/

double rng_uniform(RNG *rng, double min, double max);



/******************************************************************************
 * Returns a uniform random value in the range (0,1) exclusive of the
 * endpoints.
 *****************************************************************************/

double rng_uniform01(RNG *rng);


/**********************************************
 * Prototypes for depricated RAN1 calls (jcg)
 **********************************************/
void RAN1_SeedRandom(long seed);    
double RAN1_SimpleRandom();
double RAN1_UniformRandom(double l, double u);  
double RAN1_GaussianRandom(double m, double v); 

#endif
/* EOF */
