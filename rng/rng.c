/******************************************************************************
 * rng.c
 *
 * Assorted routines for pseudo-random number generation.
 *
 * Gregory R. Kramer 
 * Department of Computer Science and Engineering
 * Wright State University 
 * gkramer@cs.wright.edu
 *
 * John C. Gallagher
 * Department of Computer Science and Engineering
 * Wright State University
 * jgallagh@cs.wright.edu
 *
 * Last Revised on 08/03/2003
 *
 * SPECIAL NOTES
 * -------------
 *
 * A good deal of the code in this file is derived from code found in
 * the book 'Numerical Receipies in C' It has been modified to hide
 * the details of random number generation from the user and to allow
 * the user to create more than one random number generator at once.
 * 
 * This version was expanded by jcg to include code for producing 
 * Gaussian distributions and to be backward compatable with his own
 * RAN1.c.  Compatability with the old RAN1 API is handled mainly 
 * through macro definitions that appear in rng.c.  The RAN1 interface
 * for WSU EHRG code is depricated and should NOT be used for new code.
 * You have been warned.... ;)
 *
 *****************************************************************************/

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "rng.h"

/******************************************************************************
 *                  PRIVATE DATA AND FUNCTION DECLARATIONS
 *****************************************************************************/

/******************************************************************************
 * Constants needed by the random number generation algorithm.
 *****************************************************************************/

#define RNG_IA 16807
#define RNG_IM 2147483647
#define RNG_AM (1.0/RNG_IM)
#define RNG_IQ 127773
#define RNG_IR 2836
#define RNG_NTAB 32
#define RNG_NDIV (1+(RNG_IM-1)/RNG_NTAB)
#define RNG_EPS  1.2e-7
#define RNG_RNMX (1.0-RNG_EPS)

/******************************************************************************
 * The RNG struct which stores the data needed for a single random
 * number generator. These fields should only be accessed through the
 * random number generation functions, never directly.
 *****************************************************************************/

struct RNG 
{
  long idum;
  long iv[RNG_NTAB];
  long iy;
};

/******************************************************************************
 * Returns the next value from the specified random number
 * generator. Used only internally by specific random number
 * generation functions like uniform and uniform01.
 *****************************************************************************/

double _rng_next_val(RNG *rng);



/******************************************************************************
 * Utility function used by rng_gaussian. For more info see 'Numerical
 * Recipes in C'
 *****************************************************************************/

void _rng_generate_normals(RNG *rng, double *gX1, double *gX2, double 
                          *gSX);



/******************************************************************************
 * THIS _EVIL_ is brought to you by jcg.... essentially I have a private 
 * RNG struct used for the RAN1 compatabilty macros.  It should never need
 * to be accessed directly.  Really... just don't do it..... unless you're
 * greg and just want to for the hell of it....
 ******************************************************************************/

struct RNG _ran1_rng;

/******************************************************************************
 *                        RNG FUNCTION DEFINITIONS
 *
 *                 FOR FUNCTION AND ARGUMENT DESCRIPTIONS 
 *                    SEE COMMENTS IN RNG HEADER FILE
 *****************************************************************************/



RNG *rng_create() 
{
  RNG *rng;
  void rng_set_seed();
  
  rng = 0;
  rng = malloc(sizeof(RNG));
  if (rng == NULL)
    return NULL;
  rng_set_seed(rng, time(NULL));
  return rng;
}



void rng_destroy(RNG *rng)
{
  if (rng != NULL)
    free(rng);
  rng = NULL;
}



double rng_gaussian(RNG *rng, double mean, double variance)
{ 
  double gX1, gX2, gSX;
  void _rng_generate_normals();
  _rng_generate_normals(rng, &gX1, &gX2, &gSX);
  return(sqrt(variance) * gX1 + mean);
}



double _rng_next_val(RNG *rng) 
{
  int j;
  long k;
  float temp;

  if (rng == NULL)
    return 0.0;

  /* Do random number magic */
  if (rng->idum <= 0 || !rng->iy) 
    { 
      if (-(rng->idum) < 1) 
	rng->idum = 1;	/* Initialize */
      else rng->idum = -(rng->idum);
      for (j=RNG_NTAB+7;j>=0;j--) 
	{ 
	  k=rng->idum/RNG_IQ;
	  rng->idum=RNG_IA*(rng->idum-k*RNG_IQ)-RNG_IR*k;
	  if (rng->idum < 0) 
	    rng->idum += RNG_IM;
	  if (j < RNG_NTAB) 
	    rng->iv[j] = rng->idum;
	}
      rng->iy=rng->iv[0];
    }
  k=rng->idum/RNG_IQ;
  rng->idum=RNG_IA*(rng->idum-k*RNG_IQ)-RNG_IR*k;
  if (rng->idum < 0) 
    rng->idum += RNG_IM;
  j=rng->iy/RNG_NDIV;
  rng->iy=rng->iv[j];
  rng->iv[j] = rng->idum;
  if ((temp=RNG_AM*rng->iy) > RNG_RNMX) 
    return RNG_RNMX;
  else return temp;
}



void _rng_generate_normals(RNG *rng, double *gX1, double *gX2, double *gSX)
{ 
  double v1,v2,s,d;
  double rng_uniform();
  
  /* This is a private function that should never be called by
     a user */
  do
   { 
     v1 = rng_uniform(rng, -1.0,1.0);
     v2 = rng_uniform(rng, -1.0,1.0);
     s = v1 * v1 + v2 * v2;
   }
  while (s >= 1.0);
  *gSX = -2 * log(s);
  d = sqrt(*gSX/s);
  *gX1 = v1 * d;
  *gX2 = v2 * d;
}



void rng_set_seed(RNG *rng, long seed) 
{
  if (rand == NULL)
    return;
  rng->idum = -1 * seed;
  rng->iy = 0;
}



double rng_uniform(RNG *rng, double min, double max)
{
  return min + ((max - min) * _rng_next_val(rng));
}



double rng_uniform01(RNG *rng)
{
  return rng_uniform(rng, 0.0, 1.0);
}

/*****************************************
 * Old RAN1 Style Compatability Functions 
 * (depreicated, not for use in new code)
 *****************************************/

void RAN1_SeedRandom(long seed)    
{ rng_set_seed(&_ran1_rng, seed);}

double RAN1_SimpleRandom()
{ return rng_uniform01(&_ran1_rng);}

double RAN1_UniformRandom(double l, double u)  
{ return rng_uniform(&_ran1_rng, l, u);}

double RAN1_GaussianRandom(double m, double v) 
{ return rng_gaussian(&_ran1_rng, m, v); }

#ifdef RNG_TEST_BLOC

/**************
 * TEST BLOC 
 **************/

int main()
{ RNG *foo;
  int i;
  foo = rng_create();
  printf("Five uniform random numbers\n");
  for (i=0; i < 5; i++)
      printf("%f ", rng_uniform01(foo));
  printf("\n");
  printf("Five Gaussian random numbers\n");
  for (i=0; i < 5; i++)
      printf("%f ", rng_gaussian(foo, 0.0, 1.0));
  printf("\n\n");
  rng_destroy(foo);

  RAN1_SeedRandom((long)time(NULL));
    printf("Five uniform random numbers (ran1 api)\n");
  for (i=0; i < 5; i++)
      printf("%f ", RAN1_SimpleRandom());
  printf("\n");
  printf("Five Gaussian random numbers (ran1 api)\n");
  for (i=0; i < 5; i++)
      printf("%f ", RAN1_GaussianRandom(0.0, 1.0));
  printf("\n");
  return(0);
}

#endif

/* EOF */

  
  
