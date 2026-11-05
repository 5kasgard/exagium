// same Sattolo chase as probe/kernels.c, fixed number of laps; only the measured laps are instrumented
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <valgrind/cachegrind.h>
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t xs(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
int main(int c, char **v){
  size_t bytes=strtoull(v[1],0,10), line=64, laps=strtoull(v[2],0,10); uint64_t seed=strtoull(v[3],0,10);
  if(seed) rng=seed;
  size_t n=bytes/line; if(n<2)n=2;
  char *buf=aligned_alloc(2u<<20,(n*line+(2u<<20)-1)/(2u<<20)*(2u<<20));
  size_t *perm=malloc(n*sizeof(size_t)); for(size_t i=0;i<n;i++)perm[i]=i;
  for(size_t i=n-1;i>0;i--){size_t j=xs()%i,t=perm[i];perm[i]=perm[j];perm[j]=t;}
  for(size_t i=0;i<n;i++)*(void**)(buf+i*line)=buf+perm[i]*line;
  free(perm);
  void **p=(void**)buf; for(size_t k=0;k<n;k++)p=(void**)*p;
  size_t steps=laps*n;
  CACHEGRIND_START_INSTRUMENTATION;
  for(size_t k=0;k<steps;k++)p=(void**)*p;
  CACHEGRIND_STOP_INSTRUMENTATION;
  printf("%zu %zu %d\n",n*line,steps,p==0); return 0;
}
