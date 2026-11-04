#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  // 1) latency vs working set: pointer chase over a random cycle
  size_t sizes[]={16<<10,32<<10,64<<10,128<<10,256<<10,512<<10,1<<20,2<<20,4<<20,8<<20,16<<20,32<<20,64<<20,128<<20};
  printf("working_set_KB,ns_per_load\n");
  for(int s=0;s<14;s++){
    size_t n=sizes[s]/sizeof(size_t); size_t*a=malloc(n*sizeof(size_t)); size_t*p=malloc(n*sizeof(size_t));
    for(size_t i=0;i<n;i++)p[i]=i;
    srand(1); for(size_t i=n-1;i>0;i--){size_t j=((size_t)rand()*RAND_MAX+rand())%(i+1);size_t t=p[i];p[i]=p[j];p[j]=t;}
    for(size_t i=0;i<n;i++)a[p[i]]=p[(i+1)%n];
    size_t idx=0,steps=20000000; double best=1e9;
    for(int r=0;r<3;r++){double t0=now(); for(size_t k=0;k<steps;k++) idx=a[idx]; double dt=(now()-t0)/steps*1e9; if(dt<best)best=dt;}
    printf("%zu,%.2f\n",sizes[s]>>10,best+ (idx==(size_t)-1));
    free(a);free(p);
  }
  // 2) bandwidth: streaming sum over 256 MB
  size_t n=(256<<20)/sizeof(double); double*b=malloc(n*sizeof(double)); for(size_t i=0;i<n;i++)b[i]=i*0.5;
  double best=1e9,sum=0; for(int r=0;r<5;r++){double t0=now(); for(size_t i=0;i<n;i++)sum+=b[i]; double dt=now()-t0; if(dt<best)best=dt;}
  printf("stream_read_GBps,%.2f\n",(n*sizeof(double))/best/1e9);
  // 3) compute: independent FMAs in registers
  double x[8]={1,1,1,1,1,1,1,1}; long it=200000000; double t0=now();
  for(long i=0;i<it;i++) for(int j=0;j<8;j++) x[j]=x[j]*1.0000001+1e-9;
  double dt=now()-t0; printf("scalar_GFLOPs,%.2f\n",(2.0*8*it)/dt/1e9);
  printf("chk,%f\n",sum+x[0]);
}
