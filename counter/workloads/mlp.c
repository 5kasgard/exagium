// small deterministic MLP forward pass: 64-128-10, batch of inputs, f32
#include <stdio.h>
#include <stdlib.h>
#define I 64
#define H 128
#define O 10
static float W1[H][I], b1[H], W2[O][H], b2[O];
static unsigned s = 12345; static float r(void){ s = s*1103515245u+12345u; return ((s>>8)&0xffff)/65536.0f-0.5f; }
int main(int argc, char **argv){
  int n = argc > 1 ? atoi(argv[1]) : 1000;
  for(int i=0;i<H;i++){ b1[i]=r(); for(int j=0;j<I;j++) W1[i][j]=r(); }
  for(int i=0;i<O;i++){ b2[i]=r(); for(int j=0;j<H;j++) W2[i][j]=r(); }
  float x[I], h[H]; int correct=0;
  for(int k=0;k<n;k++){
    for(int j=0;j<I;j++) x[j]=r();
    for(int i=0;i<H;i++){ float a=b1[i]; for(int j=0;j<I;j++) a+=W1[i][j]*x[j]; h[i]=a>0?a:0; }
    int best=0; float bv=-1e30f;
    for(int i=0;i<O;i++){ float a=b2[i]; for(int j=0;j<H;j++) a+=W2[i][j]*h[j]; if(a>bv){bv=a;best=i;} }
    correct += best==(k%O);
  }
  printf("%d\n", correct); return 0;
}
