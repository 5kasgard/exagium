/* Independent DRAM read-rate check: scalar 64-bit loads, 16 independent accumulators,
 * 1 GiB array, no vector types, no relation to probe's stream kernel. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    size_t n = (size_t)1 << 27;           /* 128 Mi uint64_t = 1 GiB */
    uint64_t *a = aligned_alloc(4096, n * sizeof(uint64_t));
    if (!a) return 1;
    for (size_t i = 0; i < n; i++) a[i] = (uint64_t)i * 2654435761u;
    uint64_t s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0,s8=0,s9=0,s10=0,s11=0,s12=0,s13=0,s14=0,s15=0;
    double best = 0;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now();
        for (size_t i = 0; i < n; i += 16) {
            s0+=a[i]; s1+=a[i+1]; s2+=a[i+2]; s3+=a[i+3]; s4+=a[i+4]; s5+=a[i+5]; s6+=a[i+6]; s7+=a[i+7];
            s8+=a[i+8]; s9+=a[i+9]; s10+=a[i+10]; s11+=a[i+11]; s12+=a[i+12]; s13+=a[i+13]; s14+=a[i+14]; s15+=a[i+15];
        }
        double dt = now() - t0;
        double gbps = (double)n * 8 / dt / 1e9;
        if (gbps > best) best = gbps;
    }
    printf("independent scalar-read DRAM rate: %.1f GB/s (sink %llu)\n", best,
           (unsigned long long)(s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15));
    free(a);
    return 0;
}
