	.file	"infer.c"
# GNU C11 (Ubuntu 13.3.0-6ubuntu2~24.04.1) version 13.3.0 (x86_64-linux-gnu)
#	compiled by GNU C version 13.3.0, GMP version 6.3.0, MPFR version 4.2.1, MPC version 1.3.1, isl version isl-0.26-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -mtune=generic -march=x86-64 -O2 -std=c11 -ffp-contract=off -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
	.text
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"rb"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB29:
	.cfi_startproc
	endbr64	
	pushq	%r15	#
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14	#
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13	#
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12	#
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp	#
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx	#
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$824, %rsp	#,
	.cfi_def_cfa_offset 880
# ../../loop/candidates/mlp16/infer.c:22: int main(int argc, char **argv) {
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp869
	movq	%rax, 808(%rsp)	# tmp869, D.3182
	xorl	%eax, %eax	# tmp869
# ../../loop/candidates/mlp16/infer.c:23:   if (argc != 4) return 2;
	cmpl	$4, %edi	#, tmp792
	je	.L2	#,
.L4:
# ../../loop/candidates/mlp16/infer.c:23:   if (argc != 4) return 2;
	movl	$2, %eax	#, <retval>
.L1:
# ../../loop/candidates/mlp16/infer.c:93: }
	movq	808(%rsp), %rdx	# D.3182, tmp870
	subq	%fs:40, %rdx	# MEM[(<address-space-1> long unsigned int *)40B], tmp870
	jne	.L61	#,
	addq	$824, %rsp	#,
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx	#
	.cfi_def_cfa_offset 48
	popq	%rbp	#
	.cfi_def_cfa_offset 40
	popq	%r12	#
	.cfi_def_cfa_offset 32
	popq	%r13	#
	.cfi_def_cfa_offset 24
	popq	%r14	#
	.cfi_def_cfa_offset 16
	popq	%r15	#
	.cfi_def_cfa_offset 8
	ret	
.L2:
	.cfi_restore_state
# ../../loop/candidates/mlp16/infer.c:24:   FILE *f = fopen(argv[1], "rb");
	movq	8(%rsi), %rdi	# MEM[(char * *)argv_97(D) + 8B], MEM[(char * *)argv_97(D) + 8B]
	leaq	.LC0(%rip), %r13	#, tmp398
	movq	%rsi, %rbp	# tmp793, argv
	movq	%r13, %rsi	# tmp398,
	call	fopen@PLT	#
	movq	%rax, %rbx	# tmp794, tmp399
# ../../loop/candidates/mlp16/infer.c:25:   if (!f || fread(T, sizeof T, 1, f) != 1 || fread(B1, sizeof B1, 1, f) != 1 || fread(W2, sizeof W2, 1, f) != 1 ||
	testq	%rax, %rax	# tmp399
	je	.L4	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	movq	%rax, %rcx	# tmp399,
	movl	$1, %edx	#,
	movl	$34816, %esi	#,
	leaq	T(%rip), %rdi	#, ivtmp.41
	call	fread@PLT	#
# ../../loop/candidates/mlp16/infer.c:25:   if (!f || fread(T, sizeof T, 1, f) != 1 || fread(B1, sizeof B1, 1, f) != 1 || fread(W2, sizeof W2, 1, f) != 1 ||
	subq	$1, %rax	#, tmp795
	jne	.L4	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	movq	%rbx, %rcx	# tmp399,
	movl	$1, %edx	#,
	movl	$32, %esi	#,
	leaq	B1(%rip), %rdi	#, tmp401
	call	fread@PLT	#
# ../../loop/candidates/mlp16/infer.c:25:   if (!f || fread(T, sizeof T, 1, f) != 1 || fread(B1, sizeof B1, 1, f) != 1 || fread(W2, sizeof W2, 1, f) != 1 ||
	subq	$1, %rax	#, tmp796
	jne	.L4	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	movq	%rbx, %rcx	# tmp399,
	movl	$1, %edx	#,
	movl	$768, %esi	#,
	leaq	W2(%rip), %rdi	#, tmp402
	call	fread@PLT	#
# ../../loop/candidates/mlp16/infer.c:25:   if (!f || fread(T, sizeof T, 1, f) != 1 || fread(B1, sizeof B1, 1, f) != 1 || fread(W2, sizeof W2, 1, f) != 1 ||
	subq	$1, %rax	#, tmp797
	jne	.L4	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	movq	%rbx, %rcx	# tmp399,
	movl	$1, %edx	#,
	movl	$48, %esi	#,
	leaq	B2(%rip), %rdi	#, tmp403
	call	fread@PLT	#
# ../../loop/candidates/mlp16/infer.c:25:   if (!f || fread(T, sizeof T, 1, f) != 1 || fread(B1, sizeof B1, 1, f) != 1 || fread(W2, sizeof W2, 1, f) != 1 ||
	subq	$1, %rax	#, tmp798
	jne	.L4	#,
# ../../loop/candidates/mlp16/infer.c:26:       fread(B2, sizeof B2, 1, f) != 1 || fgetc(f) != EOF)
	movq	%rbx, %rdi	# tmp399,
	call	fgetc@PLT	#
# ../../loop/candidates/mlp16/infer.c:26:       fread(B2, sizeof B2, 1, f) != 1 || fgetc(f) != EOF)
	addl	$1, %eax	#, tmp799
	jne	.L4	#,
# ../../loop/candidates/mlp16/infer.c:28:   fclose(f);
	movq	%rbx, %rdi	# tmp399,
	call	fclose@PLT	#
# /usr/include/stdlib.h:488:   return strtol (__nptr, (char **) NULL, 10);
	movq	24(%rbp), %rdi	# MEM[(char * *)argv_97(D) + 24B], MEM[(char * *)argv_97(D) + 24B]
	movl	$10, %edx	#,
	xorl	%esi, %esi	#
	call	strtol@PLT	#
	movq	%rax, %r12	# tmp800, _155
# ../../loop/candidates/mlp16/infer.c:30:   if (n < 0) return 2;
	testq	%rax, %rax	# _155
	js	.L4	#,
# ../../loop/candidates/mlp16/infer.c:31:   FILE *xf = fopen(argv[2], "rb");
	movq	16(%rbp), %rdi	# MEM[(char * *)argv_97(D) + 16B], MEM[(char * *)argv_97(D) + 16B]
	movq	%r13, %rsi	# tmp398,
	call	fopen@PLT	#
	movq	%rax, %r14	# tmp801, xf
# ../../loop/candidates/mlp16/infer.c:32:   if (!xf) return 2;
	testq	%rax, %rax	# xf
	je	.L4	#,
# ../../loop/candidates/mlp16/infer.c:33:   unsigned char *x = malloc((size_t)n * 64 + 1);
	movq	%r12, %r15	# _155, tmp408
	salq	$6, %r15	#, tmp408
# ../../loop/candidates/mlp16/infer.c:33:   unsigned char *x = malloc((size_t)n * 64 + 1);
	addq	$1, %r15	#, _7
	movq	%r15, %rdi	# _7,
	call	malloc@PLT	#
# ../../loop/candidates/mlp16/infer.c:34:   char *out = malloc((size_t)n + 2);
	leaq	2(%r12), %rdi	#, tmp410
# ../../loop/candidates/mlp16/infer.c:33:   unsigned char *x = malloc((size_t)n * 64 + 1);
	movq	%rax, %rbx	# tmp802, x
# ../../loop/candidates/mlp16/infer.c:34:   char *out = malloc((size_t)n + 2);
	call	malloc@PLT	#
# ../../loop/candidates/mlp16/infer.c:35:   if (!x || !out) return 2;
	testq	%rbx, %rbx	# x
	sete	%r13b	#, tmp413
# ../../loop/candidates/mlp16/infer.c:35:   if (!x || !out) return 2;
	testq	%rax, %rax	# out
# ../../loop/candidates/mlp16/infer.c:34:   char *out = malloc((size_t)n + 2);
	movq	%rax, %rbp	# tmp803, out
# ../../loop/candidates/mlp16/infer.c:35:   if (!x || !out) return 2;
	sete	%al	#, tmp415
# ../../loop/candidates/mlp16/infer.c:35:   if (!x || !out) return 2;
	orb	%al, %r13b	# tmp415, bad_lsm_flag.30
	jne	.L4	#,
# ../../loop/candidates/mlp16/infer.c:36:   if (n && fread(x, 64, (size_t)n, xf) != (size_t)n) return 2;
	testq	%r12, %r12	# _155
	je	.L7	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:215:   return __fread_chk (__ptr, sz, __size, __n, __stream);
	movq	%r14, %r8	# xf,
	movq	%r12, %rcx	# _155,
	movl	$64, %edx	#,
	movq	%r15, %rsi	# _7,
	movq	%rbx, %rdi	# x,
	call	__fread_chk@PLT	#
# ../../loop/candidates/mlp16/infer.c:36:   if (n && fread(x, 64, (size_t)n, xf) != (size_t)n) return 2;
	cmpq	%rax, %r12	# tmp804, _155
	jne	.L4	#,
.L7:
# ../../loop/candidates/mlp16/infer.c:37:   fclose(xf);
	movq	%r14, %rdi	# xf,
	call	fclose@PLT	#
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:59:   return __builtin___memset_chk (__dest, __ch, __len,
	movdqa	.LC1(%rip), %xmm3	#, tmp644
# ../../loop/candidates/mlp16/infer.c:39:   v16u bad = {0}, lim;
	pxor	%xmm0, %xmm0	# tmp416
# ../../loop/candidates/mlp16/infer.c:41:   for (size_t i = 0; i < (size_t)n * 4; i++) {
	xorl	%eax, %eax	# i
# ../../loop/candidates/mlp16/infer.c:39:   v16u bad = {0}, lim;
	movaps	%xmm0, 720(%rsp)	# tmp416, bad
# ../../loop/candidates/mlp16/infer.c:41:   for (size_t i = 0; i < (size_t)n * 4; i++) {
	leaq	0(,%r12,4), %rcx	#, _20
# ../../loop/candidates/mlp16/infer.c:41:   for (size_t i = 0; i < (size_t)n * 4; i++) {
	movdqa	%xmm0, %xmm1	#, bad_lsm.29
# ../../loop/candidates/mlp16/infer.c:44:     bad |= (v16u)(v > lim);
	movdqa	%xmm0, %xmm2	#, tmp790
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:59:   return __builtin___memset_chk (__dest, __ch, __len,
	movaps	%xmm3, 736(%rsp)	# tmp644, MEM <char[1:16]> [(void *)&lim]
# ../../loop/candidates/mlp16/infer.c:41:   for (size_t i = 0; i < (size_t)n * 4; i++) {
	jmp	.L6	#
.L8:
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29:   return __builtin___memcpy_chk (__dest, __src, __len,
	movq	%rax, %rdx	# i, tmp419
# ../../loop/candidates/mlp16/infer.c:41:   for (size_t i = 0; i < (size_t)n * 4; i++) {
	movl	$1, %r13d	#, bad_lsm_flag.30
	addq	$1, %rax	#, i
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29:   return __builtin___memcpy_chk (__dest, __src, __len,
	salq	$4, %rdx	#, tmp419
# ../../loop/candidates/mlp16/infer.c:44:     bad |= (v16u)(v > lim);
	movdqu	(%rbx,%rdx), %xmm0	# MEM <uint128_t> [(char * {ref-all})x_109 + _635 * 1], MEM <uint128_t> [(char * {ref-all})x_109 + _635 * 1]
	psubusb	%xmm3, %xmm0	# tmp644, tmp423
	pcmpeqb	%xmm2, %xmm0	# tmp790, tmp425
	pcmpeqb	%xmm2, %xmm0	# tmp790, tmp427
# ../../loop/candidates/mlp16/infer.c:44:     bad |= (v16u)(v > lim);
	por	%xmm0, %xmm1	# tmp427, bad_lsm.29
.L6:
# ../../loop/candidates/mlp16/infer.c:41:   for (size_t i = 0; i < (size_t)n * 4; i++) {
	cmpq	%rcx, %rax	# _20, i
	jb	.L8	#,
	testb	%r13b, %r13b	# bad_lsm_flag.30
	je	.L9	#,
	movaps	%xmm1, 720(%rsp)	# bad_lsm.29, bad
.L9:
# ../../loop/candidates/mlp16/infer.c:41:   for (size_t i = 0; i < (size_t)n * 4; i++) {
	xorl	%eax, %eax	# ivtmp.50
	leaq	720(%rsp), %rdx	#, tmp646
.L10:
# ../../loop/candidates/mlp16/infer.c:47:     if (bad[k]) return 2;
	cmpb	$0, (%rdx,%rax)	#, MEM <unsigned char> [(vector(16) unsigned char *)&bad + ivtmp.50_633 * 1]
	jne	.L4	#,
# ../../loop/candidates/mlp16/infer.c:46:   for (int k = 0; k < 16; k++)
	addq	$1, %rax	#, ivtmp.50
	cmpq	$16, %rax	#, ivtmp.50
	jne	.L10	#,
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	48+W2(%rip), %xmm6	# W2[1][0], _268
# ../../loop/candidates/mlp16/infer.c:52:     for (int k = 0; k < H / 8; k++) acc[k] = B1[k];
	movdqa	16+B1(%rip), %xmm7	# B1[1], _636
# ../../loop/candidates/mlp16/infer.c:48:   for (long i = 0; i < n; i++) {
	xorl	%r15d, %r15d	# i
	leaq	34816+T(%rip), %r14	#, _610
# ../../loop/candidates/mlp16/infer.c:70:     v4f s0 = B2[0], s1 = B2[1], s2 = B2[2];
	movaps	B2(%rip), %xmm11	# B2[0], _46
# ../../loop/candidates/mlp16/infer.c:70:     v4f s0 = B2[0], s1 = B2[1], s2 = B2[2];
	movaps	16+B2(%rip), %xmm10	# B2[1], _47
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	%xmm6, (%rsp)	# _268, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	64+W2(%rip), %xmm6	# W2[1][1], _271
# ../../loop/candidates/mlp16/infer.c:52:     for (int k = 0; k < H / 8; k++) acc[k] = B1[k];
	movdqa	B1(%rip), %xmm8	# B1[0], _631
# ../../loop/candidates/mlp16/infer.c:70:     v4f s0 = B2[0], s1 = B2[1], s2 = B2[2];
	movaps	32+B2(%rip), %xmm9	# B2[2], _48
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	W2(%rip), %xmm14	# W2[0][0], _247
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	%xmm6, 16(%rsp)	# _271, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	80+W2(%rip), %xmm6	# W2[1][2], _274
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	16+W2(%rip), %xmm13	# W2[0][1], _250
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	32+W2(%rip), %xmm12	# W2[0][2], _253
	movaps	%xmm6, 32(%rsp)	# _274, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	96+W2(%rip), %xmm6	# W2[2][0], _289
	movaps	%xmm6, 48(%rsp)	# _289, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	112+W2(%rip), %xmm6	# W2[2][1], _292
	movaps	%xmm6, 64(%rsp)	# _292, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	128+W2(%rip), %xmm6	# W2[2][2], _295
	movaps	%xmm6, 80(%rsp)	# _295, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	144+W2(%rip), %xmm6	# W2[3][0], _310
	movaps	%xmm6, 96(%rsp)	# _310, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	160+W2(%rip), %xmm6	# W2[3][1], _313
	movaps	%xmm6, 112(%rsp)	# _313, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	176+W2(%rip), %xmm6	# W2[3][2], _316
	movaps	%xmm6, 128(%rsp)	# _316, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	192+W2(%rip), %xmm6	# W2[4][0], _331
	movaps	%xmm6, 144(%rsp)	# _331, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	208+W2(%rip), %xmm6	# W2[4][1], _334
	movaps	%xmm6, 160(%rsp)	# _334, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	224+W2(%rip), %xmm6	# W2[4][2], _337
	movaps	%xmm6, 176(%rsp)	# _337, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	240+W2(%rip), %xmm6	# W2[5][0], _352
	movaps	%xmm6, 192(%rsp)	# _352, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	256+W2(%rip), %xmm6	# W2[5][1], _355
	movaps	%xmm6, 208(%rsp)	# _355, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	272+W2(%rip), %xmm6	# W2[5][2], _358
	movaps	%xmm6, 224(%rsp)	# _358, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	288+W2(%rip), %xmm6	# W2[6][0], _373
	movaps	%xmm6, 240(%rsp)	# _373, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	304+W2(%rip), %xmm6	# W2[6][1], _376
	movaps	%xmm6, 256(%rsp)	# _376, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	320+W2(%rip), %xmm6	# W2[6][2], _379
	movaps	%xmm6, 272(%rsp)	# _379, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	336+W2(%rip), %xmm6	# W2[7][0], _394
	movaps	%xmm6, 288(%rsp)	# _394, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	352+W2(%rip), %xmm6	# W2[7][1], _397
	movaps	%xmm6, 304(%rsp)	# _397, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	368+W2(%rip), %xmm6	# W2[7][2], _400
	movaps	%xmm6, 320(%rsp)	# _400, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	384+W2(%rip), %xmm6	# W2[8][0], _415
	movaps	%xmm6, 336(%rsp)	# _415, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	400+W2(%rip), %xmm6	# W2[8][1], _418
	movaps	%xmm6, 352(%rsp)	# _418, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	416+W2(%rip), %xmm6	# W2[8][2], _421
	movaps	%xmm6, 368(%rsp)	# _421, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	432+W2(%rip), %xmm6	# W2[9][0], _436
	movaps	%xmm6, 384(%rsp)	# _436, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	448+W2(%rip), %xmm6	# W2[9][1], _439
	movaps	%xmm6, 400(%rsp)	# _439, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	464+W2(%rip), %xmm6	# W2[9][2], _442
	movaps	%xmm6, 416(%rsp)	# _442, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	480+W2(%rip), %xmm6	# W2[10][0], _457
	movaps	%xmm6, 432(%rsp)	# _457, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	496+W2(%rip), %xmm6	# W2[10][1], _460
	movaps	%xmm6, 448(%rsp)	# _460, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	512+W2(%rip), %xmm6	# W2[10][2], _463
	movaps	%xmm6, 464(%rsp)	# _463, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	528+W2(%rip), %xmm6	# W2[11][0], _478
	movaps	%xmm6, 480(%rsp)	# _478, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	544+W2(%rip), %xmm6	# W2[11][1], _481
	movaps	%xmm6, 496(%rsp)	# _481, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	560+W2(%rip), %xmm6	# W2[11][2], _484
	movaps	%xmm6, 512(%rsp)	# _484, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	576+W2(%rip), %xmm6	# W2[12][0], _499
	movaps	%xmm6, 528(%rsp)	# _499, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	592+W2(%rip), %xmm6	# W2[12][1], _502
	movaps	%xmm6, 544(%rsp)	# _502, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	608+W2(%rip), %xmm6	# W2[12][2], _505
	movaps	%xmm6, 560(%rsp)	# _505, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	624+W2(%rip), %xmm6	# W2[13][0], _520
	movaps	%xmm6, 576(%rsp)	# _520, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	640+W2(%rip), %xmm6	# W2[13][1], _523
	movaps	%xmm6, 592(%rsp)	# _523, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	656+W2(%rip), %xmm6	# W2[13][2], _526
	movaps	%xmm6, 608(%rsp)	# _526, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	672+W2(%rip), %xmm6	# W2[14][0], _541
	movaps	%xmm6, 624(%rsp)	# _541, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	688+W2(%rip), %xmm6	# W2[14][1], _544
	movaps	%xmm6, 640(%rsp)	# _544, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	704+W2(%rip), %xmm6	# W2[14][2], _547
	movaps	%xmm6, 656(%rsp)	# _547, %sfp
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	720+W2(%rip), %xmm6	# W2[15][0], _562
	movaps	%xmm6, 672(%rsp)	# _562, %sfp
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	736+W2(%rip), %xmm6	# W2[15][1], _565
	movaps	%xmm6, 688(%rsp)	# _565, %sfp
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	movaps	752+W2(%rip), %xmm6	# W2[15][2], _568
	movaps	%xmm6, 704(%rsp)	# _568, %sfp
# ../../loop/candidates/mlp16/infer.c:67:       a &= (v8s)(a > (v8s){0});
	pxor	%xmm6, %xmm6	# tmp783
.L11:
# ../../loop/candidates/mlp16/infer.c:48:   for (long i = 0; i < n; i++) {
	cmpq	%r12, %r15	# _155, i
	je	.L62	#,
	movq	%r15, %rdx	# i, tmp482
	leaq	T(%rip), %rax	#, ivtmp.41
# ../../loop/candidates/mlp16/infer.c:52:     for (int k = 0; k < H / 8; k++) acc[k] = B1[k];
	movdqa	%xmm7, %xmm0	# _636, acc$1
	salq	$6, %rdx	#, tmp482
	movdqa	%xmm8, %xmm1	# _631, acc$0
	addq	%rbx, %rdx	# x, ivtmp.40
.L12:
# ../../loop/candidates/mlp16/infer.c:58:         const v8s *row = (const v8s *)(tb + q * sizeof T[0] + (size_t)pp[q] * sizeof T[0][0]);
	movzbl	1(%rdx), %r11d	# MEM[(const unsigned char *)_538 + 1B], MEM[(const unsigned char *)_538 + 1B]
	movzbl	2(%rdx), %r10d	# MEM[(const unsigned char *)_538 + 2B], MEM[(const unsigned char *)_538 + 2B]
# ../../loop/candidates/mlp16/infer.c:53:     for (int p = 0; p < 64; p += 8) {
	addq	$4352, %rax	#, ivtmp.41
	addq	$8, %rdx	#, ivtmp.40
# ../../loop/candidates/mlp16/infer.c:58:         const v8s *row = (const v8s *)(tb + q * sizeof T[0] + (size_t)pp[q] * sizeof T[0][0]);
	movzbl	-5(%rdx), %r9d	# MEM[(const unsigned char *)_538 + 3B], MEM[(const unsigned char *)_538 + 3B]
	movzbl	-4(%rdx), %r8d	# MEM[(const unsigned char *)_538 + 4B], MEM[(const unsigned char *)_538 + 4B]
	movzbl	-3(%rdx), %edi	# MEM[(const unsigned char *)_538 + 5B], MEM[(const unsigned char *)_538 + 5B]
	movzbl	-2(%rdx), %esi	# MEM[(const unsigned char *)_538 + 6B], MEM[(const unsigned char *)_538 + 6B]
	addq	$17, %r11	#, tmp485
	addq	$34, %r10	#, tmp487
	movzbl	-1(%rdx), %ecx	# MEM[(const unsigned char *)_538 + 7B], MEM[(const unsigned char *)_538 + 7B]
	addq	$51, %r9	#, tmp489
	addq	$68, %r8	#, tmp491
	salq	$5, %r11	#, _709
	salq	$5, %r10	#, _732
	addq	$85, %rdi	#, tmp493
	addq	$102, %rsi	#, tmp495
	salq	$5, %r9	#, _755
	movzbl	-8(%rdx), %r13d	# MEM[(const unsigned char *)_538], MEM[(const unsigned char *)_538]
	salq	$5, %r8	#, _778
	addq	$119, %rcx	#, tmp497
	salq	$5, %rdi	#, _801
	salq	$5, %rsi	#, _824
	salq	$5, %rcx	#, _847
	salq	$5, %r13	#, _686
# ../../loop/candidates/mlp16/infer.c:60:         for (int k = 0; k < H / 8; k++) acc[k] += row[k];
	paddw	-4352(%rax,%r13), %xmm1	# *_688, tmp498
	paddw	-4336(%rax,%r13), %xmm0	# *_694, tmp505
	paddw	-4352(%rax,%r11), %xmm1	# *_711, tmp499
	paddw	-4336(%rax,%r11), %xmm0	# *_717, tmp506
	paddw	-4352(%rax,%r10), %xmm1	# *_734, tmp500
	paddw	-4336(%rax,%r10), %xmm0	# *_740, tmp507
	paddw	-4352(%rax,%r9), %xmm1	# *_757, tmp501
	paddw	-4336(%rax,%r9), %xmm0	# *_763, tmp508
	paddw	-4352(%rax,%r8), %xmm1	# *_780, tmp502
	paddw	-4336(%rax,%r8), %xmm0	# *_786, tmp509
	paddw	-4352(%rax,%rdi), %xmm1	# *_803, tmp503
	paddw	-4336(%rax,%rdi), %xmm0	# *_809, tmp510
	paddw	-4352(%rax,%rsi), %xmm1	# *_826, tmp504
	paddw	-4336(%rax,%rsi), %xmm0	# *_832, tmp511
	paddw	-4352(%rax,%rcx), %xmm1	# *_849, acc$0
	paddw	-4336(%rax,%rcx), %xmm0	# *_855, acc$1
# ../../loop/candidates/mlp16/infer.c:53:     for (int p = 0; p < 64; p += 8) {
	cmpq	%r14, %rax	# _610, ivtmp.41
	jne	.L12	#,
# ../../loop/candidates/mlp16/infer.c:67:       a &= (v8s)(a > (v8s){0});
	movdqa	%xmm1, %xmm4	# acc$0, tmp513
	movdqa	%xmm0, %xmm3	# acc$1, tmp515
	pcmpgtw	%xmm6, %xmm4	# tmp783, tmp513
	pcmpgtw	%xmm6, %xmm3	# tmp783, tmp515
	pand	%xmm1, %xmm4	# acc$0, _579
	pand	%xmm0, %xmm3	# acc$1, _595
	pextrw	$0, %xmm4, %eax	#, _579, tmp516
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp516
	je	.L39	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm2, %xmm2	# v
	cvtsi2ssl	%eax, %xmm2	# _26, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm2, %xmm2	# tmp518
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	%xmm2, %xmm1	# tmp518, tmp519
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	%xmm2, %xmm0	# tmp518, tmp520
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm14, %xmm1	# _247, tmp519
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	mulps	%xmm13, %xmm0	# _250, tmp520
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	%xmm12, %xmm2	# _253, tmp521
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm11, %xmm1	# _46, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm10, %xmm0	# _47, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm9, %xmm2	# _48, s2
.L13:
	pextrw	$1, %xmm4, %eax	#, _579, tmp522
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp522
	je	.L14	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm5, %xmm5	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	(%rsp), %xmm15	# %sfp, tmp525
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm5	# _27, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm5, %xmm5	# tmp524
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm5, %xmm15	# tmp524, tmp525
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm15, %xmm1	# tmp525, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	16(%rsp), %xmm15	# %sfp, tmp526
	mulps	%xmm5, %xmm15	# tmp524, tmp526
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	32(%rsp), %xmm5	# %sfp, tmp527
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm15, %xmm0	# tmp526, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm5, %xmm2	# tmp527, s2
.L14:
	pextrw	$2, %xmm4, %eax	#, _579, tmp528
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp528
	je	.L15	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm5, %xmm5	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	48(%rsp), %xmm15	# %sfp, tmp531
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm5	# _126, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm5, %xmm5	# tmp530
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm5, %xmm15	# tmp530, tmp531
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm15, %xmm1	# tmp531, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	64(%rsp), %xmm15	# %sfp, tmp532
	mulps	%xmm5, %xmm15	# tmp530, tmp532
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	80(%rsp), %xmm5	# %sfp, tmp533
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm15, %xmm0	# tmp532, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm5, %xmm2	# tmp533, s2
.L15:
	pextrw	$3, %xmm4, %eax	#, _579, tmp534
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp534
	je	.L16	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm5, %xmm5	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	96(%rsp), %xmm15	# %sfp, tmp537
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm5	# _28, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm5, %xmm5	# tmp536
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm5, %xmm15	# tmp536, tmp537
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm15, %xmm1	# tmp537, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	112(%rsp), %xmm15	# %sfp, tmp538
	mulps	%xmm5, %xmm15	# tmp536, tmp538
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	128(%rsp), %xmm5	# %sfp, tmp539
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm15, %xmm0	# tmp538, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm5, %xmm2	# tmp539, s2
.L16:
	pextrw	$4, %xmm4, %eax	#, _579, tmp540
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp540
	je	.L17	#,
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	144(%rsp), %xmm15	# %sfp, tmp543
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm5, %xmm5	# v
	cvtsi2ssl	%eax, %xmm5	# _29, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm5, %xmm5	# tmp542
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm5, %xmm15	# tmp542, tmp543
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm15, %xmm1	# tmp543, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	160(%rsp), %xmm15	# %sfp, tmp544
	mulps	%xmm5, %xmm15	# tmp542, tmp544
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	176(%rsp), %xmm5	# %sfp, tmp545
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm15, %xmm0	# tmp544, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm5, %xmm2	# tmp545, s2
.L17:
	pextrw	$5, %xmm4, %eax	#, _579, tmp546
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp546
	je	.L18	#,
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	192(%rsp), %xmm15	# %sfp, tmp549
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm5, %xmm5	# v
	cvtsi2ssl	%eax, %xmm5	# _30, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm5, %xmm5	# tmp548
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm5, %xmm15	# tmp548, tmp549
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm15, %xmm1	# tmp549, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	208(%rsp), %xmm15	# %sfp, tmp550
	mulps	%xmm5, %xmm15	# tmp548, tmp550
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	224(%rsp), %xmm5	# %sfp, tmp551
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm15, %xmm0	# tmp550, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm5, %xmm2	# tmp551, s2
.L18:
	pextrw	$6, %xmm4, %eax	#, _579, tmp552
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp552
	je	.L19	#,
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	240(%rsp), %xmm15	# %sfp, tmp555
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm5, %xmm5	# v
	cvtsi2ssl	%eax, %xmm5	# _31, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm5, %xmm5	# tmp554
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm5, %xmm15	# tmp554, tmp555
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm15, %xmm1	# tmp555, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	256(%rsp), %xmm15	# %sfp, tmp556
	mulps	%xmm5, %xmm15	# tmp554, tmp556
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	272(%rsp), %xmm5	# %sfp, tmp557
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm15, %xmm0	# tmp556, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm5, %xmm2	# tmp557, s2
.L19:
	pextrw	$7, %xmm4, %eax	#, _579, tmp558
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp558
	je	.L20	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm4, %xmm4	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	288(%rsp), %xmm5	# %sfp, tmp561
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm4	# _32, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm4, %xmm4	# tmp560
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm4, %xmm5	# tmp560, tmp561
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm5, %xmm1	# tmp561, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	304(%rsp), %xmm5	# %sfp, tmp562
	mulps	%xmm4, %xmm5	# tmp560, tmp562
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	320(%rsp), %xmm4	# %sfp, tmp563
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm5, %xmm0	# tmp562, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm4, %xmm2	# tmp563, s2
.L20:
	pextrw	$0, %xmm3, %eax	#, _595, tmp564
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp564
	je	.L21	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm4, %xmm4	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	336(%rsp), %xmm5	# %sfp, tmp567
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm4	# _609, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm4, %xmm4	# tmp566
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm4, %xmm5	# tmp566, tmp567
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm5, %xmm1	# tmp567, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	352(%rsp), %xmm5	# %sfp, tmp568
	mulps	%xmm4, %xmm5	# tmp566, tmp568
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	368(%rsp), %xmm4	# %sfp, tmp569
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm5, %xmm0	# tmp568, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm4, %xmm2	# tmp569, s2
.L21:
	pextrw	$1, %xmm3, %eax	#, _595, tmp570
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp570
	je	.L22	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm4, %xmm4	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	384(%rsp), %xmm5	# %sfp, tmp573
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm4	# _613, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm4, %xmm4	# tmp572
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm4, %xmm5	# tmp572, tmp573
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm5, %xmm1	# tmp573, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	400(%rsp), %xmm5	# %sfp, tmp574
	mulps	%xmm4, %xmm5	# tmp572, tmp574
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	416(%rsp), %xmm4	# %sfp, tmp575
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm5, %xmm0	# tmp574, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm4, %xmm2	# tmp575, s2
.L22:
	pextrw	$2, %xmm3, %eax	#, _595, tmp576
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp576
	je	.L23	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm4, %xmm4	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	432(%rsp), %xmm5	# %sfp, tmp579
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm4	# _614, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm4, %xmm4	# tmp578
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm4, %xmm5	# tmp578, tmp579
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm5, %xmm1	# tmp579, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	448(%rsp), %xmm5	# %sfp, tmp580
	mulps	%xmm4, %xmm5	# tmp578, tmp580
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	464(%rsp), %xmm4	# %sfp, tmp581
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm5, %xmm0	# tmp580, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm4, %xmm2	# tmp581, s2
.L23:
	pextrw	$3, %xmm3, %eax	#, _595, tmp582
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp582
	je	.L24	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm4, %xmm4	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	480(%rsp), %xmm5	# %sfp, tmp585
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm4	# _615, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm4, %xmm4	# tmp584
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm4, %xmm5	# tmp584, tmp585
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm5, %xmm1	# tmp585, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	496(%rsp), %xmm5	# %sfp, tmp586
	mulps	%xmm4, %xmm5	# tmp584, tmp586
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	512(%rsp), %xmm4	# %sfp, tmp587
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm5, %xmm0	# tmp586, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm4, %xmm2	# tmp587, s2
.L24:
	pextrw	$4, %xmm3, %eax	#, _595, tmp588
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp588
	je	.L25	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm4, %xmm4	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	528(%rsp), %xmm5	# %sfp, tmp591
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm4	# _616, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm4, %xmm4	# tmp590
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm4, %xmm5	# tmp590, tmp591
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm5, %xmm1	# tmp591, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	544(%rsp), %xmm5	# %sfp, tmp592
	mulps	%xmm4, %xmm5	# tmp590, tmp592
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	560(%rsp), %xmm4	# %sfp, tmp593
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm5, %xmm0	# tmp592, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm4, %xmm2	# tmp593, s2
.L25:
	pextrw	$5, %xmm3, %eax	#, _595, tmp594
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp594
	je	.L26	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm4, %xmm4	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	576(%rsp), %xmm5	# %sfp, tmp597
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm4	# _620, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm4, %xmm4	# tmp596
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm4, %xmm5	# tmp596, tmp597
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm5, %xmm1	# tmp597, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	592(%rsp), %xmm5	# %sfp, tmp598
	mulps	%xmm4, %xmm5	# tmp596, tmp598
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	608(%rsp), %xmm4	# %sfp, tmp599
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm5, %xmm0	# tmp598, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm4, %xmm2	# tmp599, s2
.L26:
	pextrw	$6, %xmm3, %eax	#, _595, tmp600
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp600
	je	.L27	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm4, %xmm4	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	624(%rsp), %xmm5	# %sfp, tmp603
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm4	# _623, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm4, %xmm4	# tmp602
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm4, %xmm5	# tmp602, tmp603
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm5, %xmm1	# tmp603, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	640(%rsp), %xmm5	# %sfp, tmp604
	mulps	%xmm4, %xmm5	# tmp602, tmp604
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	656(%rsp), %xmm4	# %sfp, tmp605
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm5, %xmm0	# tmp604, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm4, %xmm2	# tmp605, s2
.L27:
	pextrw	$7, %xmm3, %eax	#, _595, tmp606
# ../../loop/candidates/mlp16/infer.c:73:       if (!h[j]) continue;
	testw	%ax, %ax	# tmp606
	je	.L28	#,
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cwtl
	pxor	%xmm3, %xmm3	# v
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	movaps	672(%rsp), %xmm4	# %sfp, tmp609
# ../../loop/candidates/mlp16/infer.c:74:       float v = (float)h[j];
	cvtsi2ssl	%eax, %xmm3	# _624, v
# ../../loop/candidates/mlp16/infer.c:75:       v4f b = {v, v, v, v};
	shufps	$0, %xmm3, %xmm3	# tmp608
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	mulps	%xmm3, %xmm4	# tmp608, tmp609
# ../../loop/candidates/mlp16/infer.c:76:       s0 += b * W2[j][0];
	addps	%xmm4, %xmm1	# tmp609, s0
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	movaps	688(%rsp), %xmm4	# %sfp, tmp610
	mulps	%xmm3, %xmm4	# tmp608, tmp610
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	mulps	704(%rsp), %xmm3	# %sfp, tmp611
# ../../loop/candidates/mlp16/infer.c:77:       s1 += b * W2[j][1];
	addps	%xmm4, %xmm0	# tmp610, s1
# ../../loop/candidates/mlp16/infer.c:78:       s2 += b * W2[j][2];
	addps	%xmm3, %xmm2	# tmp611, s2
.L28:
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movaps	%xmm1, %xmm3	# s0, tmp613
	xorl	%edx, %edx	# best
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29:   return __builtin___memcpy_chk (__dest, __src, __len,
	movaps	%xmm1, 752(%rsp)	# s0, MEM <uint128_t> [(char * {ref-all})&s]
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	shufps	$85, %xmm1, %xmm3	#, s0, tmp613
	comiss	%xmm1, %xmm3	# tmp614, tmp612
	movaps	%xmm1, %xmm3	# s0, tmp619
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29:   return __builtin___memcpy_chk (__dest, __src, __len,
	movaps	%xmm0, 768(%rsp)	# s1, MEM <uint128_t> [(char * {ref-all})&s + 16B]
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	unpckhps	%xmm1, %xmm3	# s0, tmp619
# /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29:   return __builtin___memcpy_chk (__dest, __src, __len,
	movaps	%xmm2, 784(%rsp)	# s2, MEM <uint128_t> [(char * {ref-all})&s + 32B]
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	seta	%al	#, best
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	seta	%dl	#, best
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movzbl	%al, %eax	# best, best
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	comiss	752(%rsp,%rax,4), %xmm3	# s[_605], tmp618
	jbe	.L29	#,
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movl	$2, %edx	#, best
	movl	$2, %eax	#, best
.L29:
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	shufps	$255, %xmm1, %xmm1	#, s0, tmp622
	comiss	752(%rsp,%rax,4), %xmm1	# s[best_195], tmp621
	jbe	.L30	#,
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movl	$3, %edx	#, best
	movl	$3, %eax	#, best
.L30:
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	comiss	752(%rsp,%rax,4), %xmm0	# s[best_201], tmp624
	jbe	.L31	#,
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movl	$4, %edx	#, best
	movl	$4, %eax	#, best
.L31:
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movaps	%xmm0, %xmm1	# s1, tmp627
	shufps	$85, %xmm0, %xmm1	#, s1, tmp627
	comiss	752(%rsp,%rax,4), %xmm1	# s[best_207], tmp626
	jbe	.L32	#,
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movl	$5, %edx	#, best
	movl	$5, %eax	#, best
.L32:
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movaps	%xmm0, %xmm1	# s1, tmp630
	unpckhps	%xmm0, %xmm1	# s1, tmp630
	comiss	752(%rsp,%rax,4), %xmm1	# s[best_213], tmp629
	jbe	.L33	#,
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movl	$6, %edx	#, best
	movl	$6, %eax	#, best
.L33:
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	shufps	$255, %xmm0, %xmm0	#, s1, tmp633
	comiss	752(%rsp,%rax,4), %xmm0	# s[best_219], tmp632
	jbe	.L34	#,
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movl	$7, %edx	#, best
	movl	$7, %eax	#, best
.L34:
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	comiss	752(%rsp,%rax,4), %xmm2	# s[best_225], tmp635
	jbe	.L35	#,
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	movl	$8, %edx	#, best
	movl	$8, %eax	#, best
.L35:
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	shufps	$85, %xmm2, %xmm2	#, s2, tmp638
# ../../loop/candidates/mlp16/infer.c:87:       if (s[c] > s[best]) best = c;
	ucomiss	752(%rsp,%rax,4), %xmm2	# s[best_231], tmp637
	movl	$9, %eax	#, tmp715
	cmova	%eax, %edx	# best,, tmp715, best
# ../../loop/candidates/mlp16/infer.c:88:     out[i] = (char)('0' + best);
	addl	$48, %edx	#, tmp639
	movb	%dl, 0(%rbp,%r15)	# tmp639, MEM[(char *)out_111 + _629 * 1]
# ../../loop/candidates/mlp16/infer.c:48:   for (long i = 0; i < n; i++) {
	addq	$1, %r15	#, i
	jmp	.L11	#
.L39:
# ../../loop/candidates/mlp16/infer.c:70:     v4f s0 = B2[0], s1 = B2[1], s2 = B2[2];
	movaps	%xmm9, %xmm2	# _48, s2
# ../../loop/candidates/mlp16/infer.c:70:     v4f s0 = B2[0], s1 = B2[1], s2 = B2[2];
	movaps	%xmm10, %xmm0	# _47, s1
# ../../loop/candidates/mlp16/infer.c:70:     v4f s0 = B2[0], s1 = B2[1], s2 = B2[2];
	movaps	%xmm11, %xmm1	# _46, s0
	jmp	.L13	#
.L62:
# ../../loop/candidates/mlp16/infer.c:90:   out[n] = '\n';
	movb	$10, 0(%rbp,%r15)	#, *_68
# ../../loop/candidates/mlp16/infer.c:91:   fwrite(out, 1, (size_t)n + 1, stdout);
	movq	stdout(%rip), %rcx	# stdout,
	leaq	1(%r15), %rdx	#, tmp641
	movq	%rbp, %rdi	# out,
	movl	$1, %esi	#,
	call	fwrite@PLT	#
# ../../loop/candidates/mlp16/infer.c:92:   return 0;
	xorl	%eax, %eax	# <retval>
	jmp	.L1	#
.L61:
# ../../loop/candidates/mlp16/infer.c:93: }
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE29:
	.size	main, .-main
	.local	B2
	.comm	B2,48,32
	.local	W2
	.comm	W2,768,32
	.local	B1
	.comm	B1,32,32
	.local	T
	.comm	T,34816,32
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC1:
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.byte	16
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
