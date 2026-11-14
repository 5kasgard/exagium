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
.LFB22:
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
	subq	$168, %rsp	#,
	.cfi_def_cfa_offset 224
# ../../loop/baselines/linear/infer.c:7: int main(int argc, char **argv) {
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp201
	movq	%rax, 152(%rsp)	# tmp201, D.3005
	xorl	%eax, %eax	# tmp201
# ../../loop/baselines/linear/infer.c:8:   if (argc != 4) return 2;
	cmpl	$4, %edi	#, tmp192
	je	.L32	#,
.L3:
# ../../loop/baselines/linear/infer.c:8:   if (argc != 4) return 2;
	movl	$2, %eax	#, <retval>
.L1:
# ../../loop/baselines/linear/infer.c:33: }
	movq	152(%rsp), %rdx	# D.3005, tmp202
	subq	%fs:40, %rdx	# MEM[(<address-space-1> long unsigned int *)40B], tmp202
	jne	.L33	#,
	addq	$168, %rsp	#,
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
.L32:
	.cfi_restore_state
# ../../loop/baselines/linear/infer.c:9:   FILE *f = fopen(argv[1], "rb");
	movq	8(%rsi), %rdi	# MEM[(char * *)argv_45(D) + 8B], MEM[(char * *)argv_45(D) + 8B]
	leaq	.LC0(%rip), %r14	#, tmp183
	movq	%rsi, %r12	# tmp193, argv
	movq	%r14, %rsi	# tmp183,
	call	fopen@PLT	#
	movq	%rax, %rbp	# tmp194, tmp146
# ../../loop/baselines/linear/infer.c:10:   if (!f || fread(W, sizeof W, 1, f) != 1 || fread(b, sizeof b, 1, f) != 1) return 2;
	testq	%rax, %rax	# tmp146
	je	.L3	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	leaq	W(%rip), %rbx	#, ivtmp.81
	movq	%rax, %rcx	# tmp146,
	movl	$1, %edx	#,
	movl	$2560, %esi	#,
	movq	%rbx, %rdi	# ivtmp.81,
	call	fread@PLT	#
# ../../loop/baselines/linear/infer.c:10:   if (!f || fread(W, sizeof W, 1, f) != 1 || fread(b, sizeof b, 1, f) != 1) return 2;
	subq	$1, %rax	#, tmp195
	jne	.L3	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	leaq	b(%rip), %r13	#, tmp185
	movq	%rbp, %rcx	# tmp146,
	movl	$1, %edx	#,
	movl	$40, %esi	#,
	movq	%r13, %rdi	# tmp185,
	call	fread@PLT	#
# ../../loop/baselines/linear/infer.c:10:   if (!f || fread(W, sizeof W, 1, f) != 1 || fread(b, sizeof b, 1, f) != 1) return 2;
	subq	$1, %rax	#, tmp196
	jne	.L3	#,
# ../../loop/baselines/linear/infer.c:11:   fclose(f);
	movq	%rbp, %rdi	# tmp146,
	leaq	V(%rip), %rbp	#, ivtmp.82
	call	fclose@PLT	#
	movss	.LC2(%rip), %xmm2	#, tmp188
	movq	%rbx, %rsi	# ivtmp.81, ivtmp.81
	movq	%rbp, %rdi	# ivtmp.82, ivtmp.82
	leaq	256(%rbx), %rcx	#, ivtmp.83
	leaq	2560(%rcx), %r8	#, _176
	shufps	$0, %xmm2, %xmm2	# tmp188
.L5:
# ../../loop/baselines/linear/infer.c:7: int main(int argc, char **argv) {
	movq	%rdi, %rax	# ivtmp.82, ivtmp.71
	movq	%rsi, %rdx	# ivtmp.81, ivtmp.68
	.p2align 4,,10
	.p2align 3
.L6:
# ../../loop/baselines/linear/infer.c:12:   for (int c = 0; c < 10; c++) for (int p = 0; p < 64; p++) V[p][c] = W[c][p] / 16.0f;
	movaps	(%rdx), %xmm0	# MEM <vector(4) float> [(float *)_158], vect__3.29
	addq	$16, %rdx	#, ivtmp.68
	addq	$160, %rax	#, ivtmp.71
	mulps	%xmm2, %xmm0	# tmp188, vect__3.29
# ../../loop/baselines/linear/infer.c:12:   for (int c = 0; c < 10; c++) for (int p = 0; p < 64; p++) V[p][c] = W[c][p] / 16.0f;
	movaps	%xmm0, %xmm1	# vect__3.29, tmp153
	movss	%xmm0, -160(%rax)	# vect__3.29, MEM[(float *)_159]
	shufps	$85, %xmm0, %xmm1	#, vect__3.29, tmp153
	movss	%xmm1, -120(%rax)	# tmp153, MEM[(float *)_159 + 40B]
	movaps	%xmm0, %xmm1	# vect__3.29, tmp155
	unpckhps	%xmm0, %xmm1	# vect__3.29, tmp155
	shufps	$255, %xmm0, %xmm0	#, vect__3.29, tmp157
	movss	%xmm0, -40(%rax)	# tmp157, MEM[(float *)_159 + 120B]
	movss	%xmm1, -80(%rax)	# tmp155, MEM[(float *)_159 + 80B]
	cmpq	%rcx, %rdx	# ivtmp.83, ivtmp.68
	jne	.L6	#,
# ../../loop/baselines/linear/infer.c:12:   for (int c = 0; c < 10; c++) for (int p = 0; p < 64; p++) V[p][c] = W[c][p] / 16.0f;
	leaq	256(%rdx), %rcx	#, ivtmp.83
	addq	$256, %rsi	#, ivtmp.81
	addq	$4, %rdi	#, ivtmp.82
	cmpq	%r8, %rcx	# _176, ivtmp.83
	jne	.L5	#,
# /usr/include/stdlib.h:488:   return strtol (__nptr, (char **) NULL, 10);
	movq	24(%r12), %rdi	# MEM[(char * *)argv_45(D) + 24B], MEM[(char * *)argv_45(D) + 24B]
	movl	$10, %edx	#,
	xorl	%esi, %esi	#
	call	strtol@PLT	#
# ../../loop/baselines/linear/infer.c:14:   FILE *x = fopen(argv[2], "rb");
	movq	16(%r12), %rdi	# MEM[(char * *)argv_45(D) + 16B], MEM[(char * *)argv_45(D) + 16B]
	movq	%r14, %rsi	# tmp183,
# /usr/include/stdlib.h:488:   return strtol (__nptr, (char **) NULL, 10);
	movq	%rax, %rbx	# tmp197, _73
# ../../loop/baselines/linear/infer.c:14:   FILE *x = fopen(argv[2], "rb");
	call	fopen@PLT	#
	movq	%rax, %r12	# tmp198, x
# ../../loop/baselines/linear/infer.c:15:   if (!x) return 2;
	testq	%rax, %rax	# x
	je	.L3	#,
# ../../loop/baselines/linear/infer.c:16:   char *out = malloc(n + 2);
	leaq	2(%rbx), %rdi	#, tmp162
# ../../loop/baselines/linear/infer.c:16:   char *out = malloc(n + 2);
	call	malloc@PLT	#
# ../../loop/baselines/linear/infer.c:18:   for (long i = 0; i < n; i++) {
	testq	%rbx, %rbx	# _73
	jle	.L8	#,
	leaq	(%rbx,%rax), %r10	#, _151
	leaq	80(%rsp), %rdi	#, tmp186
	movq	%r12, %r9	# x, x
	movq	%rbx, %r8	# _73, _73
	movq	%rax, %r15	# out, ivtmp.65
	leaq	2600(%rbp), %r14	#, _128
	movq	%rax, %r11	# out, out
	movq	%r10, %r12	# _151, _151
	movq	%rdi, %rbx	# tmp186, tmp186
.L16:
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	movq	%r9, %rcx	# x,
	movl	$1, %edx	#,
	movl	$64, %esi	#,
	movq	%rbx, %rdi	# tmp186,
	movq	%r8, 24(%rsp)	# _73, %sfp
	movq	%r11, 16(%rsp)	# out, %sfp
	movq	%r9, 8(%rsp)	# x, %sfp
	call	fread@PLT	#
# ../../loop/baselines/linear/infer.c:19:     if (fread(px, 64, 1, x) != 1) return 2;
	movq	8(%rsp), %r9	# %sfp, x
	movq	16(%rsp), %r11	# %sfp, out
	cmpq	$1, %rax	#, ivtmp.38
	movq	24(%rsp), %r8	# %sfp, _73
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	movq	%rax, %rdx	# tmp200, ivtmp.38
# ../../loop/baselines/linear/infer.c:19:     if (fread(px, 64, 1, x) != 1) return 2;
	jne	.L3	#,
# ../../loop/baselines/linear/infer.c:21:     for (int c = 0; c < 10; c++) s[c] = b[c];
	movdqa	0(%r13), %xmm3	# MEM <unsigned char[40]> [(char * {ref-all})&b], tmp255
	movq	%rbx, %rdi	# tmp186, ivtmp.58
	leaq	40(%rbp), %rcx	#, ivtmp.60
	movq	%rbp, %rsi	# ivtmp.82, ivtmp.59
	movq	32+b(%rip), %rax	# MEM <unsigned char[40]> [(char * {ref-all})&b], MEM <unsigned char[40]> [(char * {ref-all})&b]
	movq	%rdx, 8(%rsp)	# ivtmp.38, %sfp
	leaq	32(%rsp), %r10	#, tmp184
	movdqa	16+b(%rip), %xmm4	# MEM <unsigned char[40]> [(char * {ref-all})&b], tmp256
	movaps	%xmm3, 32(%rsp)	# tmp255, MEM <unsigned char[40]> [(char * {ref-all})&s]
	movq	%rax, 64(%rsp)	# MEM <unsigned char[40]> [(char * {ref-all})&b], MEM <unsigned char[40]> [(char * {ref-all})&s]
	movaps	%xmm4, 48(%rsp)	# tmp256, MEM <unsigned char[40]> [(char * {ref-all})&s]
	.p2align 4,,10
	.p2align 3
.L12:
# ../../loop/baselines/linear/infer.c:23:       float v = (float)px[p];
	movzbl	(%rdi), %eax	# MEM[(unsigned char *)_84], MEM[(unsigned char *)_84]
	pxor	%xmm2, %xmm2	# v
	movq	%r10, %rdx	# tmp184, ivtmp.48
	cvtsi2ssl	%eax, %xmm2	# MEM[(unsigned char *)_84], v
	movq	%rsi, %rax	# ivtmp.59, ivtmp.49
	shufps	$0xe0, %xmm2, %xmm2	# vect_cst__83
.L11:
# ../../loop/baselines/linear/infer.c:24:       for (int c = 0; c < 10; c++) s[c] += V[p][c] * v;
	movq	(%rax), %xmm0	# MEM <vector(2) float> [(float *)_12], MEM <vector(2) float> [(float *)_12]
# ../../loop/baselines/linear/infer.c:24:       for (int c = 0; c < 10; c++) s[c] += V[p][c] * v;
	movq	(%rdx), %xmm1	# MEM <vector(2) float> [(float *)_10], MEM <vector(2) float> [(float *)_10]
	addq	$8, %rax	#, ivtmp.49
	addq	$8, %rdx	#, ivtmp.48
# ../../loop/baselines/linear/infer.c:24:       for (int c = 0; c < 10; c++) s[c] += V[p][c] * v;
	mulps	%xmm2, %xmm0	# vect_cst__83, vect__12.22
# ../../loop/baselines/linear/infer.c:24:       for (int c = 0; c < 10; c++) s[c] += V[p][c] * v;
	addps	%xmm0, %xmm1	# vect__12.22, vect__13.23
	movlps	%xmm1, -8(%rdx)	# vect__13.23, MEM <vector(2) float> [(float *)_10]
	cmpq	%rax, %rcx	# ivtmp.49, ivtmp.60
	jne	.L11	#,
# ../../loop/baselines/linear/infer.c:22:     for (int p = 0; p < 64; p++) {
	addq	$40, %rcx	#, ivtmp.60
	addq	$1, %rdi	#, ivtmp.58
	addq	$40, %rsi	#, ivtmp.59
	cmpq	%r14, %rcx	# _128, ivtmp.60
	jne	.L12	#,
# ../../loop/baselines/linear/infer.c:26:     int best = 0;
	movq	8(%rsp), %rdx	# %sfp, ivtmp.38
	xorl	%eax, %eax	# c
	.p2align 4,,10
	.p2align 3
.L15:
# ../../loop/baselines/linear/infer.c:27:     for (int c = 1; c < 10; c++) if (s[c] > s[best]) best = c;
	movslq	%eax, %rcx	# c, c
# ../../loop/baselines/linear/infer.c:27:     for (int c = 1; c < 10; c++) if (s[c] > s[best]) best = c;
	movss	(%r10,%rdx,4), %xmm0	# MEM[(float *)&s + ivtmp.38_38 * 4], MEM[(float *)&s + ivtmp.38_38 * 4]
	ucomiss	32(%rsp,%rcx,4), %xmm0	# s[best_105], MEM[(float *)&s + ivtmp.38_38 * 4]
	cmova	%edx, %eax	# c,, ivtmp.38, c
# ../../loop/baselines/linear/infer.c:27:     for (int c = 1; c < 10; c++) if (s[c] > s[best]) best = c;
	addq	$1, %rdx	#, ivtmp.38
	cmpq	$10, %rdx	#, ivtmp.38
	jne	.L15	#,
# ../../loop/baselines/linear/infer.c:28:     out[i] = (char)('0' + best);
	addl	$48, %eax	#, tmp178
# ../../loop/baselines/linear/infer.c:18:   for (long i = 0; i < n; i++) {
	addq	$1, %r15	#, ivtmp.65
# ../../loop/baselines/linear/infer.c:28:     out[i] = (char)('0' + best);
	movb	%al, -1(%r15)	# tmp178, MEM[(char *)_148]
# ../../loop/baselines/linear/infer.c:18:   for (long i = 0; i < n; i++) {
	cmpq	%r12, %r15	# _151, ivtmp.65
	jne	.L16	#,
	movq	%r11, %rax	# out, out
	movq	%r8, %rbx	# _73, _73
.L8:
# ../../loop/baselines/linear/infer.c:30:   out[n] = '\n';
	movb	$10, (%rax,%rbx)	#, *_22
# ../../loop/baselines/linear/infer.c:31:   fwrite(out, 1, n + 1, stdout);
	leaq	1(%rbx), %rdx	#, tmp180
# ../../loop/baselines/linear/infer.c:31:   fwrite(out, 1, n + 1, stdout);
	movl	$1, %esi	#,
	movq	%rax, %rdi	# out,
	movq	stdout(%rip), %rcx	# stdout,
	call	fwrite@PLT	#
# ../../loop/baselines/linear/infer.c:32:   return 0;
	xorl	%eax, %eax	# <retval>
	jmp	.L1	#
.L33:
# ../../loop/baselines/linear/infer.c:33: }
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE22:
	.size	main, .-main
	.local	b
	.comm	b,40,32
	.local	V
	.comm	V,2560,32
	.local	W
	.comm	W,2560,32
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC2:
	.long	1031798784
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
