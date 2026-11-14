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
	subq	$152, %rsp	#,
	.cfi_def_cfa_offset 208
# ../../loop/baselines/thermo/infer.c:6: int main(int argc, char **argv) {
	movq	%fs:40, %rax	# MEM[(<address-space-1> long unsigned int *)40B], tmp176
	movq	%rax, 136(%rsp)	# tmp176, D.2965
	xorl	%eax, %eax	# tmp176
# ../../loop/baselines/thermo/infer.c:7:   if (argc != 4) return 2;
	cmpl	$4, %edi	#, tmp167
	je	.L27	#,
.L3:
# ../../loop/baselines/thermo/infer.c:7:   if (argc != 4) return 2;
	movl	$2, %eax	#, <retval>
.L1:
# ../../loop/baselines/thermo/infer.c:32: }
	movq	136(%rsp), %rdx	# D.2965, tmp177
	subq	%fs:40, %rdx	# MEM[(<address-space-1> long unsigned int *)40B], tmp177
	jne	.L28	#,
	addq	$152, %rsp	#,
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
.L27:
	.cfi_restore_state
# ../../loop/baselines/thermo/infer.c:8:   FILE *f = fopen(argv[1], "rb");
	movq	8(%rsi), %rdi	# MEM[(char * *)argv_44(D) + 8B], MEM[(char * *)argv_44(D) + 8B]
	leaq	.LC0(%rip), %r13	#, tmp128
	movq	%rsi, %r12	# tmp168, argv
	movq	%r13, %rsi	# tmp128,
	call	fopen@PLT	#
	movq	%rax, %rbp	# tmp169, tmp129
# ../../loop/baselines/thermo/infer.c:9:   if (!f || fread(T, sizeof T, 1, f) != 1 || fread(b, sizeof b, 1, f) != 1) return 2;
	testq	%rax, %rax	# tmp129
	je	.L3	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	leaq	T(%rip), %r15	#, tmp162
	movq	%rax, %rcx	# tmp129,
	movl	$1, %edx	#,
	movl	$43520, %esi	#,
	movq	%r15, %rdi	# tmp162,
	call	fread@PLT	#
# ../../loop/baselines/thermo/infer.c:9:   if (!f || fread(T, sizeof T, 1, f) != 1 || fread(b, sizeof b, 1, f) != 1) return 2;
	subq	$1, %rax	#, tmp170
	jne	.L3	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	leaq	b(%rip), %rbx	#, tmp163
	movq	%rbp, %rcx	# tmp129,
	movl	$1, %edx	#,
	movl	$40, %esi	#,
	movq	%rbx, %rdi	# tmp163,
	call	fread@PLT	#
# ../../loop/baselines/thermo/infer.c:9:   if (!f || fread(T, sizeof T, 1, f) != 1 || fread(b, sizeof b, 1, f) != 1) return 2;
	subq	$1, %rax	#, tmp171
	jne	.L3	#,
# ../../loop/baselines/thermo/infer.c:10:   fclose(f);
	movq	%rbp, %rdi	# tmp129,
	call	fclose@PLT	#
# /usr/include/stdlib.h:488:   return strtol (__nptr, (char **) NULL, 10);
	movq	24(%r12), %rdi	# MEM[(char * *)argv_44(D) + 24B], MEM[(char * *)argv_44(D) + 24B]
	movl	$10, %edx	#,
	xorl	%esi, %esi	#
	call	strtol@PLT	#
# ../../loop/baselines/thermo/infer.c:12:   FILE *x = fopen(argv[2], "rb");
	movq	16(%r12), %rdi	# MEM[(char * *)argv_44(D) + 16B], MEM[(char * *)argv_44(D) + 16B]
	movq	%r13, %rsi	# tmp128,
# /usr/include/stdlib.h:488:   return strtol (__nptr, (char **) NULL, 10);
	movq	%rax, %rbp	# tmp172, _69
# ../../loop/baselines/thermo/infer.c:12:   FILE *x = fopen(argv[2], "rb");
	call	fopen@PLT	#
	movq	%rax, %r12	# tmp173, x
# ../../loop/baselines/thermo/infer.c:13:   if (!x) return 2;
	testq	%rax, %rax	# x
	je	.L3	#,
# ../../loop/baselines/thermo/infer.c:14:   char *out = malloc(n + 2);
	leaq	2(%rbp), %rdi	#, tmp136
# ../../loop/baselines/thermo/infer.c:16:   for (long i = 0; i < n; i++) {
	xorl	%r14d, %r14d	# i
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	leaq	64(%rsp), %r13	#, tmp166
# ../../loop/baselines/thermo/infer.c:14:   char *out = malloc(n + 2);
	call	malloc@PLT	#
	movq	%rax, 8(%rsp)	# tmp174, %sfp
.L5:
# ../../loop/baselines/thermo/infer.c:16:   for (long i = 0; i < n; i++) {
	cmpq	%rbp, %r14	# _69, i
	jge	.L29	#,
# /usr/include/x86_64-linux-gnu/bits/stdio2.h:212:     return __fread_alias (__ptr, __size, __n, __stream);
	movq	%r12, %rcx	# x,
	movl	$1, %edx	#,
	movl	$64, %esi	#,
	movq	%r13, %rdi	# tmp166,
	call	fread@PLT	#
	movq	%rax, %r10	# tmp175, ivtmp.27
# ../../loop/baselines/thermo/infer.c:17:     if (fread(px, 64, 1, x) != 1) return 2;
	cmpq	$1, %rax	#, ivtmp.27
	jne	.L3	#,
# ../../loop/baselines/thermo/infer.c:19:     for (int c = 0; c < 10; c++) s[c] = b[c];
	leaq	16(%rsp), %r9	#, tmp161
	movl	$10, %ecx	#, tmp142
	movq	%rbx, %rsi	# tmp163, tmp141
	movq	%r9, %rdi	# tmp161, tmp140
	rep movsl
	movq	%r13, %rsi	# tmp166, ivtmp.49
	.p2align 4,,10
	.p2align 3
.L8:
# ../../loop/baselines/thermo/infer.c:21:       if (px[p] > 16) return 2;
	movzbl	(%rsi), %eax	# MEM[(unsigned char *)_86],
# ../../loop/baselines/thermo/infer.c:21:       if (px[p] > 16) return 2;
	cmpb	$16, %al	#, _8
	ja	.L3	#,
	leaq	(%rax,%rax,4), %rax	#, tmp146
	movq	%r9, %rdx	# tmp161, ivtmp.37
	leaq	(%rcx,%rax,8), %rax	#, tmp148
	addq	%r15, %rax	# tmp162, ivtmp.40
	leaq	40(%rax), %rdi	#, _14
.L7:
# ../../loop/baselines/thermo/infer.c:23:       for (int c = 0; c < 10; c++) s[c] += row[c];
	movq	(%rdx), %xmm0	# MEM <vector(2) float> [(float *)_10], MEM <vector(2) float> [(float *)_10]
	movq	(%rax), %xmm1	# MEM <const vector(2) float> [(const float *)_12], MEM <const vector(2) float> [(const float *)_12]
	addq	$8, %rax	#, ivtmp.40
	addq	$8, %rdx	#, ivtmp.37
	addps	%xmm1, %xmm0	# MEM <const vector(2) float> [(const float *)_12], vect__15.19
	movlps	%xmm0, -8(%rdx)	# vect__15.19, MEM <vector(2) float> [(float *)_10]
	cmpq	%rax, %rdi	# ivtmp.40, _14
	jne	.L7	#,
# ../../loop/baselines/thermo/infer.c:20:     for (int p = 0; p < 64; p++) {
	addq	$680, %rcx	#, ivtmp.50
	addq	$1, %rsi	#, ivtmp.49
	cmpq	$43520, %rcx	#, ivtmp.50
	jne	.L8	#,
# ../../loop/baselines/thermo/infer.c:25:     int best = 0;
	xorl	%eax, %eax	# c
.L11:
# ../../loop/baselines/thermo/infer.c:26:     for (int c = 1; c < 10; c++) if (s[c] > s[best]) best = c;
	movslq	%eax, %rdx	# c, c
# ../../loop/baselines/thermo/infer.c:26:     for (int c = 1; c < 10; c++) if (s[c] > s[best]) best = c;
	movss	(%r9,%r10,4), %xmm0	# MEM[(float *)&s + ivtmp.27_90 * 4], MEM[(float *)&s + ivtmp.27_90 * 4]
	ucomiss	16(%rsp,%rdx,4), %xmm0	# s[best_87], MEM[(float *)&s + ivtmp.27_90 * 4]
	cmova	%r10d, %eax	# c,, ivtmp.27, c
# ../../loop/baselines/thermo/infer.c:26:     for (int c = 1; c < 10; c++) if (s[c] > s[best]) best = c;
	addq	$1, %r10	#, ivtmp.27
	cmpq	$10, %r10	#, ivtmp.27
	jne	.L11	#,
# ../../loop/baselines/thermo/infer.c:27:     out[i] = (char)('0' + best);
	movq	8(%rsp), %rcx	# %sfp, out
	addl	$48, %eax	#, tmp156
	movb	%al, (%rcx,%r14)	# tmp156, MEM[(char *)out_52 + _84 * 1]
# ../../loop/baselines/thermo/infer.c:16:   for (long i = 0; i < n; i++) {
	addq	$1, %r14	#, i
	jmp	.L5	#
.L29:
# ../../loop/baselines/thermo/infer.c:29:   out[n] = '\n';
	movq	8(%rsp), %rdi	# %sfp, out
# ../../loop/baselines/thermo/infer.c:30:   fwrite(out, 1, n + 1, stdout);
	movq	stdout(%rip), %rcx	# stdout,
# ../../loop/baselines/thermo/infer.c:30:   fwrite(out, 1, n + 1, stdout);
	leaq	1(%rbp), %rdx	#, tmp158
# ../../loop/baselines/thermo/infer.c:30:   fwrite(out, 1, n + 1, stdout);
	movl	$1, %esi	#,
# ../../loop/baselines/thermo/infer.c:29:   out[n] = '\n';
	movb	$10, (%rdi,%rbp)	#, *_24
# ../../loop/baselines/thermo/infer.c:30:   fwrite(out, 1, n + 1, stdout);
	call	fwrite@PLT	#
# ../../loop/baselines/thermo/infer.c:31:   return 0;
	xorl	%eax, %eax	# <retval>
	jmp	.L1	#
.L28:
# ../../loop/baselines/thermo/infer.c:32: }
	call	__stack_chk_fail@PLT	#
	.cfi_endproc
.LFE22:
	.size	main, .-main
	.local	b
	.comm	b,40,32
	.local	T
	.comm	T,43520,32
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
