	.file	"test.cc"
	.text
	.globl	_Z5threev
	.type	_Z5threev, @function
_Z5threev:
.LFB3:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movzbl	__errno_location(%rip), %eax
	testb	%al, %al
	je	.L3
	movl	$4, %edi
	call	__cxa_allocate_exception
	movl	$10, (%rax)
	movl	$0, %edx
	movl	$_ZTIi, %esi
	movq	%rax, %rdi
	call	__cxa_throw
.L3:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	_Z5threev, .-_Z5threev
	.globl	_Z1cv
	.type	_Z1cv, @function
_Z1cv:
.LFB4:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$72, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	$8, -56(%rbp)
#APP
# 19 "test.cc" 1
	nop
# 0 "" 2
# 20 "test.cc" 1
	nop
# 0 "" 2
# 21 "test.cc" 1
	nop
# 0 "" 2
# 22 "test.cc" 1
	nop
# 0 "" 2
# 23 "test.cc" 1
	nop
# 0 "" 2
# 24 "test.cc" 1
	nop
# 0 "" 2
# 25 "test.cc" 1
	nop
# 0 "" 2
# 26 "test.cc" 1
	nop
# 0 "" 2
# 27 "test.cc" 1
	nop
# 0 "" 2
# 28 "test.cc" 1
	nop
# 0 "" 2
# 29 "test.cc" 1
	nop
# 0 "" 2
#NO_APP
	movl	$0, -60(%rbp)
	jmp	.L5
.L6:
	addl	$3, -60(%rbp)
	movl	-60(%rbp), %eax
	cltq
	movq	-56(%rbp), %rdx
	imulq	%rdx, %rax
	movq	%rax, -56(%rbp)
	subl	$1, -60(%rbp)
.L5:
	cmpl	$149, -60(%rbp)
	jle	.L6
	call	_Z5threev
	movq	-56(%rbp), %rax
	addl	$59, %eax
	addq	$72, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	_Z1cv, .-_Z1cv
	.section	.rodata
.LC0:
	.string	"%i\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	call	_Z1cv
	movl	%eax, %esi
	movl	$.LC0, %edi
	movl	$0, %eax
	call	printf
	movl	$0, %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.ident	"GCC: (GNU) 14.1.0"
	.section	.note.GNU-stack,"",@progbits
