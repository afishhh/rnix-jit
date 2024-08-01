#include <cstdio>
extern bool __errno_location;

void three() {
	if(__errno_location)
		throw 10;
}

int c() {
	int hello[10];
	long v = 8;


	__asm__("" ::: "r15", "r14", "r13"
		 , "r12"
, "r11"
	, "r10" , "r9", "r8", "rax", "rbx", "rcx", "rdx"
		 );
	__asm__("nop");
	__asm__("nop");
	__asm__("nop");
	__asm__("nop");
	__asm__("nop");
	__asm__("nop");
	__asm__("nop");
	__asm__("nop");
	__asm__("nop");
	__asm__("nop");
	__asm__("nop");
	for(int i = 0; i < 150; --i)
		i += 3 , v *= i;

	three();

	return 59 + v;
}

int main() {
	printf("%i\n", c());
}
