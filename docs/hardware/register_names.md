# Register Names - Backwards Compatibility
## 1. Intel's First Processor - 8008
[register_first](../images/register1.png)
* 8 bit registers

## 2. Intel's Base Processor Structure - 8086
[register_8086](../images/register2.png)
* 16bit registers
* H and L for **backward compatibility**


## Intel 80386
[register_80386](../images/register3.png)
* "Extended" prefix for 32bit
* Can still use 8086 register names for backward compatibility
* After that, naming was being terrible so Intel went to make a whole new Titanium chip - didn't work well
* AMD takes over

## AMD x86-64
[amd x86-64](../images/register4.png)
* "Register" prefix
* 9 -> 17 registers
* for the new registers just give a number

[word size](../images/register5.png)

* For backward compatibility with 8086: **word is 16 bytes**
* x86-64 words are 64 bytes, and we call them **quad-words**


## Assembly
* Intel syntax: destination on the left

[weird_compiler_behavor](../images/lea.png)
   * adding 32 bits and saving to 64 bit compiler?
   * adding and pointing as if rdx and rcx is a base memory pointer?

* weird in "human conception"
* optimized compiler code is **always not pretty, but fast and works**
* calling convention for functions - different for every architecture, each register has a unique role

[assembly_function_convention](../images/assembly_function.png)