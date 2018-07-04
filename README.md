# Procedural-Noise
Implementation of an assortment of noise functions in *C++11*. Bundled is an noise explorer for real time tweaking and noise visualization.

# Goals
* Provide a standard cross-platform (Linux, macOS) implementation of a bunch of noise functions/algorithms together with references to the sources and/or authors.
* Correct implementation rather than speed.
* Readable code rather than speed.
* Variants of different algorithms in order to promote comparisons between algorithms and different implementation techniques.

The end objective for this library is to be the drop-in noise generation source for a lot of projects. Those who need the fastest performance will find it easy enough to optimize and those who want to experiment and learn should find the source code readable and inviting.

# Dependencies
## Noise header
* C++ standard library
* GLM 
## Noise explorer program
* _(all of the above)_
* SDL2
* OpenGL
* ImGUI

# License
The MIT License (MIT)
Copyright (c) 2017 Alexander Lingtorp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
