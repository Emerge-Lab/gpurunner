# gpurunner
## Prebuild requirements
If you did not clone recursively, run

`git submodule update --init --recursive`

## Build
In the root directory of your `gpurunner` clone:
`mkdir build`
`cd build`
`cmake ..`
`make -j $CORE_COUNT`
## Run

