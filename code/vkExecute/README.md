# vkExecute

This is a pybind11 based Python extension for ShaderPerFormer.

After installation, you should have an Python extension that can be imported with `import vkExecute`.

Detailed documentations are not available, but you can look at `src/PythonMain.cpp` to see what's provided.

A brief note on what are implemented:

- `vkExecute.ImageOnlyExecutor`: Implements a pipeline capable of replay ImageOnly Shadertoy shaders with Vulkan. Features include:
  - GPU Timestamp based timing solutions
  - Uniform input block editing
  - RenderDoc capture integration
  - SPIR-V trace buffer (a piece of storage buffer, actually) binding and readout
  - Vulkan vendor driver description readout
  - Final render target readout
- `vkExecute.ShaderProcessor`: Provides interface for calling glslang.
  - `compileShaderToSPIRV_Vulkan` should be the most useful
- `vkExecute.SpvProcessor`: Provides interface for calling SPIRV-Tools
  - `loadSpv`: load SPIR-V blob
    - expects a SPIR-V blob encoded as `std::vector<char>` (equiv. to `list[str]` under Python side)
  - `runPassSequence`: call `spvtools::opt` and run optimization sequences (e.g. `-Os`)
  - `validate`: call SPIR-V validator to ensure the currently transformed SPIR-V is valid
  - `assemble` and `disassemble`: converts from and to SPIR-V text form.
    - Note that this may rearrange ID allocation for some of the SPIR-V identifier
  - `exhaustiveInlining`: aggressive inlining, converting program to a single entrypoint one (if applicable)
  - `instrumentBasicBlockTrace`: instrument atomic increment to a dedicated counter in a slot inside trace buffer for each reachable basic block in the shader program
    - **SEE ALSO: the separate section illustrating this in paper**
- `vkExecute.editDistance`: C++ version of calculating edit distance between sequences, a cheap way of determining how far a transformed program exists from the original program 
- `vkExecute.spv.Tokenizer`: SPIR-V blob tokenizer
  - `tokenize`: tokenize without trace
    - Useful for ablation study (to train model without trace)
  - `tokenizeWithTrace`: tokenize with trace
  - `deTokenize`: useful for debugging

## Testing

Note that the `CMakeLists.txt` can also be called standalone (instead of being called by setup.py, which in turn calls by pip for installing the package itself).

Therefore a suite of standalone tests are implemented using `Catch2` framework.

See tests under `test` for more info.

## How to install (from source)

### Environment variables

`CMAKE_BUILD_PARALLEL_LEVEL`: (Choose from positive integer) How many process can be used in compilation
`DEBUG`: (Choose from `1` or `0`) Whether to build a Debug build

### Dependencies

Summary:
- Windows: `Visual Studio 2019+` with C++ Compiler & CMake
- Linux: Take CentOS 7 as an example 
  ```
  yum install gcc gcc-c++ make -y
  yum install vulkan vulkan-devel -y
  yum install libXi libXi-devel libXinerama libXinerama-devel libXcursor libXcursor-devel libXrandr libXrandr-devel -y
  yum install git -y
  ```

### Building 

For Windows, the following should build and install the package from Visual Studio PowerShell:

```powershell
# This is an example
$env:DEBUG='1'; $env:CMAKE_BUILD_PARALLEL_LEVEL=12;
pip install --verbose .
```

For Linux, 

```bash
# This is an example
DEBUG=1 CMAKE_BUILD_PARALLEL_LEVEL=12 pip install --verbose .
```

### CI

`cibuildwheel --platform linux` may also used and is tested to be working.

This produces a `.whl` capable of running in Python 3.8 on x86-64 Linux with `glibc>=2.28`.
