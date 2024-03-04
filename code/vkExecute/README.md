# vkExecute

- Linux
  - libxrandr libxinerama libxcursor libxi (used by glfw)

```
$env:DEBUG='1'; $env:CMAKE_BUILD_PARALLEL_LEVEL=12;

pip install --verbose .

# cibuildwheel --platform linux

# python setup.py bdist_wheel
```

