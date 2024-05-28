# Python vss C++ Predict test
This experiment was made to compare differences in speed between C++ and Python.
An increasing inference time has been observed in fpga_emu target.

The C++ and Python codes are not identical.
The accuracy computation is done in Python once, whereas in C++ it's done every predict call.
However, it should still show the same trent.

## Run tutorial 1
Run `part1_getting_started.ipynb` notebook.
This should produce the following files:
```text
X_train_val.npy
X_test.npy
y_train_val.npy
y_test.npy
classes.npy
```

Afterward, a folder called `model_1` should exist.

## Run Python sandbox
Adjust `python_sandbox.py` to use the correct file paths.
Let it produce `.dat` files by setting:
```python
export_test_data = True
```

By letting the loop run 1 iteration, an accuracy score should appear.

## Beware
Before making changes to the C++ files, either make a copy 
or be aware that the project folder will be lost after model.compile() is called in hls4ml.

In `hls4ml/model/graph.py`, go to `ModelGraph.compile()`:
```python
def compile(self):
    """Compile the generated project and link the library into current environment.

    Users should call this function if they want to use `predict` functionality for simulation.
    """
    # self.write()
    self._compile()
```
Comment out self.write() to protect against overwriting some files from template.

## Make C++ build
Copy `my_project_bridge.h` and `my_project_test_bridge.cpp` to:
`model_1/hls4ml_prj/src`

Adjust `model_1/hls4ml_prj/CMakeLists.txt` to use the new sources instead.

```text
set(SOURCE_FILES src/firmware/myproject.cpp src/myproject_bridge.h src/myproject_test_bridge.cpp src/myproject_bridge.cpp)
```

Adjust `myproject_test_bridge.cpp` accordingly to match file paths.

Now run:
```bash
cd build
cmake ..
make fpga_emu
```

And run executable. 
