import ctypes
import os
import time

import numpy as np
import numpy.ctypeslib as npc
from sklearn.metrics import accuracy_score


class MockLayer:
    def __init__(self, size: int):
        self._size = size

    def size(self):
        return self._size


class StaticPredict:
    def __init__(self, project_name: str, so_path: str, output_layer_size: int,
                 test_data_dir_path: str):
        self.project_name = project_name
        self.so_path = so_path
        self.output_layer_size = output_layer_size
        self.test_data_dir_path = test_data_dir_path

    def get_top_function(self):
        """
        Copied from hls4ml/model/graph.py and modified to load existing model.
        Change `_double` suffix of top function to `_float` to let
        .predict() consume fp32 precision instead
        """
        n_outputs = 1

        _top_function_lib = ctypes.cdll.LoadLibrary(
            self.so_path)
        top_function = getattr(_top_function_lib, self.project_name + '_double')
        ctype = ctypes.c_double
        top_function.restype = None
        top_function.argtypes = [npc.ndpointer(ctype, flags="C_CONTIGUOUS")
                                 for _ in range(
                len(self.get_output_variables()) + n_outputs)]

        return top_function, ctype

    def get_output_variables(self):
        """
        Placeholder to keep .predict() code as close as possible to original
        """
        return [MockLayer(self.output_layer_size)]  # Mock

    def predict(self, x):
        """
        Modified from `hls4ml/model/graph.py`.
        Doesn't include data validation anymore;
        Make sure X data matches model input shape.
        """
        top_function, ctype = self.get_top_function()
        n_samples = len(x)
        n_inputs = 1
        n_outputs = len(self.get_output_variables())

        output = []
        if n_samples == 1 and n_inputs == 1:
            x = [x]

        try:
            for i in range(n_samples):
                predictions = [np.zeros(yj.size(), dtype=ctype) for yj in
                               self.get_output_variables()]
                if n_inputs == 1:
                    inp = [np.asarray(x[i])]
                else:
                    inp = [np.asarray(xj[i]) for xj in x]
                argtuple = inp
                argtuple += predictions
                argtuple = tuple(argtuple)
                top_function(*argtuple)
                output.append(predictions)

            # Convert to list of numpy arrays (one for each output)
            output = [
                np.asarray(
                    [output[i_sample][i_output] for i_sample in range(n_samples)])
                for i_output in range(n_outputs)
            ]
        finally:
            pass

        if n_samples == 1 and n_outputs == 1:
            return output[0][0]
        elif n_outputs == 1:
            return output[0]
        elif n_samples == 1:
            return [output_i[0] for output_i in output]
        else:
            return output

    @staticmethod
    def export_numpy_to_double_vec(np_arr: np.ndarray, output_path: str):
        """
        Converts numpy array to binary file which can be read by the C++ counterpart.
        Casts data in 64-bit double
        """
        with open(output_path, "wb") as f:
            ca = np.ascontiguousarray(np_arr).astype(np.float64)
            ca = list(ca)
            for b in ca:
                f.write(b)

    @staticmethod
    def export_numpy_to_float_vec(np_arr: np.ndarray, output_path: str):
        """
        Converts numpy array to binary file which can be read by the C++ counterpart.
        Casts data in 32-bit float
        """
        with open(output_path, "wb") as f:
            ca = np.ascontiguousarray(np_arr).astype(np.float32)
            ca = list(ca)
            for b in ca:
                f.write(b)

    def export_test_data(self, x: np.ndarray, y: np.ndarray):
        """
        Export the X and Y numpy arrays to binary files; both in 32float and 64double.
        These files can be used by the C++ counterpart
        """

        os.makedirs(self.test_data_dir_path, exist_ok=True)
        tdd = self.test_data_dir_path
        sp.export_numpy_to_double_vec(x,
                                      os.path.join(tdd, "input_double_1k_py.dat"))
        sp.export_numpy_to_float_vec(x,
                                     os.path.join(tdd, "input_float_1k_py.dat"))

        sp.export_numpy_to_double_vec(y,
                                      os.path.join(tdd, "output_double_gt_py.dat"))

        sp.export_numpy_to_float_vec(y,
                                     os.path.join(tdd, "output_float_gt_py.dat"))


if __name__ == '__main__':
    X_test = np.load("X_test.npy")
    X_test = np.ascontiguousarray(X_test[:1000])
    y_test = np.load("y_test.npy")[:1000]
    sp = StaticPredict(project_name="myproject",
                       so_path="model_1/hls4ml_prj/build/libmyproject-74054b45.so",
                       output_layer_size=5,
                       test_data_dir_path="model_1/test_data")

    export_test_data = False
    if export_test_data:
        sp.export_test_data(X_test, y_test)

    for i in range(0, 100):
        t = time.time()
        y = sp.predict(X_test)
        print(f"Iter {i} took {(time.time() - t) * 1000:.0f} ms")

    print("hls4ml Accuracy: {}".format(
        accuracy_score(np.argmax(y_test, axis=1), np.argmax(y, axis=1))))
