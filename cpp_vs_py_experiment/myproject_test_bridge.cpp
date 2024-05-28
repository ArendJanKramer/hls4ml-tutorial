#include "myproject_bridge.h"
#include "firmware/myproject.h"
#include <sys/time.h>

template <typename T, typename A>
int arg_max(std::vector<T, A> const &vec) {
    return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

template <typename T>
void predict(const char *x_file,
             const char *y_file,
             bool compute_accuracy) {
    T fc1_input[N_INPUT_1_1] = {0};
    T layer13_out_gt[N_LAYER_11] = {0};
    T layer13_out_pred[N_LAYER_11] = {0};

    FILE *input_fp;
    input_fp = fopen(x_file, "r");

    FILE *output_fp;
    output_fp = fopen(y_file, "r");

    std::vector<int> all_preds(1000, 0);
    std::vector<int> all_gts(1000, 0);

    for (int i = 0; i < 1000; i++) {
        fseek(input_fp, N_INPUT_1_1 * sizeof(T) * i, SEEK_SET);
        fseek(output_fp, N_LAYER_11 * sizeof(T) * i, SEEK_SET);

        fread(fc1_input, sizeof(T),N_INPUT_1_1, input_fp);
        fread(layer13_out_gt, sizeof(T),N_LAYER_11, output_fp);

        if constexpr (std::is_same<T, double>::value) {
            myproject_double(fc1_input, layer13_out_pred);
        } else if constexpr (std::is_same<T, float>::value) {
            myproject_float(fc1_input, layer13_out_pred);
        } else {
            throw std::invalid_argument("Only float and double data types supported");
        }

        std::vector gt(std::begin(layer13_out_gt), std::end(layer13_out_gt));
        std::vector pred(std::begin(layer13_out_pred), std::end(layer13_out_pred));
        int class_gt = arg_max(gt);
        int class_pred = arg_max(pred);

        all_gts[i] = class_gt;
        all_preds[i] = class_pred;

    }

    fclose(input_fp);
    fclose(output_fp);

    if (compute_accuracy) {
        int n_corr = 0;
        for (int i = 0; i < 1000; i++) {
            if (all_gts[i] == all_preds[i])
                n_corr++;
        }
        float acc = (float)n_corr / (float)1000;

        printf("Accuracy %f  \n", acc);
        fflush(stdout);
    }

}

long long time() {
    timeval tp;
    gettimeofday(&tp, NULL);
    long long mslong = (long long)tp.tv_sec * 1000L + tp.tv_usec / 1000; //get current timestamp in milliseconds
    return mslong;
}

int main(int argc, char **argv) {
    auto x_double = "hlstest/input_double_1k_py.dat";
    auto y_double = "hlstest/output_double_gt_py.dat";

    for (int i = 0; i < 100; i++) {
        auto t1 = time();
        predict<double>(x_double, y_double, false);
        auto t2 = time();
        printf("Iter %i took %lld ms\n", i, t2 - t1);
        fflush(stdout); // For debugger shell
    }

    return 0;
}
