#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void mul(const float* left, const float* right, float* target, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            float sum = 0.0;
            for (size_t index = 0; index < n; ++index) {
                sum += left[i * n + index] * right[index * k + j];
            }
            target[i * k + j] = sum;
        }
    }
}
void normalize(float* target, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        float sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            sum += std::exp(target[i * n + j]);
        }
        for (size_t j = 0; j < n; ++j) {
            target[i * n + j] = std::exp(target[i * n + j]) / sum;
        }
    }
}
void one_hot(const unsigned char* y, float* target, size_t m, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            target[i * k + j] = 0.0;
        }
        target[i * k + y[i]] = 1.0;
    }
}
void calc_single(float* target, char type, float value, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            switch (type) {
                case '+':
                    target[i * n + j] += value;
                    break;
                case '-':
                    target[i * n + j] -= value;
                    break;
                case '*':
                    target[i * n + j] *= value;
                    break;
                case '/':
                    target[i * n + j] /= value;
                    break;
            }
        }
    }
}
void calc_matrix(float* target, char type, float* value, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            switch (type) {
                case '+':
                    target[i * n + j] += value[i * n + j];
                    break;
                case '-':
                    target[i * n + j] -= value[i * n + j];;
                    break;
                case '*':
                    target[i * n + j] *= value[i * n + j];;
                    break;
                case '/':
                    target[i * n + j] /= value[i * n + j];;
                    break;
            }
        }
    }
}
// m, n -> n, m
void transpose(const float* source, float* target, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            target[j * m + i] = source[i * n + j];
        }
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float* z_batch = new float[batch * k];
    float* Iy = new float[batch * k];
    float* x_t = new float[batch * n];
    float* grad = new float[n * k];

    for (size_t bi = 0; bi < m; bi += batch) {
        size_t cur_batch = m - bi < batch ? m - bi : batch;

        const float* cur_x = X + bi * n;
        const unsigned char* cur_y = y + bi;

        // (batch_size, n) * (n, k) = (batch_size, k)
        mul(cur_x, theta, z_batch, cur_batch, n, k);
        normalize(z_batch, cur_batch, k);

        one_hot(cur_y, Iy, cur_batch, k);

        calc_matrix(z_batch, '-', Iy, cur_batch, k);
        transpose(cur_x, x_t, cur_batch, n);
        mul(x_t, z_batch, grad, n, cur_batch, k);
        calc_single(grad, '*', (lr / batch), n, k);

        calc_matrix(theta, '-', grad, n, k);
    }

    delete[] z_batch;
    delete[] Iy;
    delete[] x_t;
    delete[] grad;

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
