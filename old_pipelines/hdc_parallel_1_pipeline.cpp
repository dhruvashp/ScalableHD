#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace std;

using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXfColMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

// Hyperparams
int N = 512, F = 512, D = 10000;
int n = 32, f = 32, d = 32;
int R = 2;
int T = 64;

// Global matrices
MatrixXfRowMajor X;
MatrixXfColMajor B;
MatrixXfRowMajor H_parallel;
MatrixXfRowMajor H_naive;


// Hardsign function
template <typename Derived>
void apply_hardsign(Eigen::MatrixBase<Derived>& mat) {
    for (int i = 0; i < mat.rows(); ++i)
        for (int j = 0; j < mat.cols(); ++j)
            mat(i, j) = mat(i, j) >= 0.0f ? 1.0f : -1.0f;
}

// Print matrix helper
template <typename Derived>
void print_matrix(const Eigen::MatrixBase<Derived>& mat, const std::string& name = "Matrix") {
    std::cout << name << " (" << mat.rows() << "x" << mat.cols() << "):\n";
    for (int i = 0; i < mat.rows(); ++i) {
        std::cout << "[ ";
        for (int j = 0; j < mat.cols(); ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << mat(i, j) << " ";
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

// Hardsign output mismatch helper
int count_hardsign_matches(const MatrixXfRowMajor& A, const MatrixXfRowMajor& B) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());

    int match_count = 0;
    int total = A.rows() * A.cols();

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            float sign_a = A(i, j) >= 0.0f ? 1.0f : -1.0f;
            float sign_b = B(i, j) >= 0.0f ? 1.0f : -1.0f;
            if (sign_a == sign_b)
                ++match_count;
        }
    }

    double match_percent = 100.0 * match_count / total;
    std::cout << "HardSign match: " << match_count << " / " << total 
              << " = " << std::fixed << std::setprecision(2) 
              << match_percent << "%\n";

    return match_count;
}

// Generic mismatch helper
void print_mismatch(const MatrixXfRowMajor& A, const MatrixXfRowMajor& B, float tol = 1e-4f) {
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            float diff = std::abs(A(i, j) - B(i, j));
            if (diff > tol) {
                std::cout << "Mismatch at (" << i << "," << j << "): A = "
                          << A(i, j) << ", B = " << B(i, j)
                          << ", diff = " << diff << "\n";
            }
        }
    }
}

// Thread worker
void task1_worker(int thread_id) {
    
    for (int col_start = thread_id * d; col_start < D; col_start += T * d) {
        
        int col_end = min(col_start + d, D);        // col_end excluded
        int local_d = col_end - col_start;          // working on length d

        MatrixXfRowMajor h_column_block = MatrixXfRowMajor::Zero(N, local_d);

        for (int r_block_offset = 0; r_block_offset < F; r_block_offset += R * f) {
            
            int r_end = min(r_block_offset + R * f, F);     // r_end is excluded            

            for (int row_start = 0; row_start < N; row_start += n) {
                
                int row_end = min(row_start + n, N);        // row_end is excluded
                int local_n = row_end - row_start;

                MatrixXfRowMajor h_block_local = MatrixXfRowMajor::Zero(local_n, local_d);

                for (int block_internal_start = r_block_offset; block_internal_start < r_end; block_internal_start += f) {

                    int block_internal_end = min(block_internal_start + f, r_end);
                    int local_f = block_internal_end - block_internal_start;

                    auto x_block_local = X.block(row_start, block_internal_start, local_n, local_f);
                    auto b_block_local = B.block(block_internal_start, col_start, local_f, local_d);

                    h_block_local.noalias() += x_block_local * b_block_local;
            
                }

                h_column_block.block(row_start, 0, local_n, local_d) += h_block_local;
                
            }

        }
        
        apply_hardsign(h_column_block); 
        H_parallel.block(0, col_start, N, local_d) = h_column_block;          
    
    }
}

// Correctness check
bool check_correctness(const MatrixXfRowMajor& A, const MatrixXfRowMajor& B, float tol = 1e-4f) {
    return ((A - B).cwiseAbs().maxCoeff() < tol);
}

int main() {
    
    // Init
    X = MatrixXfRowMajor::Random(N, F);
    B = MatrixXfColMajor::Random(F, D);
    H_parallel = MatrixXfRowMajor::Zero(N, D);
    H_naive = MatrixXfRowMajor::Zero(N, D);

    // Naive
    auto t1 = chrono::high_resolution_clock::now();
    H_naive.noalias() = X * B;
    apply_hardsign(H_naive); // DBG
    auto t2 = chrono::high_resolution_clock::now();
    double time_naive = chrono::duration<double, milli>(t2 - t1).count();

    // Parallel
    auto t3 = chrono::high_resolution_clock::now();
    vector<thread> threads;
    for (int t = 0; t < T; ++t)
        threads.emplace_back(task1_worker, t);
    for (auto& th : threads) th.join();
    auto t4 = chrono::high_resolution_clock::now();
    double time_parallel = chrono::duration<double, milli>(t4 - t3).count();

    // Timing
    cout << fixed << setprecision(3);
    cout << "Naive time   : " << time_naive    << " ms\n";
    cout << "Parallel time : " << time_parallel << " ms\n";

    return 0;
}
