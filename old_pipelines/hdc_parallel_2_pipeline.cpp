#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "concurrentqueue.h"


using namespace std;

using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXfColMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

// Measurement trials
int trials = 100;

// Application params, fixed for an application
int F = 784, D = 10000, K = 10;       // MNIST


// Hyperparams
int N = 32;                           // Variable                                           Batch size, # inputs processed at once
int n = 16, f = 16, d = 16;              // GUIDANCE: Obvious                                  Chunk sizes, NOTE: k not defined and not used (chunk along classes) 

int R = 4;                            // GUIDANCE: R <= ceil(F/f) 

int T = 1;                            // GUIDANCE: T <= max_threads_feasible/2              max_threads_feasible -> Total logical processor threads/CPUs to twice that amount




// Global matrices
MatrixXfRowMajor X;                             // Raw features (N x F)

MatrixXfColMajor B;                             // Parameter (Base Vector, F x D)
MatrixXfColMajor J;                             // Parameter (Class Vector, Transposed, D x K)

MatrixXfRowMajor S_parallel;                    // S from parallel algorithm (N x K)
Eigen::VectorXi Y_pred_parallel;                // Prediction from parallel algorithm (N)

MatrixXfRowMajor H_naive;                       // H, naive algorithm
MatrixXfRowMajor S_naive;                       // S, naive algorithm
Eigen::VectorXi Y_pred_naive;                   // Y_pred, naive algorithm



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

// Struct for passing column blocks from Task I to Task II
struct ColumnBlockPayload {
    MatrixXfRowMajor H_block; // N x d
    int col_start;            // column location of this block
};

vector<moodycamel::ConcurrentQueue<ColumnBlockPayload>> task2_queues(T);
vector<atomic<bool>> task1_done(T);

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
        task2_queues[thread_id].enqueue(ColumnBlockPayload{h_column_block, col_start});      
    
    }

    task1_done[thread_id] = true;

}

// Task II thread
void task2_worker(int thread_id) {
    
    MatrixXfRowMajor S_local = MatrixXfRowMajor::Zero(N, K);
    ColumnBlockPayload payload;

    while (true) {
        
        bool has_work = task2_queues[thread_id].try_dequeue(payload);
        if (!has_work) {
            if (task1_done[thread_id]) break;
            std::this_thread::yield();
            continue;
        }
        
        int col_start = payload.col_start;
        int local_d = payload.H_block.cols();

        MatrixXfColMajor J_block = J.block(col_start, 0, local_d, K);   // d x K

        S_local.noalias() += payload.H_block * J_block;     // Not looped by chunking for simplicity
        
    }


    #pragma omp critical
    {
        S_parallel += S_local;
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
    J = MatrixXfColMajor::Random(D, K);
    
    S_parallel = MatrixXfRowMajor::Zero(N, K);
    Y_pred_parallel = Eigen::VectorXi::Zero(N);

    H_naive = MatrixXfRowMajor::Zero(N, D);
    S_naive = MatrixXfRowMajor::Zero(N, K);
    Y_pred_naive = Eigen::VectorXi::Zero(N);


    // Naive (Generic Baseline)
    auto t1 = chrono::high_resolution_clock::now();
    H_naive.noalias() = X * B;
    apply_hardsign(H_naive); 
    S_naive.noalias() = H_naive * J;
    for (Eigen::Index i = 0; i < S_naive.rows(); ++i) {
        Eigen::Index maxIndex;
        S_naive.row(i).maxCoeff(&maxIndex);
        Y_pred_naive(i) = maxIndex;
    }
    auto t2 = chrono::high_resolution_clock::now();
    double time_naive = chrono::duration<double, milli>(t2 - t1).count();

    // Parallel
    auto t3 = chrono::high_resolution_clock::now();
    vector<thread> task1_threads, task2_threads;
    for (int t = 0; t < T; ++t)
        task2_threads.emplace_back(task2_worker, t);
    for (int t = 0; t < T; ++t)
        task1_threads.emplace_back(task1_worker, t);
    
    for (auto& t : task1_threads) t.join();
    for (auto& t : task2_threads) t.join();
    
    #pragma omp parallel for    
    for (int i = 0; i < N; ++i) {
        S_parallel.row(i).maxCoeff(&Y_pred_parallel(i));
    }
    auto t4 = chrono::high_resolution_clock::now();
    double time_parallel = chrono::duration<double, milli>(t4 - t3).count();

    // Timing
    cout << fixed << setprecision(3);
    
    cout << "################################################################################################################################# \n";
    cout << "Application Parameters ------------------------------------------------------------- \n";
    cout << "F: " << F << "    D: " << D << "       K: " << K << "\n";
    
    cout << "Algorithm HyperParameters ---------------------------------------------------------- \n";
    cout << "f: " << f << "    d: " << d << "    n: " << n << "\n";
    cout << "R: " << R << "\n";
    
    cout << "Processed Images ------------------------------------------------------------------- \n";
    cout << "N: " << N << "\n";

    cout << "Total Threads = 2*(Threads Per Task): " << 2*T << "\n";
    
    cout << "\n\n\n\n";

    cout << "Naive Time: " << time_naive    << " ms\n";
    cout << "Parallel Time: " << time_parallel << " ms\n";
    cout << "Speedup: " << time_naive/time_parallel << "\n";
    cout << "Total Threads: " << 2*T << "\n";
    cout << "Parallel Algorithm Optimality Ratio: " << (time_naive/(time_parallel*2*T)) << "\n";
    
    return 0;
}
