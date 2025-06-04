

// Small N version (codenamed low-latency)


#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "concurrentqueue.h"
#include <omp.h>  // Added for OpenMP
#include <sched.h>
#include <pthread.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <optional>
#include <span>

using namespace std;
using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXfColMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

#define TOTAL_CORES 32
#define NUMA_NODES 2

// Core details
int num_total_cores = TOTAL_CORES;
int numa_nodes = NUMA_NODES;

// Measurement trials
int trials = 1;

// Application params, fixed for an application
int F = 784, D = 10000, K = 1000;       

// HyperParameters
int N = 1024;                                                   // Variable                                           Batch size, # inputs processed at once
int n = 32, f = 32, d = 32, k = 32;                             // GUIDANCE: Obvious                                  Chunk sizes, NOTE: Adding k

int R = 12;                                                      // GUIDANCE: R <= ceil(F/f) 

// For Parallel (Task I -> T threads, Task II -> T threads)
int T = 16;                                                     // GUIDANCE: T <= max_threads_feasible/2              max_threads_feasible -> Total logical processor threads/CPUs to twice that amount

// For OMP/EIGEN (OMP/EIGEN has affinity for threads = total physical cores)
int T_omp = 32;


// Global matrices
MatrixXfRowMajor X;                             // Raw features (N x F)
MatrixXfColMajor B;                             // Parameter (Base Vector, F x D)
MatrixXfColMajor J;                             // Parameter (Class Vector, Transposed, D x K)

MatrixXfRowMajor S_parallel;                    // S from parallel algorithm (N x K)
Eigen::VectorXi Y_pred_parallel;                // Prediction from parallel algorithm (N)

MatrixXfRowMajor S_tiled;                       // Testing the tiled variant
Eigen::VectorXi Y_pred_tiled;                   // Testing the tiled variant 

MatrixXfRowMajor H_omp;                         // H for OpenMP path
MatrixXfRowMajor S_omp;                         // S for OpenMP path
Eigen::VectorXi Y_pred_omp;                     // Y_pred for OpenMP path


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


std::pair<std::vector<int>, std::vector<int>> generate_affinity_mapping(int T, int num_total_cores, int numa_nodes) {
    assert(T <= num_total_cores);
    assert(num_total_cores % numa_nodes == 0);

    int cores_per_node = num_total_cores / numa_nodes;
    int logical_cpus_per_node = 2 * cores_per_node;
    int total_logical_cpus = 2 * num_total_cores;

    std::vector<int> task1_cpu_ids;
    std::vector<int> task2_cpu_ids;
    int threads_assigned = 0;

    // First pass: use thread 0 on all cores (lower half of logical CPUs)
    for (int thread_level = 0; thread_level <= 1 && threads_assigned < T; ++thread_level) {
        for (int node = 0; node < numa_nodes && threads_assigned < T; ++node) {
            int base_core = node * cores_per_node;

            for (int i = 0; i < cores_per_node && threads_assigned < T; i += 2) {
                int core_id1 = base_core + i;
                int core_id2 = base_core + i + 1;

                int cpu1 = core_id1 + thread_level * num_total_cores;
                int cpu2 = core_id2 + thread_level * num_total_cores;

                // Assign core1 to Task I, core2 to Task II (their SMT siblings will be opposite in the next pass)
                task1_cpu_ids.push_back(cpu1);
                task2_cpu_ids.push_back(cpu2);

                ++threads_assigned;
                if (threads_assigned >= T) break;
            }
        }
    }

    assert(task1_cpu_ids.size() == T && task2_cpu_ids.size() == T);
    return {task1_cpu_ids, task2_cpu_ids};
}




void bind_thread_to_logical_core(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error binding to CPU " << cpu_id << ", error code: " << rc << std::endl;
    }
}

















struct RowBlockTiledMatrix {
    
    // Objects 
    int P, Q, p, q;
    int num_block_rows, num_block_cols;
    std::vector<MatrixXfRowMajor> blocks;

    // Constructor
    RowBlockTiledMatrix(const MatrixXfRowMajor& input, int tile_p, int tile_q)
        : P(input.rows()), Q(input.cols()), p(tile_p), q(tile_q) 
    {
        num_block_rows = (P + p - 1) / p;
        num_block_cols = (Q + q - 1) / q;
        blocks.reserve(num_block_rows * num_block_cols);

        for (int i = 0; i < P; i += p) {
            int block_rows = std::min(p, P - i);
            for (int j = 0; j < Q; j += q) {
                int block_cols = std::min(q, Q - j);
                MatrixXfRowMajor block = input.block(i, j, block_rows, block_cols);
                blocks.emplace_back(std::move(block));
            }
        }
    }

    // Methods (on members)
    inline MatrixXfRowMajor& block(int i, int j) {
        return blocks[i * num_block_cols + j];
    }

    // Methods (on members)
    inline const MatrixXfRowMajor& block(int i, int j) const {
        return blocks[i * num_block_cols + j];
    }
};












struct ColumnBlockTiledMatrix {
    int P, Q, p, q;
    int num_block_rows, num_block_cols;
    std::vector<MatrixXfColMajor> blocks;

    ColumnBlockTiledMatrix(const MatrixXfColMajor& input, int tile_p, int tile_q)
        : P(input.rows()), Q(input.cols()), p(tile_p), q(tile_q)
    {
        num_block_rows = (P + p - 1) / p;
        num_block_cols = (Q + q - 1) / q;
        blocks.reserve(num_block_rows * num_block_cols);

        for (int j = 0; j < Q; j += q) {
            int block_cols = std::min(q, Q - j);
            for (int i = 0; i < P; i += p) {
                int block_rows = std::min(p, P - i);
                MatrixXfColMajor block = input.block(i, j, block_rows, block_cols);
                blocks.emplace_back(std::move(block));
            }
        }
    }

    inline MatrixXfColMajor& block(int i, int j) {
        return blocks[j * num_block_rows + i];  // Column-major block order
    }

    inline const MatrixXfColMajor& block(int i, int j) const {
        return blocks[j * num_block_rows + i];
    }
};








struct ExtraForJColumnBlockTiledMatrix {
    int P, Q, p, q;
    int num_block_rows, num_block_cols;
    std::vector<MatrixXfColMajor> blocks;

    ExtraForJColumnBlockTiledMatrix(const MatrixXfColMajor& input, int tile_p, int tile_q)
        : P(input.rows()), Q(input.cols()), p(tile_p), q(tile_q)
    {
        num_block_rows = (P + p - 1) / p;
        num_block_cols = (Q + q - 1) / q;
        blocks.reserve(num_block_rows * num_block_cols);

        for (int i = 0; i < P; i += p) {
            int block_rows = std::min(p, P - i);
            for (int j = 0; j < Q; j += q) {
                int block_cols = std::min(q, Q - j);
                MatrixXfColMajor block = input.block(i, j, block_rows, block_cols);
                blocks.emplace_back(std::move(block));
            }
        }
    }

    inline MatrixXfColMajor& block(int i, int j) {
        return blocks[i*num_block_cols + j];  
    }

    inline const MatrixXfColMajor& block(int i, int j) const {
        return blocks[i*num_block_cols + j];
    }
};




struct ExtraForHRowBlockTiledMatrix {
    
    // Objects 
    int P, Q, p, q;
    int num_block_rows, num_block_cols;
    std::vector<MatrixXfRowMajor> blocks;

    // Constructor
    ExtraForHRowBlockTiledMatrix(const MatrixXfRowMajor& input, int tile_p, int tile_q)
        : P(input.rows()), Q(input.cols()), p(tile_p), q(tile_q) 
    {
        num_block_rows = (P + p - 1) / p;
        num_block_cols = (Q + q - 1) / q;
        blocks.reserve(num_block_rows * num_block_cols);
        
        for (int j = 0; j < Q; j += q) {
            int block_cols = std::min(q, Q - j);
            for (int i = 0; i < P; i += p) {
                int block_rows = std::min(p, P - i);
                MatrixXfRowMajor block = input.block(i, j, block_rows, block_cols);
                blocks.emplace_back(std::move(block));
            }
        }
    }

    // Methods (on members)
    inline MatrixXfRowMajor& block(int i, int j) {
        return blocks[j*num_block_rows + i];
    }

    // Methods (on members)
    inline const MatrixXfRowMajor& block(int i, int j) const {
        return blocks[j*num_block_rows + i];
    }
};




// Struct for passing column blocks from Task I to Task II
struct ColumnBlockPayload {
    MatrixXfRowMajor H_block; // N x d
    int col_start;            // column location of this block
};

vector<moodycamel::ConcurrentQueue<ColumnBlockPayload>> task2_queues(T);
vector<atomic<bool>> task1_done(T);

// Thread worker
void task1_worker(int thread_id, int cpu_id) {
    
    bind_thread_to_logical_core(cpu_id);

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
void task2_worker(int thread_id, int cpu_id) {
    
    bind_thread_to_logical_core(cpu_id);

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

        // payload.H_block      ->      N x local_d         (chunk size --> n x local_d) 
        // J_bloc               ->      local_d x K         (chunk size --> local_d x k)

        for (int row_start = 0; row_start < N; row_start += n){
            int row_end = min(row_start + n, N);
            int local_n = row_end - row_start;
            for (int col_start = 0; col_start < K; col_start += k){
                int col_end = min(col_start + k, K);
                int local_k = col_end - col_start;

                S_local.block(row_start, col_start, local_n, local_k).noalias() += payload.H_block.block(row_start, 0, local_n, local_d) * J_block.block(0, col_start, local_d, local_k);
            
            } 
        }

        
    }


    #pragma omp critical
    {
        S_parallel += S_local;
    }

}




// Tiled variants
std::optional<RowBlockTiledMatrix> X_tiled;
std::optional<ColumnBlockTiledMatrix> B_tiled;
std::optional<ExtraForJColumnBlockTiledMatrix> J_tiled;


vector<moodycamel::ConcurrentQueue<ColumnBlockPayload>> task2_queues_tiled(T);
vector<atomic<bool>> task1_done_tiled(T);



int COLUMN_BLOCKS = ((D + d - 1)/d);
int ROW_BLOCKS = ((N + n - 1)/n);

int LEFT_COLUMN_BLOCKS = ((F + f - 1)/f);
int RIGHT_COLUMN_BLOCKS = ((K + k - 1)/k);

// ----------------------------------------- Tiled variants ---------------------------------------------------------------

// Optimizations left ------> Caching in Task II
// H --> Convert to tiled format (option)

// Thread worker
void task1_worker_tiled(int thread_id, int cpu_id) {
    
    bind_thread_to_logical_core(cpu_id);

    for (int col_start = thread_id; col_start < COLUMN_BLOCKS; col_start += T) {
        
        int local_d = B_tiled->block(0, col_start).cols(); 

        MatrixXfRowMajor h_column_block = MatrixXfRowMajor::Zero(N, local_d);

        for (int r_block_offset = 0; r_block_offset < LEFT_COLUMN_BLOCKS; r_block_offset += R) {
             
            int r_end = min(r_block_offset + R, LEFT_COLUMN_BLOCKS);       

            for (int row_start = 0; row_start < ROW_BLOCKS; ++row_start) {
                
                int local_n = X_tiled->block(row_start, 0).rows();

                MatrixXfRowMajor h_block_local = MatrixXfRowMajor::Zero(local_n, local_d);

                for (int block_internal_start = r_block_offset; block_internal_start < r_end; ++block_internal_start) {

                    auto& x_block_local = X_tiled->block(row_start, block_internal_start);
                    auto& b_block_local = B_tiled->block(block_internal_start, col_start);

                    h_block_local.noalias() += x_block_local * b_block_local;
            
                }

                h_column_block.block(row_start*n, 0, local_n, local_d) += h_block_local;
                
            }

        }
        
        apply_hardsign(h_column_block);
        task2_queues_tiled[thread_id].enqueue(ColumnBlockPayload{h_column_block, col_start});      
    
    }

    task1_done_tiled[thread_id] = true;

}

// Task II thread
void task2_worker_tiled(int thread_id, int cpu_id) {
    
    bind_thread_to_logical_core(cpu_id);

    MatrixXfRowMajor S_local = MatrixXfRowMajor::Zero(N, K);
    ColumnBlockPayload payload;

    while (true) {
        
        bool has_work = task2_queues_tiled[thread_id].try_dequeue(payload);
        if (!has_work) {
            if (task1_done_tiled[thread_id]) break;
            std::this_thread::yield();
            continue;
        }
        
        int local_d = payload.H_block.cols();
        
        for (int row_start = 0; row_start < ROW_BLOCKS; ++row_start){
            
            int local_n = (row_start == (ROW_BLOCKS - 1)) ? (N - ((ROW_BLOCKS - 1)*n)) : n;
            
            MatrixXfRowMajor left_block = payload.H_block.block(row_start*n, 0, local_n, local_d);
            
            for (int col_start = 0; col_start < RIGHT_COLUMN_BLOCKS; ++col_start){
                
                auto& right_block = J_tiled->block(payload.col_start, col_start);
                int local_k = right_block.cols();

                S_local.block(row_start*n, col_start*k, local_n, local_k).noalias() += left_block * right_block;     
            
            } 
        }

        
    }


    #pragma omp critical
    {
        S_tiled += S_local;
    }

}



















// Correctness check
bool check_correctness(const MatrixXfRowMajor& A, const MatrixXfRowMajor& B, float tol = 1e-4f) {
    return ((A - B).cwiseAbs().maxCoeff() < tol);
}

int main() {

    // Assert -> Switched off for now
    // assert( (N <= 1024) && (K <= 1000) && "This is a low-latency version and does not support large N values -- use high-throughput implementation for this" );
    
    auto [task1_cpu_ids, task2_cpu_ids] = generate_affinity_mapping(T, num_total_cores, numa_nodes);

    // Init
    X = MatrixXfRowMajor::Random(N, F);
    B = MatrixXfColMajor::Random(F, D);
    J = MatrixXfColMajor::Random(D, K);

    X_tiled.emplace(X, n, f);   // X is N × F → Row-block-tiled with block size n × f
    B_tiled.emplace(B, f, d);   // B is F × D → Column-block-tiled with block size f × d
    J_tiled.emplace(J, d, k);   // J is D × K → Column-block-tiled with block size d × k

    S_parallel = MatrixXfRowMajor::Zero(N, K);
    Y_pred_parallel = Eigen::VectorXi::Zero(N);

    S_tiled = MatrixXfRowMajor::Zero(N, K);
    Y_pred_tiled = Eigen::VectorXi::Zero(N);

    double total_time_parallel = 0.0;
    double total_time_omp = 0.0;
    double total_time_tiled = 0.0;

    for (int trial = 0; trial < trials; ++trial) {

        // ------------------ OpenMP Parallelization ------------------
        Eigen::initParallel();
        Eigen::setNbThreads(T_omp);       // T_omp as OMP affinity different from ours

        Y_pred_omp.setZero(N);

        auto t_omp_start = chrono::high_resolution_clock::now();
        
        H_omp.noalias() = X * B;

        #pragma omp parallel for
        for (int i = 0; i < H_omp.rows(); ++i)
            for (int j = 0; j < H_omp.cols(); ++j)
                H_omp(i, j) = H_omp(i, j) >= 0.0f ? 1.0f : -1.0f;
        
        S_omp.noalias() = H_omp * J;

        #pragma omp parallel for
        for (int i = 0; i < N; ++i)
            S_omp.row(i).maxCoeff(&Y_pred_omp(i));
        
        auto t_omp_end = chrono::high_resolution_clock::now();
        total_time_omp += chrono::duration<double, milli>(t_omp_end - t_omp_start).count();
        cout << "OpenMP done in : " << total_time_omp << " ms \n";








        
        // ------------------ PipeHD Parallel ------------------
        S_parallel.setZero();
        Y_pred_parallel.setZero();
        for (int i = 0; i < T; ++i) {
            task2_queues[i] = moodycamel::ConcurrentQueue<ColumnBlockPayload>();
            task1_done[i] = false;
        }

        auto t3 = chrono::high_resolution_clock::now();
        vector<thread> task1_threads, task2_threads;
        for (int t = 0; t < T; ++t)
            task2_threads.emplace_back(task2_worker, t, task2_cpu_ids[t]);
        for (int t = 0; t < T; ++t)
            task1_threads.emplace_back(task1_worker, t, task1_cpu_ids[t]);
        for (auto& t : task1_threads) t.join();
        for (auto& t : task2_threads) t.join();

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            S_parallel.row(i).maxCoeff(&Y_pred_parallel(i));
        }
        
        auto t4 = chrono::high_resolution_clock::now();
        total_time_parallel += chrono::duration<double, milli>(t4 - t3).count();











        // ------------------ PipeHD Tiled Variant --------------------
        S_tiled.setZero();
        Y_pred_tiled.setZero();
        for (int i = 0; i < T; ++i) {
            task2_queues_tiled[i] = moodycamel::ConcurrentQueue<ColumnBlockPayload>();
            task1_done_tiled[i] = false;
        }

        auto t_tiled_start = chrono::high_resolution_clock::now();
        vector<thread> task1_threads_tiled, task2_threads_tiled;
        
        for (int t = 0; t < T; ++t)
            task2_threads_tiled.emplace_back(task2_worker_tiled, t, task2_cpu_ids[t]);
        for (int t = 0; t < T; ++t)
            task1_threads_tiled.emplace_back(task1_worker_tiled, t, task1_cpu_ids[t]);
        for (auto& t : task1_threads_tiled) t.join();
        for (auto& t : task2_threads_tiled) t.join();

        
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            S_tiled.row(i).maxCoeff(&Y_pred_tiled(i));
        }

        auto t_tiled_end = chrono::high_resolution_clock::now();
        total_time_tiled += chrono::duration<double, milli>(t_tiled_end - t_tiled_start).count();
    
    }

    double avg_time_parallel = total_time_parallel / trials;
    double avg_time_omp = total_time_omp / trials;
    double avg_time_tiled = total_time_tiled / trials;


    print_matrix(Y_pred_omp);
    print_matrix(Y_pred_parallel);
    print_matrix(Y_pred_tiled);


    cout << fixed << setprecision(3);

    cout << "################################################################################################################################# \n";
    cout << "Application Parameters ------------------------------------------------------------- \n";
    cout << "F: " << F << "    D: " << D << "       K: " << K << "\n";

    cout << "Algorithm HyperParameters ---------------------------------------------------------- \n";
    cout << "f: " << f << "    d: " << d << "    n: " << n << "      k: " << k << "\n";
    cout << "R: " << R << "\n";

    cout << "Processed Images ------------------------------------------------------------------- \n";
    cout << "N: " << N << "\n";

    cout << "Total Parallel Threads = 2*(Threads Per Task): " << 2*T << "\n";
    cout << "Total Tiled Threads = 2*(Threads Per Task): " << 2*T << "\n";
    cout << "Total OpenMP Threads: " << T_omp << "\n";

    cout << "\n";

    cout << "Average Parallel Time over " << trials << " runs : " << avg_time_parallel << " ms\n";
    cout << "Average Tiled Time over " << trials << " runs : " << avg_time_tiled << " ms\n";
    cout << "Average OpenMP Time over " << trials << " runs   : " << avg_time_omp << " ms\n";


    cout << "Parallel vs OpenMP: " << avg_time_omp / avg_time_parallel << "                (>1 Win, =1 Tie, <1 Lose) \n"; 
    cout << "Tiled vs Parallel: " << avg_time_parallel / avg_time_tiled << "               (>1 Win, =1 Tie, <1 Lose) \n";

    return 0;
}
