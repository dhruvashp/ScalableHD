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
#include <utility>
#include "json.hpp"
#include <fstream>




#include "parameters.h"


using namespace std;
using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXfColMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

using json = nlohmann::json;









// Metadata for code file [<mod>][parameters.h]
std::string metadata_variant = (VARIANT == HIGH_THPT) ? std::string("high_throughput") : std::string("low_latency");
std::string metadata_binding = (USE_BIND) ? std::string("binding_used") : std::string("no_binding_used");   






// Dataset [<mod>]
std::string dataset = "HACT";









// Core details [parameters.h]
int num_total_cores = TOTAL_CORES;
int numa_nodes = NUMA_NODES;

// Measurement trials [parameters.h]
int trials = TRIALS;

// Application params, fixed for an application
int F;
int D;
int K;       

// Total images
int N;                                                   

// Chunk sizes
int n;
int f;
int d;
int k;  

// Chunks to process
int R;

// Threads
int T;                      


// Derived globals
int COLUMN_BLOCKS;
int ROW_BLOCKS;
int LEFT_COLUMN_BLOCKS;
int RIGHT_COLUMN_BLOCKS;


// Global matrices
MatrixXfRowMajor X;                             // Raw features (N x F)
MatrixXfColMajor B;                             // Parameter (Base Vector, F x D)
MatrixXfColMajor J;                             // Parameter (Class Vector, Transposed, D x K)








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






typedef struct {
    float runtime_ms, throughput_img_per_sec, latency_ms_per_image;
} FloatTriple;










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







// Tiled variants
std::optional<RowBlockTiledMatrix> X_tiled;
std::optional<ColumnBlockTiledMatrix> B_tiled;
std::optional<ExtraForJColumnBlockTiledMatrix> J_tiled;

#if VARIANT == LOW_LAT
MatrixXfRowMajor S_tiled; 
#endif                   

Eigen::VectorXi Y_pred_tiled;                   // Testing the tiled variant 

vector<moodycamel::ConcurrentQueue<ColumnBlockPayload>> task2_queues_tiled;
vector<atomic<bool>> task1_done_tiled;


// ----------------------------------------- Tiled variants ---------------------------------------------------------------

// Optimizations left ------> Caching in Task II
// H --> Convert to tiled format (option)

#if VARIANT == HIGH_THPT
// Thread worker
    void task1_worker_tiled(int thread_id, int cpu_id) {
        
        if (USE_BIND) bind_thread_to_logical_core(cpu_id);      // [parameters.h]

        int chunk_size = N/T;

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
            
            // Chunking
            for (int t = 0; t < T; ++t) {
                int row_start = t * chunk_size;
                MatrixXfRowMajor chunk = h_column_block.block(row_start, 0, chunk_size, local_d);
                task2_queues_tiled[t].enqueue(ColumnBlockPayload{std::move(chunk), col_start});
            }
        
        }

        task1_done_tiled[thread_id] = true;

    }


    // Task II thread
    void task2_worker_tiled(int thread_id, int cpu_id) {
        
        if (USE_BIND) bind_thread_to_logical_core(cpu_id);              //  [parameters.h]

        int chunk_size = N / T;  // assuming N % T == 0
        MatrixXfRowMajor S_local = MatrixXfRowMajor::Zero(chunk_size, K);
        ColumnBlockPayload payload;

        while (true) {
            bool has_work = task2_queues_tiled[thread_id].try_dequeue(payload);
            if (!has_work) {
                // Check if ALL Task I threads are done
                bool all_done = true;
                for (int i = 0; i < T; ++i) {
                    if (!task1_done_tiled[i]) {
                        all_done = false;
                        break;
                    }
                }
                if (all_done) break;

                std::this_thread::yield();
                continue;
            }

            int local_d = payload.H_block.cols();

            // Go over each row block in the thread's N/T slice
            for (int row_start = 0; row_start < chunk_size; row_start += n) {
                int local_n = std::min(n, chunk_size - row_start);
                MatrixXfRowMajor left_block = payload.H_block.block(row_start, 0, local_n, local_d);

                for (int col_block = 0; col_block < RIGHT_COLUMN_BLOCKS; ++col_block) {
                    const auto& right_block = J_tiled->block(payload.col_start, col_block);
                    int local_k = right_block.cols();

                    S_local.block(row_start, col_block * k, local_n, local_k).noalias() += left_block * right_block;
                }
            }
        }

        Eigen::VectorXi Y_local = Eigen::VectorXi::Zero(chunk_size);
        for (int i = 0; i < chunk_size; ++i)
            S_local.row(i).maxCoeff(&Y_local(i));
        
        Y_pred_tiled.segment(thread_id * chunk_size, chunk_size) = Y_local;

    }
#elif VARIANT == LOW_LAT
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
#else 
    #error "Variant unknown!!"
#endif



// Correctness check
bool check_correctness(const MatrixXfRowMajor& A, const MatrixXfRowMajor& B, float tol = 1e-4f) {
    return ((A - B).cwiseAbs().maxCoeff() < tol);
}





FloatTriple run() {

    // Assert -> Switched off for now
    // assert( ((N % T) == 0) && (N >= 1024) && (K <= 1000) && "This is a high-throughput implementation, N must be sufficiently large and N must be divisible by T" );
    
    task1_done_tiled = std::vector<std::atomic<bool>>(T);  // all false by default
    task2_queues_tiled = std::vector<moodycamel::ConcurrentQueue<ColumnBlockPayload>>(T);

    COLUMN_BLOCKS = ((D + d - 1)/d);
    ROW_BLOCKS = ((N + n - 1)/n);

    LEFT_COLUMN_BLOCKS = ((F + f - 1)/f);
    RIGHT_COLUMN_BLOCKS = ((K + k - 1)/k);
    
    auto [task1_cpu_ids, task2_cpu_ids] = generate_affinity_mapping(T, num_total_cores, numa_nodes);

    // Init
    X = MatrixXfRowMajor::Random(N, F);
    B = MatrixXfColMajor::Random(F, D);
    J = MatrixXfColMajor::Random(D, K);

    X_tiled.emplace(X, n, f);   // X is N × F → Row-block-tiled with block size n × f
    B_tiled.emplace(B, f, d);   // B is F × D → Column-block-tiled with block size f × d
    J_tiled.emplace(J, d, k);   // J is D × K → Column-block-tiled with block size d × k

    #if VARIANT == LOW_LAT
    S_tiled = MatrixXfRowMajor::Zero(N, K);
    #endif
    Y_pred_tiled = Eigen::VectorXi::Zero(N);

    double total_time_tiled = 0.0;

    for (int trial = 0; trial < trials; ++trial) {

        #if VARIANT == HIGH_THPT
            // ------------------ PipeHD Tiled Variant + HIGH_THPT --------------------
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


            auto t_tiled_end = chrono::high_resolution_clock::now();
            total_time_tiled += chrono::duration<double, milli>(t_tiled_end - t_tiled_start).count();
        #elif VARIANT == LOW_LAT
            // ------------------- PipeHD Tiled Variant + LOW_LAT ---------------------
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
        #else 
            #error "Undefined variant !!"
        #endif
    }

    double avg_time_tiled = total_time_tiled / trials;                      // to process N images in avg_time_tiled ms
    double latency_per_image = avg_time_tiled / N;
    double throughput_per_sec = ((N*1e3) / avg_time_tiled); 

    cout << fixed << setprecision(3);
    cout << "################################################################################################################################# \n";
    cout << "Application Parameters ------------------------------------------------------------- \n";
    cout << "F: " << F << "    D: " << D << "       K: " << K << "\n";

    cout << "Algorithm HyperParameters ---------------------------------------------------------- \n";
    cout << "f: " << f << "    d: " << d << "    n: " << n << "      k: " << k << "\n";
    cout << "R: " << R << "\n";

    cout << "Processed Images ------------------------------------------------------------------- \n";
    cout << "N: " << N << "\n";

    cout << "Total Tiled Threads = 2*(Threads Per Task): " << 2*T << "\n";

    cout << "\n";

    cout << "Average Tiled Time over " << trials << " runs : " << avg_time_tiled << " ms\n";
    cout << "Throughput: " << throughput_per_sec << "  images/sec processed" << "\n";
    cout << "Latency: " << latency_per_image << " ms per image" << "\n";
    
    FloatTriple output;

    output.runtime_ms = avg_time_tiled;
    output.throughput_img_per_sec = throughput_per_sec;
    output.latency_ms_per_image = latency_per_image;

    if (DEBUG) print_matrix(Y_pred_tiled);

    return output;
}






int main(){

    std::vector<int> K_vals, F_vals;
    
    /*
    if (dataset == "EMOTION"){
        K_vals = {3};
        F_vals = {1500};
   }
   else if (dataset == "FACEA"){
        K_vals = {2};
        F_vals = {512};
   }
   else if (dataset == "FACE"){
        K_vals = {2};
        F_vals = {608};
   }
    */

   if (dataset == "HACT"){
        K_vals = {6};
        F_vals = {1152};
   }
   else if (dataset == "HEART"){
        K_vals = {5};
        F_vals = {187};
   }
   else if (dataset == "ISOLET"){
        K_vals = {26};
        F_vals = {617};
   }
   
   /*
   else if (dataset == "MAR"){
        K_vals = {100};
        F_vals = {64};
   }
    */
   else if (dataset == "MNIST"){
        K_vals = {10};
        F_vals = {784};
   }
   else if (dataset == "PAMAP2"){
        K_vals = {5};
        F_vals = {27};
   }
   else if (dataset == "SA12"){
        K_vals = {12};
        F_vals = {561};
   }
   else if (dataset == "TEX"){
        K_vals = {100};
        F_vals = {64};
   }
   /*
   else if (dataset == "UCICHAR"){
        K_vals = {6};
        F_vals = {561};
   }
    */
   else{
    throw std::runtime_error("Dataset invalid/not supported!!!");
   }

    // [<mod>]
    std::vector<int> D_vals = {10000};

    // [<mod>]    
    std::vector<int> N_vals = {1024, 2048, 4096, 8192, 16384, 32768};    // [HIGH_THPT]
    // std::vector<int> N_vals = {32, 64, 128, 256, 512, 1024, 2048};          // [LOW_LAT]    
    
    // [<mod>] 
    std::vector<int> nfdk_vals = {16, 32};    // n = f = d = k                  

    // [<mod>]
    std::vector<int> R_vals = {8, 16};                // Fixed for simplicity
    
    // [parameters.h]
    std::vector<int> T_vals = T_VALS;

    
    
    
    
    
    
    
    std::vector<int> N_out, F_out, D_out, K_out, n_out, f_out, d_out, k_out, R_out, T_out;
    std::vector<float> runtime_out, throughput_out, latency_out;

    for (int K_ : K_vals)
    for (int F_ : F_vals)
    for (int D_ : D_vals)
    for (int N_ : N_vals)
    for (int nfdk_ : nfdk_vals)
    for (int T_ : T_vals)   
    for (int R_ : R_vals){

        N = N_;
        F = F_;
        D = D_;
        K = K_;

        n = nfdk_;
        f = nfdk_;
        d = nfdk_;
        k = nfdk_;

        T = T_;

        R = R_;

        FloatTriple result = run();

        N_out.push_back(N);
        F_out.push_back(F);
        D_out.push_back(D);
        K_out.push_back(K);
        n_out.push_back(n);
        f_out.push_back(f);
        d_out.push_back(d);
        k_out.push_back(k);
        T_out.push_back(T);
        R_out.push_back(R);

        runtime_out.push_back(result.runtime_ms);
        throughput_out.push_back(result.throughput_img_per_sec);
        latency_out.push_back(result.latency_ms_per_image);

    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string datetime_str = std::ctime(&now_time);
    datetime_str.erase(datetime_str.find_last_not_of(" \n\r\t")+1);

    json output_json;
    output_json["datetime"] = datetime_str;
    output_json["variant"] = metadata_variant;
    output_json["binding"] = metadata_binding;
    output_json["dataset"] = dataset;
    output_json["F"] = F_out;
    output_json["D"] = D_out;
    output_json["K"] = K_out;
    output_json["N"] = N_out;
    output_json["n"] = n_out;
    output_json["f"] = f_out;
    output_json["d"] = d_out;
    output_json["k"] = k_out;
    output_json["R"] = R_out;
    output_json["T"] = T_out;
    output_json["total_runtime_ms"] = runtime_out;
    output_json["throughput_img_per_sec"] = throughput_out;
    output_json["latency_ms_per_image"] = latency_out;

    // Append to results.json
    std::ofstream file("results.json", std::ios::app);
    file << "{\n";
    bool first = true;
    for (auto& [key, value] : output_json.items()) {
    if (!first) file << ",\n";
    file << "  \"" << key << "\": " << value.dump(-1);
    first = false;
    }
    file << "\n}\n";
    file.close();

    return 0;
}