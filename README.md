# ScalableHD: Scalable Hyperdimensional Computing Implementation

A high-performance implementation of Hyperdimensional Computing (HDC) with focus on scalability, throughput, and latency optimization across multi-core systems.

## Overview

ScalableHD is a research project that implements and analyzes scalable hyperdimensional computing algorithms. The project includes both C++ implementations for high-performance computing and Python analysis tools for performance evaluation and visualization.

## Key Features

- **Multi-threaded C++ Implementation**: High-performance HDC implementation using Eigen, OpenMP, and thread binding
- **Performance Analysis**: Comprehensive throughput and latency analysis across different datasets
- **Scalability Studies**: Analysis of performance scaling with core count
- **Multiple Datasets**: Support for various datasets including EMOTION, HACT, HEART, ISOLET, MNIST, PAMAP2, SA12, and TEX
- **Visualization Tools**: Automated generation of performance plots and analysis

## Project Structure

```
├── pipehd_tiled.cpp          # Main C++ implementation with tiled matrix operations
├── parameters.h              # Configuration parameters for cores, threads, etc.
├── throughput_analysis.py    # Python analysis and visualization tools
├── quick_throughput_analysis.py  # Quick performance analysis script
├── scalability_analysis.ipynb    # Jupyter notebook for scalability studies
├── analysis.ipynb            # General analysis notebook
├── requirements.txt          # Python dependencies
├── figs/                     # Generated performance figures (PNG)
├── figs_dropped/             # Generated performance figures (PDF)
├── scalability/              # Scalability analysis results
├── throughput_results/       # Throughput analysis results
├── datasets/                 # Dataset files
├── trainable_hdc/            # Trainable HDC implementation
└── deprecated/               # Deprecated files and results
```

## Dependencies

### C++ Dependencies
- Eigen 3.4.0 (Linear algebra library)
- OpenMP (Multi-threading)
- concurrentqueue 1.0.4 (Lock-free queue)
- nlohmann/json (JSON parsing)

### Python Dependencies
- torch (PyTorch)
- torch-hd (Hyperdimensional computing library)
- numpy, pandas, matplotlib (Data analysis and visualization)
- jupyter (Notebook environment)
- scipy (Scientific computing)

## Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:dhruvashp/ScalableHD.git
   cd ScalableHD
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Compile C++ implementation**:
   ```bash
   g++ -O3 -march=native -fopenmp -I./eigen-3.4.0 -I./concurrentqueue-1.0.4 pipehd_tiled.cpp -o pipehd_tiled
   ```

## Usage

### Running the C++ Implementation

The main C++ implementation can be configured through `parameters.h`:

```cpp
#define TOTAL_CORES 32        // Total CPU cores
#define NUMA_NODES 2          // Number of NUMA nodes
#define VARIANT LOW_LAT       // HIGH_THPT or LOW_LAT
#define USE_BIND true         // Thread pinning
```

Run the implementation:
```bash
./pipehd_tiled
```

### Performance Analysis

Run the Python analysis tools:

```bash
# Quick throughput analysis
python quick_throughput_analysis.py

# Full analysis with visualization
python throughput_analysis.py
```

### Jupyter Notebooks

Open the analysis notebooks for interactive exploration:

```bash
jupyter notebook analysis.ipynb
jupyter notebook scalability_analysis.ipynb
```

## Configuration

### Platform Configuration

Edit `parameters.h` to match your system:

```cpp
#define TOTAL_CORES 32        // Your CPU core count
#define NUMA_NODES 2          // Your NUMA node count
std::vector<int> T_VALS = {1, 2, 4, 8, 16, 32};  // Thread counts to test
```

### Dataset Configuration

The implementation supports multiple datasets. Change the dataset in `pipehd_tiled.cpp`:

```cpp
std::string dataset = "EMOTION";  // Options: EMOTION, HACT, HEART, ISOLET, MNIST, PAMAP2, SA12, TEX
```

## Performance Optimization

The implementation includes several optimization strategies:

1. **Tiled Matrix Operations**: Block-based matrix operations for better cache utilization
2. **Thread Binding**: CPU affinity for consistent performance
3. **NUMA Awareness**: Memory allocation across NUMA nodes
4. **OpenMP Parallelization**: Multi-threaded computation
5. **Memory Layout Optimization**: Row-major and column-major optimizations

## Results

The project generates comprehensive performance analysis including:

- Throughput vs. core count plots
- Latency analysis
- Scalability studies
- Speedup comparisons
- Performance across different datasets

Results are automatically saved to:
- `figs/` directory (PNG format)
- `figs_dropped/` directory (PDF format)
- `throughput_results/` directory (Python data files)
- `scalability/` directory (Scalability analysis)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for research purposes. Please cite appropriately if used in academic work.

## Contact

For questions or contributions, please open an issue on GitHub or contact the maintainers. 