#include <vector>

#define HIGH_THPT 1
#define LOW_LAT 2

// [<debug>]
constexpr bool DEBUG = false;

// [<platform>]
#define TOTAL_CORES 32
#define NUMA_NODES 2

// [<platform>] [<mod>]
std::vector<int> T_VALS = {1, 2, 4, 8, 16, 32};

// [<mod>]
#define TRIALS 3

// [<mod>] ("HIGH_THPT" or "LOW_LAT")
#define VARIANT HIGH_THPT

// [<mod>] (Thread pinning)
constexpr bool USE_BIND = true;