# ABQ Experiments

This directory contains experimental programs for evaluating the **ABQ (Adaptive Block Quantization)** index performance with different parameters and datasets.

---

## 📦 Compilation

### Build Steps

```bash
cd experiments
mkdir build
cd build
cmake ..
make nlist_abq nprobe scalability_dataset tail_latency training_data z_abq
```

After compilation, executables will be located in: `./build/`

---

## 📊 Supported Datasets

| Dataset   | Dimensions | Description                              |
|-----------|------------|------------------------------------------|
| SIFT1M    | 128        | SIFT descriptors (1M vectors)            |
| ImageNet  | 512        | CLIP embeddings (normalized)             |
| AgNews    | 1024       | mxbai embeddings (euclidean)             |
| Glove120  | 300        | Word embeddings                          |
| GooAQ     | 768        | distilroberta embeddings (normalized)    |
| Gist      | 960        | GIST descriptors                         |

---

## ⚙️ Parameter Definitions

### Common Parameters

| Parameter | Description                                                     |
|-----------|-----------------------------------------------------------------|
| `argv[1]` | Thread count (usually set to 1)                                 |
| `argv[2]` | Dataset name (SIFT1M, ImageNet, AgNews, Glove120, GooAQ, Gist)  |
| `argv[3]` | p-Percentiles value (quantization parameter, e.g., 0.095)       |
| `argv[4]` | Result output path prefix                                       |
| `argv[5]` | Index type (ABQ, ABQ_SQ, PQ, SQ, PQFS, FLAT, HNSW)              |
| `argv[6]` | block_distance flag (0 or 1, optional)                          |
| `argv[7]` | ordered_block flag (0 or 1, optional)                           |

### Index Types

| Type    | Description                    |
|---------|--------------------------------|
| ABQ     | ABQ index                      |
| ABQ_SQ  | ABQ with Scalar Quantization   |
| PQ      | Product Quantization           |
| SQ      | Scalar Quantization            |
| PQFS    | Product Quantization Fast Scan |
| FLAT    | IVF with Flat storage          |

---

## 📜 Script Descriptions

### 1. `nlist.bat` (nlist_abq)

**Purpose:** Evaluate performance with different nlist (number of clusters) values. Tests how the number of inverted lists affects search performance.

**Usage:**
```bash
../build/nlist_abq <threads> <dataset> <p_percentile> <result_path> <index_type> <block_distance>
```

**Example:**
```bash
../build/nlist_abq 1 "SIFT1M" "0.095" "${RESULT_PREFIX}/nlist/95_A_" "ABQ_SQ" 0
```

**Output:** `{dataset}_{index_type}_nlist_.txt`

---

### 2. `nprobe.bat` (nprobe)

**Purpose:** Evaluate performance with different nprobe values. Tests how the number of probed clusters affects recall and latency.

**Usage:**
```bash
../build/nprobe <threads> <dataset> <p_percentile> <result_path> <index_type> <block_distance> [ordered_block]
```

**Example:**
```bash
../build/nprobe 1 "SIFT1M" "0.095" "${RESULT_PREFIX}/nprobe/95_A_" "ABQ_SQ" 0
../build/nprobe 1 "SIFT1M" "0.095" "${RESULT_PREFIX}/nprobe/not_ordered_95_D" "ABQ" 1 0
```

**Output:** `{dataset}_{index_type}_nprobe.txt`

---

### 3. `tail_latency.bat` (tail_latency)

**Purpose:** Measure tail latency (P50, P90, P95, P99) for different index types. Analyzes query latency distribution.

**Usage:**
```bash
../build/tail_latency <threads> <dataset> <p_percentile> <result_path> <index_type> <block_distance> [ordered_block]
```

**Example:**
```bash
../build/tail_latency 1 "SIFT1M" "0.095" "${RESULT_PREFIX}/tail_latency/95_A_" "ABQ_SQ" 0
```

**Output:** `{dataset}_{index_type}_query_latency.txt`

---

### 4. `training_data.bat` (training_data)

**Purpose:** Evaluate impact of training data size on index quality. Tests with different training set sizes: 10K, 40K, 80K, 160K, 320K vectors.

**Usage:**
```bash
../build/training_data <threads> <dataset> <p_percentile> <result_path> <index_type> <block_distance> [ordered_block]
```

**Example:**
```bash
../build/training_data 1 "SIFT1M" "0.095" "${RESULT_PREFIX}/training_data/" "ABQ_SQ" 1
```

**Output:** `{dataset}_{index_type}_training_data_.txt`

---

### 5. `z_abq.bat` (p-Percentiles experiments)

**Purpose:** Evaluate the effect of different p-Percentiles values on ABQ performance.

**Tested p-Percentiles:** `0.075, 0.080, 0.085, 0.090, 0.095, 0.099`

**Usage:**
```bash
../build/nprobe <threads> <dataset> <p_percentile> <result_path> <index_type> <block_distance>
```

**Example:**
```bash
../build/nprobe 1 "SIFT1M" "0.095" "${RESULT_PREFIX}/nprobe_z/p_0.095_D_" "ABQ" 1
```

---

### 6. `scalability_dataset`

**Purpose:** Test scalability by expanding dataset size through random rotation matrices. Evaluates performance with different data multipliers.

**Usage:**
```bash
../build/scalability_dataset <dataset_name> <data_multiplier> <p_percentile> <result_path> <index_type> [block_distance] [ordered_block]
```

**Parameters:**
- `data_multiplier`: 1, 2, 4, 6, 8, 10 (dimension unchanged, data size multiplied)

**Example:**
```bash
../build/scalability_dataset AgNews 2 0.095 /data/results/ ABQ 1 1
```

---

## 🚀 Running Scripts

Navigate to the scripts directory and execute:

```bash
cd scripts

# Run nlist experiments
bash nlist.bat

# Run nprobe experiments
bash nprobe.bat

# Run tail latency experiments
bash tail_latency.bat

# Run training data experiments
bash training_data.bat

# Run p-Percentiles experiments
bash z_abq.bat
```

---

## ⚠️ Important Notes

### 1. Dataset Download Required

Before running experiments, download the required datasets:

- **SIFT1M:** fvecs format
  - `sift_base.fvecs`
  - `sift_learn.fvecs`
  - `sift_query.fvecs`
  - `sift_groundtruth.ivecs`
- **HDF5 datasets:** ImageNet, AgNews, GooAQ, etc.

Place datasets in the appropriate directories.

---

### 2. Modify `data_loader.h` File Paths

Update the `base_path` variable in `data_loader.h` to match your data directory:

**File:** `experiments/data_loader.h`

```cpp
// Change this line:
std::string base_path = "/data/Projects/data/vectors/";

// To your actual dataset location:
std::string base_path = "/your/path/to/datasets/";
```

Also update individual dataset paths if needed:
- `siftPaths.databasePath`
- `imagenetPaths` (HDF5 file path)
- `agnewsPaths` (HDF5 file path)
- `gooap` (HDF5 file path)
- `gist` paths
- `glove` paths

---

### 3. Modify Result Storage Paths

Update the `RESULT_PREFIX` variable in each script file:

```bash
# In each script, modify this line:
RESULT_PREFIX="/data/Projects/data/vectors/experiments"

# To your preferred location:
RESULT_PREFIX="/your/path/experiments"
```

Create result directories:

```bash
mkdir -p /your/path/experiments/nlist
mkdir -p /your/path/experiments/nprobe
mkdir -p /your/path/experiments/tail_latency
mkdir -p /your/path/experiments/training_data
mkdir -p /your/path/experiments/nprobe_z
```

---

### 4. Dependencies

**Required libraries:**

| Library   | Description                    |
|-----------|--------------------------------|
| HDF5      | libhdf5, libhdf5_cpp           |
| OpenBLAS  | Linear algebra operations      |
| LAPACK    | Linear algebra package         |
| OpenMP    | Multi-threading support        |

**Installation (CentOS/RHEL):**

```bash
yum install -y epel-release
yum install -y hdf5-devel
yum install -y cmake
yum install -y eigen3-devel
yum install -y openblas-devel
```

**Faiss libraries required:**
- `faiss/libs/libfaiss_avx512.a`
- `faiss/libs/libfaiss.a`

---

### 5. System Requirements

| Requirement | Description                     |
|-------------|---------------------------------|
| CPU         | AVX512 support required         |
| RAM         | Sufficient for loading datasets |
| OS          | Linux operating system          |

---