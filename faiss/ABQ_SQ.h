/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVF.h>
#include <stdint.h>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <mutex>
#include <tuple>

namespace faiss {

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
struct ABQ_SQ : IndexIVF {
    std::vector<std::pair<std::pair<float, float>, int>> cluster_min_max;
    std::vector<float> centroids_;
    std::vector<float*> centroids_simd;
    std::vector<float> centroids_L2;
    int B = 8;
    int max_q_code = std::pow(2, B) - 1;
    std::vector<float> cluster_value_min;
    std::vector<float> cluster_value_max;

    std::vector<float> dim_value_min;
    std::vector<float> dim_value_max;
    struct ClusterCandidate {
        idx_t id = 0;
        float distance = std::numeric_limits<float>::max();

        ClusterCandidate() = default;
        explicit ClusterCandidate(idx_t vec_id, float dis) : id(vec_id), distance(dis) {}

        friend bool operator<(const ClusterCandidate& first, const ClusterCandidate& second) {
            return first.distance < second.distance;
        }
        friend bool operator>(const ClusterCandidate& first, const ClusterCandidate& second) {
            return first.distance > second.distance;
        }
        friend bool operator>=(const ClusterCandidate& first, const ClusterCandidate& second) {
            return first.distance >= second.distance;
        }
        friend bool operator<=(const ClusterCandidate& first, const ClusterCandidate& second) {
            return first.distance <= second.distance;
        }
    };

    struct SQBlockIndex {
        int block_id;
        float min_dist = std::numeric_limits<float>::max();
        float max_dist = std::numeric_limits<float>::min();
        float theta_min = std::numeric_limits<float>::max();
        float theta_max = std::numeric_limits<float>::min();
        std::multiset<std::pair<float, std::pair<float, int>>> offset;
        std::multiset<std::pair<float, int>> offset_dist;
        uint8_t* block_vectors;
        float* vector_distances;
        float* delta_vl_q_codes;
        int block_size;
        float std;
        float std_accumulation;
        std::vector<idx_t> ids;
        void remove(const std::pair<float, std::pair<float, int>>& offset, size_t d);
    };

    struct SQ_Params {
        float vl;
        float vh;
        float delta;
        float dvl2;
        float delta_vl;
        float delta2;
        int max_q_code;

        float B;
        SQ_Params() = default;
        SQ_Params(float vl, float vh, size_t d, int B)
        {
            this->vl = vl;
            this->vh = vh;
            this->delta = (vh - vl) / (std::pow(2.0f, B) - 1.0f);
            this->dvl2 = d * vl * vl;
            this->delta_vl = delta * vl;
            this->delta2 = delta * delta;
            this->B = B;
            this->max_q_code = std::pow(2, B) - 1;
        }
    };

    SQ_Params sq_params_index;
    // 定义分区数据结构
    struct SQPartition {
        float min_dis;
        float max_dis;
        float min_dis_c;
        float max_dis_c;
        float theta_min;
        float theta_max;
        float std;
        float std_accumulation = 0.0f;
        float u;
        float delta_phi;
        float avg_sin;
        float gauss_delta;
        float exp_cos_theta;
        float var_cos_theta;
        int num = 0;
        float o_e_dist_avg;
        float o_e_dist_std;
        float max_eigenvalue;
        float second_eigenvalue;
//        std::mutex partition_mutex; // 分区级别的互斥锁
        float* centroid; // 聚类后每个组的中心向量
        float* perpendicular_vect; // 聚类后每个组的垂向量均值
        std::vector<float> perpendicular_M2; // 聚类后每个组的垂向量方差
        float perpendicular_std; // 聚类后每个组的垂向量均值
        std::vector<SQBlockIndex> blocks;
        std::pair<int, int> add(const float* x, const idx_t id, const float* normalized_vector, float dist, float dist_c, float theta, size_t d, bool ordered_block = true);
        void update_std(const float* x, const float theta, size_t d);
        std::vector<float> get_vect(idx_t b_id, size_t offset, size_t d) const;
        idx_t get_id(idx_t b_id, size_t offset) const;

        std::vector<float> data;
        std::vector<idx_t> ids;
        float centroid_dist;
        SQ_Params sqParams;
        SQPartition() = default;

        // 带参数的构造函数
        explicit SQPartition(size_t d) :
                  min_dis(std::numeric_limits<float>::max()),
                  max_dis(0.0f),
                  min_dis_c(std::numeric_limits<float>::max()),
                  max_dis_c(0.0f),
                  theta_min(std::numeric_limits<float>::max()),
                  theta_max(0.0f),
                  std(0.0f), u(0.0f), delta_phi(0.0f),
                  avg_sin(0.0f), gauss_delta(0.0f),
                  exp_cos_theta(0.0f), var_cos_theta(0.0f),
                  o_e_dist_avg(0.0f), o_e_dist_std(0.0f),
                  max_eigenvalue(0.0f), second_eigenvalue(0.0f),
                  perpendicular_std(0.0f),
                  num(0),
                  centroid_dist(0.0f),
                  centroid(nullptr), perpendicular_vect(nullptr), perpendicular_M2(d, 0.0f) {
            centroid = static_cast<float*>(aligned_alloc(64, d * sizeof(float)));
            perpendicular_vect = static_cast<float*>(aligned_alloc(64, d * sizeof(float)));
            if (!centroid || !perpendicular_vect) {
                throw std::bad_alloc();
            }
            // 初始化为0
            std::memset(centroid, 0, d * sizeof(float));
            std::memset(perpendicular_vect, 0, d * sizeof(float));
        }
    };
    // 存储每个簇的分区数据
    std::vector<std::vector<SQPartition>> partitions_in_clusters;
    std::vector<std::vector<float>> partition_data;
    // 每个簇的互斥锁，用于保护partitions_in_clusters的并发访问
    std::vector<std::unique_ptr<std::mutex>> cluster_mutexes;
    ABQ_SQ(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            bool block_by_distance = false,
            bool ordered_block = true,
            MetricType = METRIC_L2,
            bool own_invlists = true);

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void decode_vectors(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            float* x) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

//    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void query(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            float z = 0.8,
            const IVFSearchParameters* params = nullptr) const override;

    /// 训练索引，首先调用基类IndexIVF的train方法，再做其它处理
    void train(idx_t n, const float* x) override;

    void centroids_distances(
            size_t n, const float* query, size_t nprobe, std::vector<ClusterCandidate>& candidates
            ) const;
    void centroids_distances_simd(size_t n, const float* query, size_t nprobe, idx_t* coarse_idx, float* coarse_dis
     ) const;


    ABQ_SQ();
    void split(
            std::vector<SQPartition>& partitions,
            SQPartition& partition,
            size_t current_index,
            float* centroid);

    void split_kmeans(
            std::vector<SQPartition>& partitions,
            SQPartition& partition,
            size_t current_index,
            float* centroid);

    size_t get_best_index(
            const std::vector<std::vector<float>>& cosine_cache,
            const std::vector<size_t>& cluster1_indices) const;

    float angle_cos(
            const SQPartition& partition,
            const float* xi,
            float q_L2,
            float& d_q_p, bool load_ps = false) const noexcept;

    // 每个簇的垂向量均值方向（用于z值拟合和查询时的phi计算）
    std::vector<std::vector<float>> cluster_mean_perp;

    // ========== 方式1: z值分位数数据直接存储 ==========
    // 用于公式: cos φ = (<q_hat, w> + z * sqrt(2/π)) / sin α
    std::vector<std::vector<float>> z_percentile_data;  // [7][max_cluster_rank]
    int z_max_cluster_rank = 0;  // 存储的最大簇排名

    // ========== P99分位数原始值直接存储 ==========
    // 用于当z=0.99时直接查表返回，无需拟合计算
    std::vector<float> z_p99_raw;  // [max_cluster_rank] 每个j近邻的P99分位值

    // ========== 方式2: z值幂律拟合参数 ==========
    // z(j) = a * j^(-b) + c
    // 参数: a > 0 (振幅), b > 0 (衰减指数), c (背景噪声水平/渐近值)
    // 分别对应 P99.9, P99, P95, P90, P85, P80, P75 分位数
    float z_fit_a_p999 = 0.0f, z_fit_b_p999 = 0.0f, z_fit_c_p999 = 0.0f, z_fit_offset_p999 = 0.0f;  // P99.9
    float z_fit_a_p99 = 0.0f, z_fit_b_p99 = 0.0f, z_fit_c_p99 = 0.0f, z_fit_offset_p99 = 0.0f;
    float z_fit_a_p95 = 0.0f, z_fit_b_p95 = 0.0f, z_fit_c_p95 = 0.0f, z_fit_offset_p95 = 0.0f;
    float z_fit_a_p90 = 0.0f, z_fit_b_p90 = 0.0f, z_fit_c_p90 = 0.0f, z_fit_offset_p90 = 0.0f;
    float z_fit_a_p85 = 0.0f, z_fit_b_p85 = 0.0f, z_fit_c_p85 = 0.0f, z_fit_offset_p85 = 0.0f;
    float z_fit_a_p80 = 0.0f, z_fit_b_p80 = 0.0f, z_fit_c_p80 = 0.0f, z_fit_offset_p80 = 0.0f;
    float z_fit_a_p75 = 0.0f, z_fit_b_p75 = 0.0f, z_fit_c_p75 = 0.0f, z_fit_offset_p75 = 0.0f;

    // 根据分位值获取percentile索引
    // percentile: 75, 80, 85, 90, 95, 99, 999(99.9%)
    // 返回: 0-6 的索引
    int get_percentile_idx(int percentile) const {
        switch (percentile) {
            case 75: return 0;
            case 80: return 1;
            case 85: return 2;
            case 90: return 3;
            case 95: return 4;
            case 99: return 5;
            case 999: return 6;  // P99.9
            default: return 3;  // 默认返回P90的索引
        }
    }

    // 根据分位值获取对应的拟合参数 (a, b, c, offset)
    // percentile: 75, 80, 85, 90, 95, 99, 999(99.9%)
    // 返回: z(j) = a * j^(-b) + c
    std::tuple<float, float, float, float> get_z_fit_params(int percentile) const {
        switch (percentile) {
            case 999: return {z_fit_a_p999, z_fit_b_p999, z_fit_c_p999, z_fit_offset_p999};  // P99.9
            case 99: return {z_fit_a_p99, z_fit_b_p99, z_fit_c_p99, z_fit_offset_p99};
            case 95: return {z_fit_a_p95, z_fit_b_p95, z_fit_c_p95, z_fit_offset_p95};
            case 90: return {z_fit_a_p90, z_fit_b_p90, z_fit_c_p90, z_fit_offset_p90};
            case 85: return {z_fit_a_p85, z_fit_b_p85, z_fit_c_p85, z_fit_offset_p85};
            case 80: return {z_fit_a_p80, z_fit_b_p80, z_fit_c_p80, z_fit_offset_p80};
            case 75: return {z_fit_a_p75, z_fit_b_p75, z_fit_c_p75, z_fit_offset_p75};
            default: return {z_fit_a_p90, z_fit_b_p90, z_fit_c_p90, z_fit_offset_p90};
        }
    }

    // 根据分位值和簇排名获取z值（方式1：直接查表）
    // percentile: 75, 80, 85, 90, 95, 99, 999(99.9%)
    // cluster_rank: 簇排名（从1开始）
    float get_z_value(int percentile, int cluster_rank) const {
        int idx = get_percentile_idx(percentile);
        int rank = std::max(0, cluster_rank - 1);  // 转换为0-based索引
        if (rank >= z_max_cluster_rank || z_percentile_data.empty() || z_percentile_data[idx].empty()) {
            return 0.0f;  // 超出范围返回默认值
        }
        // 如果rank超出存储范围，返回最后一个有效值
        if (rank >= (int)z_percentile_data[idx].size()) {
            return z_percentile_data[idx].back();
        }
        return z_percentile_data[idx][rank];
    }

    // 根据分位值和簇排名计算z值（方式2：拟合函数）
    // percentile: 75, 80, 85, 90, 95, 99, 999(99.9%)
    // cluster_rank: 簇排名（从1开始）
    float get_z_value_fitted(int percentile, int cluster_rank) const {
        auto [a, b, c, offset] = get_z_fit_params(percentile);
        float j = (float)std::max(0, cluster_rank);
        return a * std::pow(j, -b) + c + offset;
    }
};

struct ABQ_SQDedup : ABQ_SQ {
    /** Maps ids stored in the index to the ids of vectors that are
     *  the same. When a vector is unique, it does not appear in the
     *  instances map */
    std::unordered_multimap<idx_t, idx_t> instances;

    ABQ_SQDedup(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2,
            bool own_invlists = true);

    /// also dedups the training set
    void train(idx_t n, const float* x) override;
    void train(idx_t n, const void* x, NumericType numeric_type) override;

    /// implemented for all IndexIVF* classes
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void add_with_ids(
            idx_t n,
            const void* x,
            NumericType numeric_type,
            const idx_t* xids) override;

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    size_t remove_ids(const IDSelector& sel) override;

    /// not implemented
    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    /// not implemented
    void update_vectors(int nv, const idx_t* idx, const float* v) override;

    /// not implemented
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    ABQ_SQDedup() {}
};

} // namespace faiss
