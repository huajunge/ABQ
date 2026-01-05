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

namespace faiss {

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
struct IndexPartitionBlockFlatSIMD : IndexIVF {
    std::vector<std::pair<std::pair<float, float>, int>> cluster_min_max;
    std::vector<int> cluster_partitions;

    IndexPartitionBlockFlatSIMD(
            Index* quantizer,
            size_t d,
            size_t nlist_,
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

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;
    struct BlockData {
        int block_id;
        float min_dist;
        float max_dist;
        float theta_min;
        float theta_max;
        std::vector<idx_t> offsets; // 向量在簇内的偏移
        // SIMD优化的数据存储
        float* vectors;      // 对齐的向量数据
        float* bbs_16 ;      // 对齐的向量数据
        float* bbs_8;      // 对齐的向量数据
        size_t aligned_dim = 0;        // 对齐后的维度
        float std_block = 0;
    };

    // 定义分区数据结构
    struct PartitionData {
        float min_dis;
        float max_dis;
        float theta_min;
        float theta_max;
        float std;
        float u;
        float delta_phi;
        float avg_sin;
        float gauss_delta;
        float exp_cos_theta;
        float var_cos_theta;
        int num;
        float o_e_dist_avg;
        float o_e_dist_std;
        float max_eigenvalue;
        float second_eigenvalue;
        std::vector<float> centroid; // 聚类后每个组的中心向量
        std::vector<float> std_dev; // 聚类后每个组的标准差
        std::vector<BlockData> blocks;
    };

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    // 新增函数：计算每个簇的block划分和距离范围
    void compute_blocks_per_cluster();

    // 新增函数：计算每个簇的锥形分区和block划分
    void train_blocks(int M, int block_size = 32);

    void train_blocks_std(int M, int block_size = 32);

    void train_blocks_by_clustering(int M, int block_size);
    void train_blocks_by_clustering_index(int M, int block_size);

    void train_blocks_by_clustering_raw(int M, int block_size);

    void train_blocks_by_clustering_e_vect(int M, int block_size);

    void train_blocks_by_distance(int block_size);

    // 新增函数：锥形分区查询
    void query(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int M,
            int block_size = 32,
            float iter_factor = 1.0,
            const IVFSearchParameters* params = nullptr,
            std::vector<double> times = std::vector<double>(),
            std::vector<double> ndis = std::vector<double>()) const;

    void query_perpendicular(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int M,
            int block_size = 32,
            float iter_factor = 1.0,
            const IVFSearchParameters* params = nullptr,
            std::vector<double> times = std::vector<double>(),
            std::vector<double> ndis = std::vector<double>()) const;

    void query_by_distance(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int M,
            int block_size,
            float iter_factor,
            const IVFSearchParameters* params,
            std::vector<double> &times,
            std::vector<double> &ndis) const;

    void query_by_cluster(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int M,
            int block_size,
            float iter_factor,
            const IVFSearchParameters* params,
            std::vector<double> times = std::vector<double>(),
            std::vector<double> ndis = std::vector<double>()) const;

    void query_by_cluster_raw(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int M,
            int block_size,
            float iter_factor,
            const IVFSearchParameters* params,
            std::vector<double> times = std::vector<double>(),
            std::vector<double> ndis = std::vector<double>()) const;

    void query_by_cluster_nprobe(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int M,
            int block_size,
            float iter_factor,
            const IVFSearchParameters* params,
            std::vector<double> &times,
            std::vector<double> &ndis) const;
    void e_vects_generator(float* e_vects, int d, int M);

    void e_vects_generator_M(float* e_vects, int d, int M);
    void e_vects_generator_M2(float* e_vects, int d, int M);

    void query_by_cluster_priority_queue(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int M,
            int block_size,
            float iter_factor,
            const IVFSearchParameters* params,
            std::vector<double> &times,
            std::vector<double> &ndis) const;

    void query_by_mini_blocks(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int M,
            int block_size,
            float iter_factor,
            const IVFSearchParameters* params,
            std::vector<double> &times,
            std::vector<double> &ndis) const;

    void query_by_e_vect(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int M,
            int block_size,
            const IVFSearchParameters* params,
            std::vector<double> times = std::vector<double>(),
            std::vector<double> ndis = std::vector<double>()) const;

    // 获取指定簇的block距离范围
    const std::vector<BlockData>& get_block_distances(idx_t list_no) const;

    // 获取指定簇的block数量
    size_t get_num_blocks(idx_t list_no) const;

    // 获取指定簇的指定block中的向量偏移
    const std::vector<idx_t>& get_block_offsets(idx_t list_no, size_t block_id)
            const;

    // 获取指定簇的指定block中的向量id
    std::vector<idx_t> get_block_ids(idx_t list_no, size_t block_id) const;

    std::vector<idx_t> get_block_ids(idx_t list_no, int partition_id, size_t block_id) const;

    // 获取指定簇的指定block中的向量数据
    std::vector<float> get_block_codes(idx_t list_no, size_t block_id) const;
    std::vector<float> get_block_codes(idx_t list_no, int partition_id, size_t block_id) const;

    void query_raw(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const IVFSearchParameters* params,
            std::vector<double> &times,
            std::vector<double> &ndis) const;
    // 存储每个簇的block距离范围 (min_dist, max_dist)
    std::vector<std::vector<std::pair<float, float>>> block_distances_;

    std::vector<std::vector<BlockData>> blocks_per_cluster_;

    // 存储每个簇的分区数据
    std::vector<std::vector<PartitionData>> partitions_per_cluster_;

    std::vector<float*> cluster_partition_centroids;
    std::vector<float*> cluster_partition_centroids_std;

    // 存储每个簇的单位向量e
    std::vector<std::vector<float>> e_vec_per_cluster_;

    const std::vector<float>& get_e_vec(idx_t list_no) const;

    const std::vector<PartitionData>& get_partitions(idx_t list_no) const;

    const BlockData& get_block(idx_t list_no, idx_t partition_id, idx_t block_id) const;
    // 获取指定簇的指定分区内的所有block
    const std::vector<BlockData>& get_blocks(idx_t list_no, idx_t partition_id) const;

    IndexPartitionBlockFlatSIMD();
 void norm_vector(
            const std::vector<float>& centroid,
            std::vector<float>& e_vec) const;
    void angle_bounds(
            float alpha,
            float delata,
            PartitionData& partition,
            float& L_j_q,
            float& U_j_q,
            float& theta_min,
            float& theta_max) const;
 float getLB(
         float R_q,
         float R_q2,
         float L_j_q,
         float U_j_q,
         float min_dist,
         float max_dist) const;
};

struct IndexPartitionBlockFlatSIMDDedup : IndexPartitionBlockFlatSIMD {
    /** Maps ids stored in the index to the ids of vectors that are
     *  the same. When a vector is unique, it does not appear in the
     *  instances map */
    std::unordered_multimap<idx_t, idx_t> instances;

    IndexPartitionBlockFlatSIMDDedup(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2,
            bool own_invlists = true);

    /// also dedups the training set
    void train(idx_t n, const float* x) override;

    /// implemented for all IndexIVF* classes
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

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

    IndexPartitionBlockFlatSIMDDedup() {}
};

} // namespace faiss