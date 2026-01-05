/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexPartitionBlockFlatSIMD.h>

#include <omp.h>

#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <random>

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>
#include <eigen3/Eigen/Dense>
#include <limits>
#include <queue>

#ifdef __AVX2__
#include <immintrin.h> // 必需的头文件
#endif

#ifdef __AVX512F__
#include <immintrin.h> // 必需的头文件
#endif

namespace faiss {

/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

IndexPartitionBlockFlatSIMD::IndexPartitionBlockFlatSIMD(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric,
        bool own_invlists)
        : IndexIVF(
                  quantizer,
                  d,
                  nlist,
                  sizeof(float) * d,
                  metric,
                  own_invlists) {
    code_size = sizeof(float) * d;
    by_residual = false;
}

IndexPartitionBlockFlatSIMD::IndexPartitionBlockFlatSIMD() {
    by_residual = false;
}

void IndexPartitionBlockFlatSIMD::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(!by_residual);
    assert(invlists);
    direct_map.check_can_add(xids);

    int64_t n_add = 0;

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];

            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                size_t offset = invlists->add_entry(
                        list_no, id, (const uint8_t*)xi, inverted_list_context);
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("IndexPartitionBlockFlatSIMD::add_core: added %" PRId64
               " / %" PRId64 " vectors\n",
               n_add,
               n);
    }
    ntotal += n;
}

void IndexPartitionBlockFlatSIMD::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(!by_residual);
    if (!include_listnos) {
        memcpy(codes, x, code_size * n);
    } else {
        size_t coarse_size = coarse_code_size();
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            uint8_t* code = codes + i * (code_size + coarse_size);
            const float* xi = x + i * d;
            if (list_no >= 0) {
                encode_listno(list_no, code);
                memcpy(code + coarse_size, xi, code_size);
            } else {
                memset(code, 0, code_size + coarse_size);
            }
        }
    }
}

void IndexPartitionBlockFlatSIMD::sa_decode(
        idx_t n,
        const uint8_t* bytes,
        float* x) const {
    size_t coarse_size = coarse_code_size();
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = bytes + i * (code_size + coarse_size);
        float* xi = x + i * d;
        memcpy(xi, code + coarse_size, code_size);
    }
}

void norm_unit_vector_simd(
        float* centroid,
        const float* vec,
        float* e_vec,
        int d) {
    for (size_t i = 0; i < d; i++) {
        e_vec[i] = vec[i] - centroid[i];
    }

    float norm_rel = fvec_norm_L2sqr(e_vec, d);
    norm_rel = std::sqrt(norm_rel);
    for (size_t j = 0; j < d; j++) {
        e_vec[j] /= norm_rel;
    }
}

void IndexPartitionBlockFlatSIMD::compute_blocks_per_cluster() {
    FAISS_THROW_IF_NOT(is_trained);
    //    block_distances_.resize(nlist);
    blocks_per_cluster_.resize(nlist);
#pragma omp parallel for schedule(dynamic)
    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        if (list_size == 0) {
            //            block_distances_[list_no].clear();
            continue;
        }
        std::vector<float> centroid(d);
        // 获取簇心
        quantizer->reconstruct(list_no, centroid.data());

        // 获取簇内所有向量
        InvertedLists::ScopedCodes scodes(invlists, list_no);
        const float* codes = (const float*)scodes.get();
        InvertedLists::ScopedIds ids(invlists, list_no);

        // 计算每个向量到簇心的距离
        std::vector<std::pair<float, idx_t>> dists_with_indices;
        dists_with_indices.reserve(list_size);
        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;
            float dist;
            if (metric_type == METRIC_INNER_PRODUCT) {
                dist = -fvec_inner_product(centroid.data(), vec, d);
            } else {
                dist = fvec_L2sqr(centroid.data(), vec, d);
            }
            dists_with_indices.emplace_back(dist, i);
        }

        // 对距离排序
        std::sort(dists_with_indices.begin(), dists_with_indices.end());

        // 每32条数据分组并记录距离范围
        const size_t block_size = 16;
        size_t num_blocks = (list_size + block_size - 1) / block_size;
        //        blocks_per_cluster_[list_no].resize(num_blocks);
        std::vector<BlockData> block_map(num_blocks);
        for (size_t b = 0; b < num_blocks; b++) {
            size_t start = b * block_size;
            size_t end = std::min((b + 1) * block_size, list_size);

            // 记录block的距离范围
            float min_dist = dists_with_indices[start].first;
            float max_dist = dists_with_indices[end - 1].first;

            // 存储block内的向量偏移
            std::vector<idx_t> offsets;
            offsets.reserve(end - start);
            for (size_t i = start; i < end; i++) {
                offsets.push_back(dists_with_indices[i].second);
            }
            block_map[b] = {
                    static_cast<int>(b),
                    min_dist,
                    max_dist,
                    0.0,
                    0.0,
                    std::move(offsets)};
            //            blocks_per_cluster_[list_no][b] = {min_dist, max_dist,
            //            std::move(offsets)};
        }
        blocks_per_cluster_[list_no] = std::move(block_map);
    }
}

void train_e_vect(
        float* e_vect,
        size_t n,
        int d,
        const float* codes,
        float* centriod) {
    int max_i = 0;
    int max_j = 0;
    float max_dist = 0.0;
    auto* a = new float[d];
    auto* b = new float[d];
    for (int i = 0; i < n - 1; i++) {
        norm_unit_vector_simd(centriod, codes + i * d, a, d);
        for (int j = i + 1; j < n; j++) {
            norm_unit_vector_simd(centriod, codes + j * d, b, d);
            float dist = fvec_L2sqr(a, b, d);
            if (max_dist < dist) {
                max_dist = dist;
                max_i = i;
                max_j = j;
            }
        }
    }
    norm_unit_vector_simd(centriod, codes + max_i * d, a, d);
    norm_unit_vector_simd(centriod, codes + max_j * d, b, d);
    norm_unit_vector_simd(a, b, e_vect, d);
    //    printf("max_dist: %f\n", std::sqrt(max_dist));
}

static std::mutex stats_mutex;
void IndexPartitionBlockFlatSIMD::train_blocks(int M, int block_size) {
    FAISS_THROW_IF_NOT(is_trained);
    blocks_per_cluster_.resize(nlist);
    e_vec_per_cluster_.resize(nlist);
    cluster_min_max.resize(nlist);
    cluster_partitions.resize(nlist);
#pragma omp parallel for schedule(dynamic)
    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        if (list_size == 0) {
            continue;
        }

        //        int partition_num = (list_size + block_size * block_factor -
        //        1) / (block_size * block_factor);

        int partition_num = 16;
        if (list_size < partition_num * block_size) {
            partition_num = (list_size + block_size - 1) / block_size;
        }

        int block_factor = ((list_size + block_size - 1) / block_size +
                            partition_num - 1) /
                partition_num;
        partition_num = (list_size + block_size * block_factor - 1) /
                (block_size * block_factor);

        std::vector<float> centroid(d);
        // 获取簇心
        quantizer->reconstruct(list_no, centroid.data());

        // 获取簇内所有向量
        InvertedLists::ScopedCodes scodes(invlists, list_no);
        const float* codes = (const float*)scodes.get();
        InvertedLists::ScopedIds ids(invlists, list_no);

        // 计算单位向量e = [1,1,...,1]/sqrt(d)
        //        std::vector<float> e_vec(d, 1.0f / std::sqrt(d));
        //        for (size_t i = 0; i < d; i++) {
        //            if ((i + 1) % 2 == 0) {
        //                e_vec[i] *= -1.0f;
        //            }
        //        }
        //       norm_vector(centroid, e_vec);

        // 构建数据矩阵（中心化后的向量）
        std::vector<float> data(list_size * d);
        std::vector<float> e_vec(d, 0.0f);
        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;
            for (size_t j = 0; j < d; j++) {
                data[i * d + j] = vec[j] - centroid[j];
            }

            float dist = std::sqrt(fvec_norm_L2sqr(data.data() + i * d, d));
            for (size_t j = 0; j < d; j++) {
                data[i * d + j] /= dist;
                e_vec[j] += data[i * d + j];
            }
        }

        // 利用平均值作为中心向量
        for (size_t j = 0; j < d; j++) {
            e_vec[j] /= list_size;
        }

        float norm = fvec_norm_L2sqr(e_vec.data(), d);
        norm = std::sqrt(norm);
        for (size_t i = 0; i < d; i++) {
            e_vec[i] /= norm;
        }

        //                 对簇内所有向量进行PCA，找到第一主成分

        //               std::vector<float> e_vec(d, 1.0 /std::sqrt(d));
        //               if (list_size > 1) {
        //                   // 构建数据矩阵（中心化后的向量）
        //       //            // 计算协方差矩阵
        //                   std::vector<float> cov(d * d, 0.0f);
        //                   for (size_t i = 0; i < d; i++) {
        //                       for (size_t j = 0; j < d; j++) {
        //                           float sum = 0.0f;
        //                           for (size_t k = 0; k < list_size; k++) {
        //                               sum += data[k * d + i] * data[k * d +
        //                               j];
        //                           }
        //                           cov[i * d + j] = sum / (list_size - 1);
        //                       }
        //                   }
        //       //
        //       //            // 使用幂迭代法找到最大特征向量（第一主成分）
        //                   std::vector<float> v(d, 1.0f / std::sqrt(d));
        //                   //初始向量
        //                   for (int iter = 0; iter < 10; iter++) {
        //                   // 迭代10次
        //                       // 矩阵乘法: cov * v
        //                       std::vector<float> w(d, 0.0f);
        //                       for (size_t i = 0; i < d; i++) {
        //                           for (size_t j = 0; j < d; j++) {
        //                               w[i] += cov[i * d + j] * v[j];
        //                           }
        //                       }
        //
        //                       // 归一化
        //                       float norm = fvec_norm_L2sqr(w.data(), d);
        //                       norm = std::sqrt(norm);
        //                       for (size_t i = 0; i < d; i++) {
        //                           v[i] = w[i] / norm;
        //                       }
        //                   }
        //                   e_vec = v;
        //               } else {
        //                   // 当簇内向量不足时，使用默认向量
        //                   for (size_t i = 0; i < d; i++) {
        //                       e_vec[i] = 1.0f;
        //                   }
        //                   float norm = fvec_norm_L2sqr(e_vec.data(), d);
        //                   norm = std::sqrt(norm);
        //                   for (size_t i = 0; i < d; i++) {
        //                       e_vec[i] /= norm;
        //                   }
        //               }

        //                        norm_vector(centroid, e_vec);
        //
        //        std::vector<float> e_vec(d, 0.0f);
        //        train_e_vect(e_vec.data(), list_size, d, codes,
        //        centroid.data());

        // 使用Eigen3计算特征向量
        //  对簇内所有向量进行PCA
        float max_eigenvalue = 0.0f;    // 最大特征值
        float second_eigenvalue = 0.0f; // 第二大特征值（垂向量方向）

        //       std::vector<float> e_vec(d, 1.0 /std::sqrt(d));
        //       if (list_size > 1) {
        //           // 使用Eigen进行PCA
        //           Eigen::Map<Eigen::MatrixXf> X(data.data(), list_size, d);
        //           Eigen::VectorXf mean = X.colwise().mean(); //
        //           计算每个特征（列）的均值
        //           // 中心化数据
        //           Eigen::MatrixXf X_centered = X.rowwise() -
        //           mean.transpose(); Eigen::MatrixXf cov =
        //           (X_centered.transpose() * X_centered) / (list_size - 1);
        //
        //
        //           // 特征分解
        //           Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov);
        //           if (solver.info() != Eigen::Success) {
        //               // 特征分解失败处理
        //               std::fill(e_vec.begin(), e_vec.end(), 0.0f);
        //               max_eigenvalue = 0.0f;
        //               second_eigenvalue = 0.0f;
        //           } else {
        //               // 获取特征值和特征向量
        //               Eigen::VectorXf eigenvalues =
        //               solver.eigenvalues().real(); Eigen::MatrixXf
        //               eigenvectors = solver.eigenvectors().real();
        //               // 特征值按升序排列，取最后两个最大的
        //               max_eigenvalue = eigenvalues[0];
        //               second_eigenvalue = eigenvalues[1];
        //
        //               // 步骤 4: 找到最小特征值对应的索引
        //               int min_index;
        //               eigenvalues.maxCoeff(&min_index); //
        //               获取最小特征值的索引 Eigen::VectorXf min_eigenvector =
        //               eigenvectors.col(min_index);
        //               min_eigenvector.normalize();
        //               // 获取最大特征值对应的特征向量
        ////               Eigen::VectorXf max_eigenvector =
        ///eigenvectors.col(0); /               max_eigenvector.normalize();
        //               // 复制到输出向量
        //               e_vec.resize(d);
        //               Eigen::VectorXf::Map(e_vec.data(), d) =
        //               min_eigenvector;
        //           }
        //       }


        //用c来作为中心
//        float c_dist = std::sqrt(faiss::fvec_norm_L2sqr(centroid.data(), d));
//        for (int i = 0; i < d; i++) {
//            e_vec[i] = centroid[i] / c_dist;
//        }
        float e_d = fvec_norm_L2sqr(e_vec.data(), d);

        e_vec_per_cluster_[list_no] = e_vec;

        // 存储每个向量的信息：偏移、距离、角度
        struct VectorData {
            long offset;
            float dist;
            float theta;
            float o_e_dist;
        };
        std::vector<VectorData> vecs(list_size);

        // 计算每个向量的距离和角度
        partitions_per_cluster_.resize(nlist);

        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;

            //            for (size_t j = 0; j < d; j++) {
            //                data[i * d + j] -= e_vec[j];
            //            }

            // 计算距离
            float dist;
            if (metric_type == METRIC_INNER_PRODUCT) {
                dist = -fvec_inner_product(centroid.data(), vec, d);
            } else {
                dist = fvec_L2sqr(centroid.data(), vec, d);
                dist = std::sqrt(dist);
            }

            // 计算单位化向量
            //            std::vector<float> rel_vec_2(d);
            //            for (size_t j = 0; j < d; j++) {
            //                rel_vec_2[j] = vec[j] - centroid[j];
            //            }

            float* rel_vec = data.data() + i * d;
            auto norm_rel = fvec_norm_L2sqr(rel_vec, d);
            if (norm_rel == 0) {
                // 零向量，角度设为0
                vecs[i] = {static_cast<long>(i), dist, 0.0f, 0.0f};
            } else {
                norm_rel = std::sqrt(norm_rel);
                // 计算与e的点积
                float dot = fvec_inner_product(rel_vec, e_vec.data(), d);
                float o_e_dist = std::sqrt(2 - 2 * dot);
                dot = std::max(
                        -1.0f, std::min(1.0f, dot)); // 确保在[-1,1]范围内
                float theta = std::acos(dot);
                vecs[i] = {static_cast<long>(i), dist, theta, o_e_dist};
            }
        }
        // 按照角度升序排序
        std::sort(
                vecs.begin(),
                vecs.end(),
                [](const VectorData& a, const VectorData& b) {
                    return a.theta < b.theta;
                });

        float start_theta = vecs.begin()->theta / M_PI * 180;
        float end_theta = vecs[vecs.size() - 1].theta / M_PI * 180;
        float interval = 5;
        // 清空该簇的分区数据
        std::vector<PartitionData>& partitions =
                partitions_per_cluster_[list_no];
        partitions.reserve((end_theta - start_theta) / interval);
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        int block_nums = 0;
        int p_id = 0;
        for (int s_index = 0; s_index < vecs.size();) {
            //            if (p_id + 1 >= partitions.size()) {
            //                printf("list_no: %d, p:%d, theta_min: %.6f,
            //                theta_max: %.6f\n", list_no, p_id,
            //                       partitions[p_id].theta_min / M_PI * 180,
            //                       partitions[p_id].theta_max / M_PI * 180);
            //            }

            const VectorData& item_s = vecs[s_index];
            float theta_s = item_s.theta / M_PI * 180;
            int start_bin = (int)(theta_s / interval);
            int end_index = s_index;
            int v_num = vecs.size();
            int count = 0;
            bool flag = false;
            for (; end_index < v_num; end_index++) {
                const VectorData& item_j = vecs[end_index];
                float theta_j = item_j.theta / M_PI * 180;
                int j_bin = (int)(theta_j / interval);
                if (j_bin != start_bin) {
                    flag = true;
                }
                if (flag && count >= block_size) {
                    break;
                }
                count++;
            }

            size_t start = s_index;
            size_t end = end_index;

            // 分区角度范围
            PartitionData partition;
            partition.theta_min = vecs[start].theta;
            partition.theta_max = vecs[end - 1].theta;
            //
            //            printf("list_no: %d, p:%d, theta_min: %.6f, theta_max:
            //            %.6f\n",
            //                   list_no,
            //                   m,
            //                   partition.theta_min / M_PI * 180,
            //                   partition.theta_max / M_PI * 180);
            float sum = 0.0;
            float sum_sim = 0.0;

            float exp_cos = 0.0;
            float var_cos = 0.0;
            float exp_cos_2 = 0.0;

            size_t num_in_partition = end - start;
            partition.num = num_in_partition;
            std::vector<float> tmp_data(num_in_partition * d);
            // 计算垂向量和垂向量标准差
            float* vect_per = new float[num_in_partition * d];
            // 计算均值向量
            float* vect_avg = new float[num_in_partition * d];
            for (size_t i = 0; i < num_in_partition; i++) {
                idx_t offset = vecs[start + i].offset;
                float ip = std::cos(vecs[start + i].theta);
                sum += vecs[start + i].theta;
                for (int j = 0; j < d; j++) {
                    vect_per[i * d + j] = data[offset * d + j] - ip * e_vec[j];
                    vect_avg[i * d + j] = data[offset * d + j];
                }
            }

            // 归一化
            for (size_t i = 0; i < num_in_partition; i++) {
                float perp_dist = fvec_norm_L2sqr(vect_per + i * d, d);
                if (perp_dist > 0.0) {
                    perp_dist = std::sqrt(perp_dist);
                    for (int j = 0; j < d; j++) {
                        vect_per[i * d + j] /= perp_dist;
                    }
                }
            }

            // 计算每个组的中心向量
            std::vector<float> pert_centroid(d, 0.0); // 聚类后每个组的中心向量
            std::vector<float> pert_centroid_avg(
                    d, 0.0); // 聚类后每个组的中心向量
            for (size_t i = 0; i < num_in_partition; i++) {
                for (int j = 0; j < d; j++) {
                    pert_centroid[j] += vect_per[i * d + j];
                    pert_centroid_avg[j] += vect_avg[i * d + j];
                }
            }

            for (int j = 0; j < d; j++) {
                pert_centroid[j] /= num_in_partition;
                pert_centroid_avg[j] /= num_in_partition;
            }

            // 计算垂向量标准差
            std::vector<float> sum_std(d, 0.0);
            for (size_t i = 0; i < num_in_partition; i++) {
                for (int j = 0; j < d; j++) {
                    float tmp_data_j = vect_per[i * d + j] - pert_centroid[j];
                    sum_std[j] += tmp_data_j * tmp_data_j;
                }
            }

            std::vector<float> std_dev(d);
            float std_sum = 0.0;
            for (int kk = 0; kk < d; ++kk) {
                float variance = (sum_std[kk] / num_in_partition);
                if (variance < 0.0) {
                    variance = -variance;
                }
                variance = std::sqrt(variance);
                std_sum += variance;
                std_dev[kk] = variance;
            }

            std_sum /= (float)d;
            partition.centroid = pert_centroid;
            partition.std_dev.resize(
                    static_cast<size_t>(d), 0.0f); // 调整大小并初始化为0
            memcpy(partition.std_dev.data(), std_dev.data(), sizeof(float) * d);

            partition.max_eigenvalue = max_eigenvalue;
            partition.second_eigenvalue = second_eigenvalue;

            exp_cos /= (float)num_in_partition;
            exp_cos_2 /= (float)num_in_partition;

            for (size_t i = 0; i < num_in_partition; i++) {
                float delta = std::cos(vecs[start + i].theta) - exp_cos;
                var_cos += delta * delta;
            }

            var_cos /= (float)num_in_partition;

            //            partition.exp_cos_theta = exp_cos;
            //            sum_sim /= (float)num_in_partition;
            //            partition.avg_sin = std::sqrt(sum_sim);
            //            partition.var_cos_theta = var_cos;

            partition.exp_cos_theta = (std::sin(partition.theta_max) -
                                       std::sin(partition.theta_min)) /
                    (partition.theta_max - partition.theta_min);
            exp_cos_2 = (std::sin(2.0 * partition.theta_max) -
                         std::sin(2.0 * partition.theta_min)) /
                    (4.0 * (partition.theta_max - partition.theta_min));
            partition.var_cos_theta = 0.5 + exp_cos_2 -
                    partition.exp_cos_theta * partition.exp_cos_theta;
            partition.avg_sin = std::sqrt(0.5 - exp_cos_2);

            float avg = sum / (float)num_in_partition;
            // 计算标准差
            float std = 0.0;
            for (size_t i = 0; i < num_in_partition; i++) {
                float tmp = std::acos(vecs[start + i].theta) - avg;
                std += tmp * tmp;
            }

            std = std / (float)num_in_partition;
            //                float delta_phi = std::sqrt(std * std + 0.5 *
            //                sum_sim);
            float delta_phi = std::sqrt(std + 0.5 * avg * avg);
            std = std::sqrt(std);
            partition.u = avg;
            //            partition.std = std;
            partition.std = std_sum;
            partition.delta_phi = delta_phi;

            // 分区内按距离排序
            std::sort(
                    vecs.begin() + start,
                    vecs.begin() + end,
                    [](const VectorData& a, const VectorData& b) {
                        return a.dist < b.dist;
                    });
            partition.min_dis = vecs[start].dist;
            partition.max_dis = vecs[end - 1].dist;
            //
            // 分区内分块
            size_t num_blocks = (end - start + block_size - 1) / block_size;
            partition.blocks.resize(num_blocks);
            block_nums += num_blocks;
            min = std::min(min, partition.min_dis);
            max = std::max(max, partition.max_dis);

            float sum_o_e_dist = 0.0f;
            for (size_t b = 0; b < num_blocks; b++) {
                size_t block_start = start + b * block_size;
                size_t block_end = std::min(start + (b + 1) * block_size, end);

                // 计算block的距离范围
                float min_dist = vecs[block_start].dist;
                float max_dist = vecs[block_end - 1].dist;

                // 收集block内的向量偏移
                std::vector<idx_t> offsets;
                offsets.reserve(block_end - block_start);

                for (size_t i = block_start; i < block_end; i++) {
                    sum_o_e_dist += vecs[i].o_e_dist;
                    offsets.push_back(vecs[i].offset);
                }
                // 存储block数据
                partition.blocks[b] = {
                        static_cast<int>(b),
                        min_dist,
                        max_dist,
                        partition.theta_min,
                        partition.theta_max,
                        std::move(offsets)};
            }
            partition.o_e_dist_avg = sum_o_e_dist / num_in_partition;
            partitions.emplace_back(std::move(partition));
            //                        partitions[p_id] = std::move(partition);
            p_id++;
            //            printf("list_no: %d, p_id: %d, block_nums: %d, min:
            //            %.6f, max: %.6f\n", list_no, p_id, block_nums, min,
            //            max);
            s_index = end_index;
        }

        //        printf("list_no: %ld, list_size: %ld, theta_start:
        //        %.6f,theta_end: %.6f\n", list_no, list_size,
        //               vecs.begin()->theta /M_PI * 180,  vecs.end()[-1].theta
        //               / M_PI * 180);
        //        // 计算每个锥形分区的大小
        //        size_t partition_size = block_size * block_factor;
        //        if (partition_size < 1)
        //            partition_size = 1;
        //
        //        // 清空该簇的分区数据
        //        std::vector<PartitionData>& partitions =
        //                partitions_per_cluster_[list_no];
        //        partitions.clear();
        //        partitions.resize(partition_num); // 预分配M个分区
        //        float min = std::numeric_limits<float>::max();
        //        float max = std::numeric_limits<float>::min();
        //        int block_nums = 0;
        //        // 遍历每个锥形分区
        ////#pragma omp parallel for num_threads(partition_num)
        //        for (int m = 0; m < partition_num; m++) {
        //            size_t start = m * partition_size;
        //            size_t end = (m + 1) * partition_size > list_size ?
        //            list_size : (m + 1) * partition_size;
        //
        //            // 分区角度范围
        //            PartitionData partition;
        //            partition.theta_min = vecs[start].theta;
        //            partition.theta_max = vecs[end - 1].theta;
        //
        //            printf("list_no: %d, p:%d, theta_min: %.6f, theta_max:
        //            %.6f\n", list_no, m, partition.theta_min / M_PI * 180,
        //            partition.theta_max / M_PI * 180); float sum = 0.0; float
        //            sum_sim = 0.0;
        //
        //            float exp_cos = 0.0;
        //            float var_cos = 0.0;
        //            float exp_cos_2 = 0.0;
        //
        //            size_t num_in_partition = end - start;
        //            partition.num = num_in_partition;
        //            std::vector<float> tmp_data(num_in_partition * d);
        //            //计算垂向量和垂向量标准差
        //            float* vect_per = new float[num_in_partition * d];
        //            //计算均值向量
        //            float* vect_avg = new float[num_in_partition * d];
        //            for (size_t i = 0; i < num_in_partition; i++) {
        //                idx_t offset = vecs[start + i].offset;
        //                float ip = std::cos(vecs[start + i].theta);
        //                sum += vecs[start + i].theta;
        //                for (int j = 0; j < d; j++) {
        //                    vect_per[i * d + j] = data[offset * d + j] - ip *
        //                    e_vec[j]; vect_avg[i * d + j] = data[offset * d +
        //                    j];
        //                }
        //            }
        //
        //            //归一化
        //            for (size_t i = 0; i < num_in_partition; i++) {
        //                float perp_dist = fvec_norm_L2sqr(vect_per + i * d,
        //                d); if (perp_dist > 0.0) {
        //                    perp_dist = std::sqrt(perp_dist);
        //                    for (int j = 0; j < d; j++) {
        //                        vect_per[i * d + j] /= perp_dist;
        //                    }
        //                }
        //            }
        //
        //            //计算每个组的中心向量
        //            std::vector<float> pert_centroid(d, 0.0); //
        //            聚类后每个组的中心向量 std::vector<float>
        //            pert_centroid_avg(d, 0.0); // 聚类后每个组的中心向量 for
        //            (size_t i = 0; i < num_in_partition; i++) {
        //                for (int j = 0; j < d; j++) {
        //                    pert_centroid[j] += vect_per[i * d + j];
        //                    pert_centroid_avg[j] += vect_avg[i * d + j];
        //                }
        //            }
        //
        //            for (int j = 0; j < d; j++) {
        //                pert_centroid[j] /= num_in_partition;
        //                pert_centroid_avg[j] /= num_in_partition;
        //            }
        //
        ////            float d_per =
        ///std::sqrt(fvec_norm_L2sqr(pert_centroid_avg.data(), d)); / if (d_per
        ///> 0.0) { /                for (int j = 0; j < d; j++) { /
        ///pert_centroid_avg[j] /= d_per; /                } /            }
        //
        //
        ////            //分组均值向量
        ////            std::vector<float> sum_std(d, 0.0);
        ////            for (size_t i = 0; i < num_in_partition; i++) {
        ////                for (int j = 0; j < d; j++) {
        ////                    float tmp_data_j = vect_avg[i * d + j] -
        ///pert_centroid_avg[j]; /                    sum_std[j] += tmp_data_j *
        ///tmp_data_j; /                } /            }
        ////
        ////            std::vector<float> std_dev(d);
        ////            float std_sum = 0.0;
        ////            for (int kk = 0; kk < d; ++kk) {
        ////                float variance = (sum_std[kk] / num_in_partition);
        ////                if (variance < 0.0) {
        ////                    variance = -variance;
        ////                }
        ////                variance = std::sqrt(variance);
        ////                std_sum += variance;
        ////                std_dev[kk] = variance;
        ////            }
        ////
        ////            std_sum /= (float) d;
        ////            partition.centroid = pert_centroid_avg;
        ////            float d_avg = fvec_norm_L2sqr(pert_centroid_avg.data(),
        ///d); /            partition.std_dev.resize(static_cast<size_t>(d),
        ///0.0f);  // 调整大小并初始化为0 / memcpy(partition.std_dev.data(),
        ///std_dev.data(), sizeof(float) * d);
        //
        //            //计算垂向量标准差
        //            std::vector<float> sum_std(d, 0.0);
        //            for (size_t i = 0; i < num_in_partition; i++) {
        //                for (int j = 0; j < d; j++) {
        //                    float tmp_data_j = vect_per[i * d + j] -
        //                    pert_centroid[j]; sum_std[j] += tmp_data_j *
        //                    tmp_data_j;
        //                }
        //            }
        //
        //            std::vector<float> std_dev(d);
        //            float std_sum = 0.0;
        //            for (int kk = 0; kk < d; ++kk) {
        //                float variance = (sum_std[kk] / num_in_partition);
        //                if (variance < 0.0) {
        //                    variance = -variance;
        //                }
        //                variance = std::sqrt(variance);
        //                std_sum += variance;
        //                std_dev[kk] = variance;
        //            }
        //
        //            std_sum /= (float) d;
        //            partition.centroid = pert_centroid;
        //            partition.std_dev.resize(static_cast<size_t>(d), 0.0f); //
        //            调整大小并初始化为0 memcpy(partition.std_dev.data(),
        //            std_dev.data(), sizeof(float) * d);
        //
        //
        ////            if (num_in_partition > 1) {
        ////                // 使用Eigen进行PCA
        ////                Eigen::Map<Eigen::MatrixXf> X(tmp_data.data(),
        ///num_in_partition, d); /                Eigen::MatrixXf cov =
        ///(X.transpose() * X) / (num_in_partition - 1); /                //
        ///特征分解 / Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf>
        ///solver(cov); /                if (solver.info() != Eigen::Success) {
        ////                    // 特征分解失败处理
        ////                    std::fill(e_vec.begin(), e_vec.end(), 0.0f);
        ////                    max_eigenvalue = 0.0f;
        ////                    second_eigenvalue = 0.0f;
        ////                } else {
        ////                    // 获取特征值和特征向量
        ////                    Eigen::VectorXf eigenvalues =
        ///solver.eigenvalues(); /                    Eigen::MatrixXf
        ///eigenvectors = solver.eigenvectors();
        ////
        ////                    // 特征值按升序排列，取最后两个最大的
        ////                    max_eigenvalue = eigenvalues[eigenvalues.size()
        ///- 1]; /                    second_eigenvalue =
        ///eigenvalues[eigenvalues.size() - 2];
        ////
        ////                    // 获取最大特征值对应的特征向量
        ////                    Eigen::VectorXf max_eigenvector =
        ///eigenvectors.col(eigenvalues.size() - 1); /
        ///max_eigenvector.normalize(); /                    // 复制到输出向量
        ////                    e_vec.resize(d);
        ////                    Eigen::VectorXf::Map(e_vec.data(), d) =
        ///max_eigenvector; /                } /            }
        //
        //            partition.max_eigenvalue = max_eigenvalue;
        //            partition.second_eigenvalue = second_eigenvalue;
        //
        //            exp_cos /= (float)num_in_partition;
        //            exp_cos_2 /= (float)num_in_partition;
        //
        //            for (size_t i = 0; i < num_in_partition; i++) {
        //                float delta = std::cos(vecs[start + i].theta) -
        //                exp_cos; var_cos += delta * delta;
        //            }
        //
        //            var_cos /= (float)num_in_partition;
        //
        ////            partition.exp_cos_theta = exp_cos;
        ////            sum_sim /= (float)num_in_partition;
        ////            partition.avg_sin = std::sqrt(sum_sim);
        ////            partition.var_cos_theta = var_cos;
        //
        //            partition.exp_cos_theta = (std::sin( partition.theta_max)
        //            -std::sin( partition.theta_min))/ ( partition.theta_max -
        //            partition.theta_min); exp_cos_2 = (std::sin(2.0 *
        //            partition.theta_max) - std::sin(2.0 *
        //            partition.theta_min)) / (4.0 * (partition.theta_max -
        //            partition.theta_min)); partition.var_cos_theta = 0.5 +
        //            exp_cos_2 - partition.exp_cos_theta *
        //            partition.exp_cos_theta; partition.avg_sin = std::sqrt(0.5
        //            - exp_cos_2);
        //
        //            float avg = sum / (float)num_in_partition;
        //            // 计算标准差
        //            float std = 0.0;
        //            for (size_t i = 0; i < num_in_partition; i++) {
        //                float tmp = std::acos(vecs[start + i].theta) - avg;
        //                std += tmp * tmp;
        //            }
        //
        //            std = std / (float)num_in_partition;
        //            //                float delta_phi = std::sqrt(std * std +
        //            0.5 *
        //            //                sum_sim);
        //            float delta_phi = std::sqrt(std + 0.5 * avg * avg);
        //            std = std::sqrt(std);
        //            partition.u = avg;
        ////            partition.std = std;
        //            partition.std = std_sum;
        //            partition.delta_phi = delta_phi;
        //
        //            // 分区内按距离排序
        //            std::sort(
        //                    vecs.begin() + start,
        //                    vecs.begin() + end,
        //                    [](const VectorData& a, const VectorData& b) {
        //                        return a.dist < b.dist;
        //                    });
        //            partition.min_dis = vecs[start].dist;
        //            partition.max_dis = vecs[end - 1].dist;
        //            //
        //            // 分区内分块
        //            size_t num_blocks = (end - start + block_size - 1) /
        //            block_size; partition.blocks.resize(num_blocks);
        //            block_nums += num_blocks;
        //            min = std::min(min, partition.min_dis);
        //            max = std::max(max, partition.max_dis);
        //
        //            float sum_o_e_dist = 0.0f;
        //            for (size_t b = 0; b < num_blocks; b++) {
        //                size_t block_start = start + b * block_size;
        //                size_t block_end = std::min(start + (b + 1) *
        //                block_size, end);
        //
        //                // 计算block的距离范围
        //                float min_dist = vecs[block_start].dist;
        //                float max_dist = vecs[block_end - 1].dist;
        //
        //                // 收集block内的向量偏移
        //                std::vector<idx_t> offsets;
        //                offsets.reserve(block_end - block_start);
        //
        //                for (size_t i = block_start; i < block_end; i++) {
        //                    sum_o_e_dist += vecs[i].o_e_dist;
        //                    offsets.push_back(vecs[i].offset);
        //                }
        //                // 存储block数据
        //                partition.blocks[b] = {
        //                        static_cast<int>(b),
        //                        min_dist,
        //                        max_dist,
        //                        partition.theta_min,
        //                        partition.theta_max,
        //                        std::move(offsets)};
        //            }
        //            partition.o_e_dist_avg = sum_o_e_dist /  num_in_partition;
        //            partitions[m] = std::move(partition);
        //        }
        cluster_min_max[list_no].first = {min, max};
        cluster_min_max[list_no].second = block_nums;
        cluster_partitions[list_no] = partitions.size();
    }
}

void IndexPartitionBlockFlatSIMD::train_blocks_std(int M, int block_size) {
    FAISS_THROW_IF_NOT(is_trained);
    blocks_per_cluster_.resize(nlist);
    e_vec_per_cluster_.resize(nlist);
    cluster_min_max.resize(nlist);
    cluster_partitions.resize(nlist);
#pragma omp parallel for schedule(dynamic)
    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        if (list_size == 0) {
            continue;
        }

        int partition_num = 4;
        if (list_size < partition_num * block_size) {
            partition_num = (list_size + block_size - 1) / block_size;
        }

        int block_factor = ((list_size + block_size - 1) / block_size +
                            partition_num - 1) /
                partition_num;
        partition_num = (list_size + block_size * block_factor - 1) /
                (block_size * block_factor);
        //        int block_factor = (list_size + block_size - 1) / block_size;
        //        int partition_num = (list_size + block_size * block_factor -
        //        1) /
        //                (block_size * block_factor);
        std::vector<float> centroid(d);
        // 获取簇心
        quantizer->reconstruct(list_no, centroid.data());

        // 获取簇内所有向量
        InvertedLists::ScopedCodes scodes(invlists, list_no);
        const float* codes = (const float*)scodes.get();
        InvertedLists::ScopedIds ids(invlists, list_no);
        // 构建数据矩阵（中心化后的向量）
        std::vector<float> data(list_size * d);
        std::vector<float> vec_centroid(d, 0.0);
        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;
            for (size_t j = 0; j < d; j++) {
                data[i * d + j] = vec[j] - centroid[j];
            }

            float dist = std::sqrt(fvec_norm_L2sqr(data.data() + i * d, d));
            for (size_t j = 0; j < d; j++) {
                data[i * d + j] /= dist;
                vec_centroid[j] += data[i * d + j];
            }
        }

        for (size_t j = 0; j < d; j++) {
            vec_centroid[j] /= (float)list_size;
        }

        float e_d = std::sqrt(fvec_norm_L2sqr(vec_centroid.data(), d));

        e_vec_per_cluster_[list_no] = vec_centroid;

        // 存储每个向量的信息：偏移、距离、角度
        struct VectorData {
            long offset;
            float dist;
            float theta;
            float o_e_dist;
        };
        std::vector<VectorData> vecs(list_size);

        // 计算每个向量的距离和角度
        partitions_per_cluster_.resize(nlist);

        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;
            // 计算距离
            float dist;
            if (metric_type == METRIC_INNER_PRODUCT) {
                dist = -fvec_inner_product(centroid.data(), vec, d);
            } else {
                dist = fvec_L2sqr(centroid.data(), vec, d);
                dist = std::sqrt(dist);
            }
            float sum_std = 0.0;
            for (int j = 0; j < d; j++) {
                float tmp_data_j = data[i * d + j] - vec_centroid[j];
                sum_std += tmp_data_j * tmp_data_j;
            }
            sum_std /= (float)d;
            sum_std = std::sqrt(sum_std);

            float* rel_vec = data.data() + i * d;
            auto norm_rel = fvec_norm_L2sqr(rel_vec, d);
            if (norm_rel == 0) {
                // 零向量，角度设为0
                vecs[i] = {static_cast<long>(i), dist, 0.0f, 0.0f};
            } else {
                norm_rel = std::sqrt(norm_rel);
                // 计算与e的点积
                float dot =
                        fvec_inner_product(rel_vec, vec_centroid.data(), d) /
                        e_d;
                float o_e_dist = std::sqrt(2 - 2 * dot);
                dot = std::max(
                        -1.0f, std::min(1.0f, dot)); // 确保在[-1,1]范围内
                float theta = std::acos(dot);
                vecs[i] = {static_cast<long>(i), dist, sum_std, o_e_dist};
            }
        }
        // 按照角度升序排序
        std::sort(
                vecs.begin(),
                vecs.end(),
                [](const VectorData& a, const VectorData& b) {
                    return a.theta < b.theta;
                });
        // 计算每个锥形分区的大小
        size_t partition_size = block_size * block_factor;
        if (partition_size < 1)
            partition_size = 1;

        // 清空该簇的分区数据
        std::vector<PartitionData>& partitions =
                partitions_per_cluster_[list_no];
        partitions.clear();
        partitions.resize(partition_num); // 预分配M个分区
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        int block_nums = 0;
        // 遍历每个锥形分区
        // #pragma omp parallel for num_threads(partition_num)
        for (int m = 0; m < partition_num; m++) {
            size_t start = m * partition_size;
            size_t end = (m + 1) * partition_size > list_size
                    ? list_size
                    : (m + 1) * partition_size;

            // 分区角度范围
            PartitionData partition;
            partition.theta_min = vecs[start].theta;
            partition.theta_max = vecs[end - 1].theta;

            //            printf("list_no: %d, p:%d, theta_min: %.6f, theta_max:
            //            %.6f\n", list_no, m, partition.theta_min / M_PI * 180,
            //            partition.theta_max / M_PI * 180);
            float std_sum = 0.0;
            float sum_sim = 0.0;

            float exp_cos = 0.0;
            float var_cos = 0.0;
            float exp_cos_2 = 0.0;

            size_t num_in_partition = end - start;
            partition.num = num_in_partition;
            std::vector<float> tmp_data(num_in_partition * d);
            std::vector<float> sum_std(d, 0.0);
            std::vector<float> sum_mean(d, 0.0);
            std::vector<float> sum_var(d, 0.0);

            for (size_t offset = 0; offset < num_in_partition; offset++) {
                float theta = vecs[start + offset].theta;
                std_sum += theta;
                const float* nvec_norm =
                        data.data() + vecs[start + offset].offset * d;
                for (size_t j = 0; j < d; j++) {
                    float diff = nvec_norm[j] - vec_centroid[j];
                    sum_var[j] += diff * diff;
                }
            }
            float avg = std_sum / (float)num_in_partition;
            partition.std = avg;
            //            printf("list_no: %d, p:%d, std: %.6f\n", list_no, m,
            //            partition.std);
            // 计算每个维度的std向量
            std::vector<float> std_dev(d);
            for (int kk = 0; kk < d; ++kk) {
                float variance = (sum_var[kk] / num_in_partition);
                if (variance < 0.0) {
                    variance = -variance;
                }
                variance = std::sqrt(variance);
                std_dev[kk] = variance;
            }
            partition.std_dev.resize(d);
            memcpy(partition.std_dev.data(), std_dev.data(), sizeof(float) * d);

            // 分区内按距离排序
            std::sort(
                    vecs.begin() + start,
                    vecs.begin() + end,
                    [](const VectorData& a, const VectorData& b) {
                        return a.dist < b.dist;
                    });
            partition.min_dis = vecs[start].dist;
            partition.max_dis = vecs[end - 1].dist;
            min = std::min(min, partition.min_dis);
            max = std::max(max, partition.max_dis);
            size_t num_blocks = (end - start + block_size - 1) / block_size;
            partition.blocks.resize(num_blocks);
            block_nums += num_blocks;
            float sum_o_e_dist = 0.0f;
            for (size_t b = 0; b < num_blocks; b++) {
                size_t block_start = start + b * block_size;
                size_t block_end = std::min(start + (b + 1) * block_size, end);

                // 计算block的距离范围
                float min_dist = vecs[block_start].dist;
                float max_dist = vecs[block_end - 1].dist;

                // 收集block内的向量偏移
                std::vector<idx_t> offsets;
                offsets.reserve(block_end - block_start);

                for (size_t i = block_start; i < block_end; i++) {
                    sum_o_e_dist += vecs[i].o_e_dist;
                    offsets.push_back(vecs[i].offset);
                }
                // 存储block数据
                partition.blocks[b] = {
                        static_cast<int>(b),
                        min_dist,
                        max_dist,
                        partition.theta_min,
                        partition.theta_max,
                        std::move(offsets)};
            }
            partition.o_e_dist_avg = sum_o_e_dist / num_in_partition;
            partitions[m] = std::move(partition);
        }
        cluster_min_max[list_no].first = {min, max};
        cluster_min_max[list_no].second = block_nums;
        cluster_partitions[list_no] = partitions.size();
    }
}

void IndexPartitionBlockFlatSIMD::train_blocks_by_clustering_e_vect(
        int M,
        int block_size) {
    FAISS_THROW_IF_NOT(is_trained);
    partitions_per_cluster_.resize(nlist);
    e_vec_per_cluster_.resize(nlist);

    // 定义四个基向量
    std::vector<float> e1(d, 1.0f / std::sqrt(d));
    std::vector<float> e2(d, -1.0f / std::sqrt(d));
    std::vector<float> e3(d);
    std::vector<float> e4(d);

    for (int i = 0; i < d; i++) {
        if (i % 2 == 0) {
            e3[i] = -1.0f / std::sqrt(d);
            e4[i] = 1.0f / std::sqrt(d);
        } else {
            e3[i] = 1.0f / std::sqrt(d);
            e4[i] = -1.0f / std::sqrt(d);
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        if (list_size == 0) {
            partitions_per_cluster_[list_no].clear();
            continue;
        }

        // 获取簇心
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        // 获取簇内所有向量
        InvertedLists::ScopedCodes scodes(invlists, list_no);
        const float* codes = (const float*)scodes.get();
        InvertedLists::ScopedIds ids(invlists, list_no);

        // 存储每个分组的数据: 距离和偏移
        std::vector<std::vector<std::pair<float, idx_t>>> groups(4);
        std::vector<float> group_theta_min(
                4, std::numeric_limits<float>::max());
        std::vector<float> group_theta_max(
                4, std::numeric_limits<float>::lowest());
        std::vector<float> group_rmin(4, std::numeric_limits<float>::max());
        std::vector<float> group_rmax(4, std::numeric_limits<float>::lowest());

        // 遍历簇内每个向量
        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;

            // 计算中心化向量
            std::vector<float> centered(d);
            for (int j = 0; j < d; j++) {
                centered[j] = vec[j] - centroid[j];
            }

            // 计算距离
            float dist = fvec_norm_L2sqr(centered.data(), d);
            dist = std::sqrt(dist);

            // 归一化中心化向量
            if (dist > 0) {
                for (int j = 0; j < d; j++) {
                    centered[j] /= dist;
                }
            }

            //            // 存储所有基向量
            //            std::vector<float> centroid_e1(d);
            //            std::vector<float> centroid_e2(d);
            //            std::vector<float> centroid_e3(d);
            //            std::vector<float> centroid_e4(d);
            //            norm_unit_vector(centroid, e1.data(), centroid_e1, d);
            //            norm_unit_vector(centroid, e2.data(), centroid_e2, d);
            //            norm_unit_vector(centroid, e3.data(), centroid_e3, d);
            //            norm_unit_vector(centroid, e4.data(), centroid_e4, d);

            std::vector<std::vector<float>> base_vectors = {e1, e2, e3, e4};

            // 计算与每个基向量的内积
            std::vector<float> ips(4);
            for (int j = 0; j < 4; j++) {
                ips[j] = fvec_inner_product(
                        centered.data(), base_vectors[j].data(), d);
            }

            // 找到内积最大的基向量
            int best_group = 0;
            float best_ip = ips[0];
            for (int j = 1; j < 4; j++) {
                if (ips[j] > best_ip) {
                    best_ip = ips[j];
                    best_group = j;
                }
            }

            // 计算角度
            float theta = std::acos(std::max(-1.0f, std::min(1.0f, best_ip)));

            // 添加到对应分组
            groups[best_group].push_back(std::make_pair(dist, i));

            // 更新分组的角度范围
            if (theta < group_theta_min[best_group]) {
                group_theta_min[best_group] = theta;
            }
            if (theta > group_theta_max[best_group]) {
                group_theta_max[best_group] = theta;
            }

            // 更新分组的距离范围
            if (dist < group_rmin[best_group]) {
                group_rmin[best_group] = dist;
            }
            if (dist > group_rmax[best_group]) {
                group_rmax[best_group] = dist;
            }
        }

        // 为当前簇创建分区数据
        std::vector<PartitionData>& partitions =
                partitions_per_cluster_[list_no];
        partitions.clear();

        // 处理每个分组
        for (int group_idx = 0; group_idx < 4; group_idx++) {
            if (groups[group_idx].empty())
                continue;

            PartitionData partition;
            partition.theta_min = group_theta_min[group_idx];
            partition.theta_max = group_theta_max[group_idx];
            partition.min_dis = group_rmin[group_idx];
            partition.max_dis = group_rmax[group_idx];

            // 按距离排序
            std::sort(
                    groups[group_idx].begin(),
                    groups[group_idx].end(),
                    [](const std::pair<float, idx_t>& a,
                       const std::pair<float, idx_t>& b) {
                        return a.first < b.first;
                    });

            // 分块
            size_t num_blocks =
                    (groups[group_idx].size() + block_size - 1) / block_size;
            partition.blocks.resize(num_blocks);

            for (size_t b = 0; b < num_blocks; b++) {
                size_t start = b * block_size;
                size_t end = std::min(
                        (b + 1) * block_size, groups[group_idx].size());

                BlockData block;
                block.block_id = b;
                block.theta_min = partition.theta_min;
                block.theta_max = partition.theta_max;

                // 计算块内距离范围
                block.min_dist = groups[group_idx][start].first;
                block.max_dist = groups[group_idx][end - 1].first;

                // 收集块内向量偏移
                for (size_t i = start; i < end; i++) {
                    block.offsets.push_back(groups[group_idx][i].second);
                }

                partition.blocks[b] = block;
            }

            partitions.push_back(partition);
        }
    }
}

void IndexPartitionBlockFlatSIMD::train_blocks_by_clustering_raw(
        int M,
        int block_size) {
    FAISS_THROW_IF_NOT(is_trained);
    printf("start:------\n");
    partitions_per_cluster_.resize(nlist);
    e_vec_per_cluster_.resize(nlist);
    //    printf("start resize:------\n");
    std::mt19937 rng(1234);

    //    for (idx_t list_no = 0; list_no < nlist; list_no++) {
    //        size_t list_size = invlists->list_size(list_no);
    //        //        printf("list_size: %d\n", list_size);
    //        if (list_size == 0) {
    //            partitions_per_cluster_[list_no].clear();
    //            continue;
    //        }
    //        // 获取簇内所有向量
    //        InvertedLists::ScopedCodes scodes(invlists, list_no);
    //        const float* codes = (const float*)scodes.get();
    //        InvertedLists::ScopedIds ids(invlists, list_no);
    //
    //        for (size_t i = 0; i < list_size - 1; i++) {
    //            const float* vec = codes + i * d;
    //            float d1 = fvec_norm_L2sqr(vec, d);
    //            for (size_t j = i + 1; j < list_size; j++) {
    //                const float* vec2 = codes + j * d;
    //                float d2 = fvec_norm_L2sqr(vec2, d);
    //                float dis = fvec_inner_product(vec, vec2, d);
    //                printf("list_no: %d, ip: %f, d1: %f, d2: %f, cos: %f\n",
    //                list_no, dis, d1, d2, dis / (std::sqrt(d1) *
    //                std::sqrt(d2)));
    //            }
    //        }
    //    }

#pragma omp parallel for schedule(dynamic)
    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        //        printf("list_size: %d\n", list_size);
        if (list_size == 0) {
            partitions_per_cluster_[list_no].clear();
            continue;
        }

        // 获取簇心
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        // 获取簇内所有向量
        InvertedLists::ScopedCodes scodes(invlists, list_no);
        const float* codes = (const float*)scodes.get();
        InvertedLists::ScopedIds ids(invlists, list_no);

        // 中心化并归一化向量
        std::vector<float> normalized_vectors(list_size * d);
        //        normalized_vectors.resize(list_size * d);
        //        printf("中心化并归一化向量:------\n");
        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;
            float* nvec = normalized_vectors.data() + i * d;
            // 归一化: o / ||o||
            float norm = fvec_norm_L2sqr(vec, d);
            if (norm > 0) {
                norm = std::sqrt(norm);
                for (size_t j = 0; j < d; j++) {
                    nvec[j] = vec[j] / norm;
                }
            }
        }
        //        printf("使用k-means聚类将数据分成M个组:------\n");

        // 使用k-means聚类将数据分成M个组
        int effective_M = std::min(static_cast<int>(list_size), M);
        if (effective_M <= 0) {
            partitions_per_cluster_[list_no].clear();
            continue;
        }
        //        printf("初始化聚类中心:------\n");
        // 初始化聚类中心
        std::mt19937 rng(1234);
        // 使用k-means++初始化聚类中心
        //        std::vector<float> centroids(effective_M * d, 0);
        //        std::vector<float> min_distances(list_size, 0);
        //        std::vector<idx_t> centroids_indices(effective_M);

        //        // 第一个中心点随机选择
        //        std::uniform_int_distribution<idx_t> rand(0, list_size - 1);
        //        idx_t first_idx = rand(rng);
        //        centroids_indices[0] = first_idx;
        //        const float* first_vec = normalized_vectors.data() + first_idx
        //        * d; memcpy(centroids.data(), first_vec, sizeof(float) * d);
        //
        //        // 计算每个点到最近中心的距离
        //        auto update_min_distances = [&](int current_centroid_count) {
        //            for (size_t i = 0; i < list_size; i++) {
        //                const float* vec = normalized_vectors.data() + i * d;
        //                float best_dist = -1.0f;
        //                for (int c = 0; c <= current_centroid_count; c++) {
        //                    const float* centroid = centroids.data() + c * d;
        //                    float dist = fvec_inner_product(vec, centroid, d);
        //                    if (dist > best_dist) {
        //                        best_dist = dist;
        //                    }
        //                }
        //                min_distances[i] = 1.0f - best_dist;  //
        //                转换为最小化问题
        //            }
        //        };
        //
        //        update_min_distances(0);
        //
        //        // 选择剩余的中心点
        //        for (int c = 1; c < effective_M; c++) {
        //            // 计算距离平方和
        //            float total_distance = 0.0f;
        //            for (size_t i = 0; i < list_size; i++) {
        //                total_distance += min_distances[i];
        //            }
        //
        //            // 按概率选择下一个中心点
        //            std::uniform_real_distribution<float> randf(0.0f,
        //            total_distance); float threshold = randf(rng); float
        //            cumulative = 0.0f; idx_t next_idx = 0; for (size_t i = 0;
        //            i < list_size; i++) {
        //                cumulative += min_distances[i];
        //                if (cumulative >= threshold) {
        //                    next_idx = i;
        //                    break;
        //                }
        //            }
        //
        //            centroids_indices[c] = next_idx;
        //            const float* next_vec = normalized_vectors.data() +
        //            next_idx * d; memcpy(centroids.data() + c * d, next_vec,
        //            sizeof(float) * d);
        //
        //            // 更新最小距离
        //            update_min_distances(c);
        //        }

        std::vector<float> centroids(effective_M * d, 0);
        std::vector<idx_t> centroids_indices;
        centroids_indices.resize(effective_M);
        centroids.resize(effective_M);
        std::vector<idx_t> perm(list_size);
        for (idx_t i = 0; i < list_size; i++) {
            perm[i] = i;
        }

        //        std::shuffle(
        //                perm.begin(), perm.end(),
        //                std::mt19937(std::random_device()()));
        std::shuffle(perm.begin(), perm.end(), std::mt19937(1024));
        for (int i = 0; i < effective_M; i++) {
            const float* nvec = normalized_vectors.data() + perm[i] * d;
            memcpy(centroids.data() + i * d, nvec, sizeof(float) * d);
        }

        // 分配向量到聚类
        std::vector<idx_t> assignment(list_size, -1);
        int max_iter = 25;
        bool changed = true;
        for (int iter = 0; iter < max_iter && changed; iter++) {
            changed = false;
            // 分配步骤：将每个点分配到最近的中心
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < list_size; i++) {
                const float* nvec = normalized_vectors.data() + i * d;
                float best_dist = -1;
                int best_cluster = -1;
                for (int c = 0; c < effective_M; c++) {
                    const float* cvec = centroids.data() + c * d;
                    float dist = fvec_inner_product(nvec, cvec, d);
                    if (dist > best_dist) {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                if (assignment[i] != best_cluster) {
                    changed = true;
                    assignment[i] = best_cluster;
                }
                //                printf("iter: %d, changed: %d, best_dist:
                //                %f\n", iter, changed, best_dist);
            }

            if (!changed)
                break;

            // 更新步骤：重新计算中心
            std::vector<float> new_centroids(effective_M * d, 0);
            new_centroids.resize(effective_M * d);
            std::vector<int> counts(effective_M, 0);
            for (size_t i = 0; i < list_size; i++) {
                int cluster = assignment[i];
                const float* nvec = normalized_vectors.data() + i * d;
                float* new_centroid = new_centroids.data() + cluster * d;
                for (size_t j = 0; j < d; j++) {
                    new_centroid[j] += nvec[j];
                }
                counts[cluster]++;
            }

            // 归一化每个中心
            for (int c = 0; c < effective_M; c++) {
                if (counts[c] == 0) {
                    // 随机选择一个点作为新中心
                    std::uniform_int_distribution<idx_t> rand(0, list_size - 1);
                    idx_t rand_idx = rand(rng);
                    const float* rand_vec =
                            normalized_vectors.data() + rand_idx * d;
                    memcpy(new_centroids.data() + c * d,
                           rand_vec,
                           sizeof(float) * d);
                    // 设置计数为1，避免后续归一化时除以0
                    counts[c] = 1;
                }

                float* new_centroid = new_centroids.data() + c * d;
                if (counts[c] > 0) {
                    float norm = fvec_norm_L2sqr(new_centroid, d);
                    if (norm > 0) {
                        norm = std::sqrt(norm);
                        for (size_t j = 0; j < d; j++) {
                            new_centroid[j] /= norm;
                        }
                    }
                }
            }
            centroids.swap(new_centroids);
        }

        // 为每个聚类（分区）收集向量
        std::vector<std::vector<idx_t>> cluster_indices(effective_M);
        for (size_t i = 0; i < list_size; i++) {
            int cluster = assignment[i];
            cluster_indices[cluster].push_back(i);
        }

        // 为每个分区创建PartitionData
        std::vector<PartitionData> partitions(effective_M);
        for (int m = 0; m < effective_M; m++) {
            PartitionData& partition = partitions[m];
            partition.centroid.resize(d);
            float* c = centroids.data() + m * d;
            memcpy(partition.centroid.data(), c, sizeof(float) * d);

            // 获取当前分区的所有向量索引
            std::vector<idx_t>& indices = cluster_indices[m];
            size_t num_in_partition = indices.size();

            // 计算每个向量到分区中心的距离
            if (num_in_partition > 0) {
                std::vector<std::pair<float, idx_t>> dist_with_index;
                dist_with_index.reserve(num_in_partition);
                for (idx_t idx : indices) {
                    const float* nvec = codes + idx * d;
                    float dist = fvec_norm_L2sqr(nvec, d);
                    dist_with_index.emplace_back(dist, idx);
                }

                // 按照距离排序
                std::sort(dist_with_index.begin(), dist_with_index.end());

                // 计算每个向量到分区中心的内积
                std::vector<std::pair<float, idx_t>> ip_with_index;
                ip_with_index.reserve(num_in_partition);
                for (idx_t idx : indices) {
                    const float* nvec = normalized_vectors.data() + idx * d;
                    //                    const float* nvec = codes + idx * d;
                    float dist = fvec_inner_product(nvec, c, d);
                    ip_with_index.emplace_back(dist, idx);
                }

                float sum = 0.0;
                float sum_sim = 0.0;

                for (size_t i = 0; i < num_in_partition; i++) {
                    float theta = std::acos(ip_with_index[i].first);
                    sum += theta;
                    sum_sim += std::sin(theta) * std::sin(theta);
                }
                sum_sim /= num_in_partition;

                float avg = sum / num_in_partition;
                // 计算标准差
                float std = 0.0;
                for (size_t i = 0; i < num_in_partition; i++) {
                    float tmp = std::acos(ip_with_index[i].first) - avg;
                    std += tmp * tmp;
                }

                std = std / (float)num_in_partition;
                //                float delta_phi = std::sqrt(std * std + 0.5 *
                //                sum_sim);
                float delta_phi = std::sqrt(std + 0.5 * avg * avg);
                std = std::sqrt(std);
                partition.u = avg;
                partition.std = std;
                partition.delta_phi = delta_phi;
                partition.avg_sin = std::sqrt(sum_sim);

                // 按照距离排序
                std::sort(ip_with_index.begin(), ip_with_index.end());

                float theta_min = ip_with_index.back().first;
                float theta_max = ip_with_index.front().first;

                theta_min = std::max(
                        -1.0f, std::min(1.0f, theta_min)); // 确保在[-1,1]范围内
                theta_min = std::acos(theta_min);

                theta_max = std::max(
                        -1.0f, std::min(1.0f, theta_max)); // 确保在[-1,1]范围内
                theta_max = std::acos(theta_max);

                partition.theta_min = theta_min;
                partition.theta_max = theta_max;

                // 分块：每block_size个一组
                size_t num_blocks =
                        (num_in_partition + block_size - 1) / block_size;

                partition.blocks.resize(num_blocks);
                for (size_t b = 0; b < num_blocks; b++) {
                    size_t start = b * block_size;
                    size_t end =
                            std::min((b + 1) * block_size, num_in_partition);
                    BlockData block;
                    block.block_id = b;
                    // 记录距离范围
                    block.min_dist = std::sqrt(dist_with_index[start].first);
                    block.max_dist = std::sqrt(dist_with_index[end - 1].first);

                    block.theta_max = theta_max;
                    block.theta_min = theta_min;

                    // 记录向量在原始列表中的偏移
                    block.offsets.reserve(end - start);
                    for (size_t i = start; i < end; i++) {
                        const float* nvec = normalized_vectors.data() +
                                dist_with_index[i].second * d;
                        block.offsets.push_back(dist_with_index[i].second);
                    }
                    // 记录每个向量到分区中心的距离
                    partition.blocks[b] = std::move(block);
                    std::destroy(
                            dist_with_index.begin(), dist_with_index.end());
                    std::destroy(ip_with_index.begin(), ip_with_index.end());
                }
            }
        }
        assignment.clear();
        cluster_indices.clear();
        normalized_vectors.clear();
        centroids.clear();
        std::destroy(assignment.begin(), assignment.end());
        std::destroy(normalized_vectors.begin(), normalized_vectors.end());
        std::destroy(cluster_indices.begin(), cluster_indices.end());
        std::destroy(centroids.begin(), centroids.end());
        partitions_per_cluster_[list_no] = std::move(partitions);
    }
}

// 存储每个簇的单位向量e
Eigen::MatrixXf RandomRotationR(int d) {
    // 验证输入参数有效性
    // 使用随机设备初始化随机数生成器
    std::random_device rd;
    std::mt19937 rng(1234);

    std::mt19937 gen(rng);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // 创建d×d的随机高斯矩阵
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(d, d);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            A(i, j) = dist(gen);
        }
    }

    // 进行QR分解获取正交矩阵
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
    Eigen::MatrixXf Q = qr.householderQ();

    return Q;
}

Eigen::MatrixXf Q = RandomRotationR(128);
void rotation(Eigen::MatrixXf Q, float* src, float* dest, int d) {
    Eigen::Map<Eigen::VectorXf> src_vec(src, d);
    Eigen::Map<Eigen::VectorXf> dest_vec(dest, d);
    // 执行旋转操作: dest = Q * src
    dest_vec = Q * src_vec;
}

void IndexPartitionBlockFlatSIMD::train_blocks_by_clustering_index(
        int M,
        int block_size) {
    FAISS_THROW_IF_NOT(is_trained);
    printf("start:------\n");
    partitions_per_cluster_.resize(nlist);
    //    e_vec_per_cluster_.resize(nlist);
    cluster_partition_centroids.resize(nlist);
    cluster_partition_centroids_std.resize(nlist);
    cluster_min_max.resize(nlist);
    //    printf("start resize:------\n");
    //    std::vector<float> centroids(M * d, 0);
    //////    e_vects_generator_M(centroids.data(), d, M);
    //    e_vects_generator_M(centroids.data(), d, M);
    int base_size = 512;
    float extend_size = 1.5;
#pragma omp parallel for schedule(dynamic)
    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        //        printf("list_size: %d\n", list_size);
        if (list_size == 0) {
            partitions_per_cluster_[list_no].clear();
            continue;
        }
        // 获取簇心
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        // 获取簇内所有向量
        InvertedLists::ScopedCodes scodes(invlists, list_no);
        const float* codes = (const float*)scodes.get();
        InvertedLists::ScopedIds ids(invlists, list_no);

        // 中心化并归一化向量
        std::vector<float> normalized_vectors(list_size * d);
        //        normalized_vectors.resize(list_size * d);
        //        printf("中心化并归一化向量:------\n");
        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;
            float* nvec = normalized_vectors.data() + i * d;
            // 中心化: o = (or - c)
            for (size_t j = 0; j < d; j++) {
                nvec[j] = vec[j] - centroid[j];
            }
            // 归一化: o / ||o||
            float norm = fvec_norm_L2sqr(nvec, d);
            if (norm > 0) {
                norm = std::sqrt(norm);
                for (size_t j = 0; j < d; j++) {
                    nvec[j] /= norm;
                }
            }
            //            std::vector<float> nvec_tmp(nvec, nvec + d);
            //            rotation(Q, nvec_tmp.data(), nvec, d);
        }

        //        printf("使用k-means聚类将数据分成M个组:------\n");

        // 使用k-means聚类将数据分成M个组
        int p_n = (list_size + base_size - 1) / base_size;
//        int p_n = 4;
//        printf("p_n: %d\n", p_n);
//        if (p_n > 10) {
//            printf("p_n: %d\n", p_n);
//        }
        int effective_M = std::min(static_cast<int>(list_size), p_n);
//        int effective_M = std::min(static_cast<int>(list_size), M);
        if (effective_M <= 0) {
            partitions_per_cluster_[list_no].clear();
            continue;
        }
        //        printf("初始化聚类中心:------\n");
        std::mt19937 rng(1234);
        // 使用k-means++初始化聚类中心

        std::vector<float> min_distances(
                list_size, std::numeric_limits<float>::max());
        std::vector<idx_t> centroids_indices(effective_M);
        std::vector<idx_t> assignment(list_size, -1);
        //
        //        //        // 第一个中心点随机选择
        std::vector<float> centroids(effective_M * d, 0);
        std::uniform_int_distribution<idx_t> rand(0, list_size - 1);
        idx_t first_idx = rand(rng);
        centroids_indices[0] = first_idx;
        const float* first_vec = normalized_vectors.data() + first_idx * d;
        memcpy(centroids.data(), first_vec, sizeof(float) * d);

        // 计算每个点到最近中心的距离
        auto update_min_distances = [&](int current_centroid_count) {
            for (size_t i = 0; i < list_size; i++) {
                const float* vec = normalized_vectors.data() + i * d;
                float best_dist = -1.0f;
                for (int c = 0; c <= current_centroid_count; c++) {
                    const float* centroid = centroids.data() + c * d;
                    float dist = fvec_inner_product(vec, centroid, d);
                    if (dist > best_dist) {
                        best_dist = dist;
                    }
                }
                min_distances[i] = 1.0f - best_dist; // 转换为最小化问题
            }
        };

        update_min_distances(0);

        // 选择剩余的中心点
        for (int c = 1; c < effective_M; c++) {
            // 计算距离平方和
            float total_distance = 0.0f;
            for (size_t i = 0; i < list_size; i++) {
                total_distance += min_distances[i];
            }

            // 按概率选择下一个中心点
            std::uniform_real_distribution<float> randf(0.0f, total_distance);
            float threshold = randf(rng);
            float cumulative = 0.0f;
            idx_t next_idx = 0;
            for (size_t i = 0; i < list_size; i++) {
                cumulative += min_distances[i];
                if (cumulative >= threshold) {
                    next_idx = i;
                    break;
                }
            }

            centroids_indices[c] = next_idx;
            const float* next_vec = normalized_vectors.data() + next_idx * d;
            memcpy(centroids.data() + c * d, next_vec, sizeof(float) * d);

            // 更新最小距离
            update_min_distances(c);
        }

        // 分配向量到聚类
        int max_iter = 25;
        bool changed = true;
        int iter = 0;
        for (; iter < max_iter && changed; iter++) {
            changed = false;
            // 分配步骤：将每个点分配到最近的中心
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < list_size; i++) {
                const float* nvec = normalized_vectors.data() + i * d;
                float best_dist = -10.0f;
                int best_cluster = -1;
                for (int c = 0; c < effective_M; c++) {
                    const float* cvec = centroids.data() + c * d;
                    //                    float dist = fvec_L2sqr(nvec, cvec,
                    //                    d);
                    float c_dist = std::sqrt(fvec_norm_L2sqr(cvec, d));
                    float dist = fvec_inner_product(nvec, cvec, d) / c_dist;
                    if (dist > best_dist) {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                if (assignment[i] != best_cluster) {
                    changed = true;
                    assignment[i] = best_cluster;
                }
            }

            if (!changed)
                break;

            // 更新步骤：重新计算中心
            std::vector<float> new_centroids(effective_M * d, 0);
            std::vector<int> counts(effective_M, 0);
            for (size_t i = 0; i < list_size; i++) {
                int cluster = assignment[i];
                const float* nvec = normalized_vectors.data() + i * d;
                float* new_centroid = new_centroids.data() + cluster * d;
                for (size_t j = 0; j < d; j++) {
                    new_centroid[j] += nvec[j];
                }
                counts[cluster]++;
            }

            for (int c = 0; c < effective_M; c++) {
                if (counts[c] == 0) {
                    // 随机选择一个点作为新中心
                    std::uniform_int_distribution<idx_t> rand(0, list_size - 1);
                    idx_t rand_idx = rand(rng);
                    const float* rand_vec =
                            normalized_vectors.data() + rand_idx * d;
                    memcpy(new_centroids.data() + c * d,
                           rand_vec,
                           sizeof(float) * d);
                    // 设置计数为1，避免后续归一化时除以0
                    counts[c] = 1;
                }

                float* new_centroid = new_centroids.data() + c * d;
                for (size_t j = 0; j < d; j++) {
                    new_centroid[j] /= (float)counts[c];
                }

                //                float norm = fvec_norm_L2sqr(new_centroid, d);
                //                if (norm > 0) {
                //                    norm = std::sqrt(norm);
                //                    for (size_t j = 0; j < d; j++) {
                //                        new_centroid[j] /= norm;
                //                    }
                //                }
            }

            centroids.swap(new_centroids);
        }

        for (int c = 0; c < effective_M; c++) {
            float* cvec = centroids.data() + c * d;
            float norm = fvec_norm_L2sqr(cvec, d);
            if (norm > 0) {
                norm = std::sqrt(norm);
                for (size_t j = 0; j < d; j++) {
                    cvec[j] /= norm;
                }
            }
        }
        // 利用随机向量生成中心
        //        e_vects_generator(centroids.data(), d, M);

        for (size_t i = 0; i < list_size; i++) {
            const float* nvec = normalized_vectors.data() + i * d;
            float best_dist = -10.0f;
            int best_cluster = 0;
            for (int c = 0; c < effective_M; c++) {
                const float* cvec = centroids.data() + c * d;
                //                    float dist = fvec_L2sqr(nvec, cvec,
                //                    d);
                float c_dist = std::sqrt(fvec_norm_L2sqr(cvec, d));
                float dist = fvec_inner_product(nvec, cvec, d) / c_dist;
                if (dist > best_dist) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            assignment[i] = best_cluster;
        }

        // 为每个聚类（分区）收集向量
        std::vector<std::vector<idx_t>> cluster_indices(effective_M);
        for (size_t i = 0; i < list_size; i++) {
            int cluster = assignment[i];
            cluster_indices[cluster].push_back(i);
        }

        // 为每个分区创建PartitionData
        std::vector<PartitionData> partitions;
        partitions.reserve(effective_M * 5);
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        int block_nums = 0;
        int p_num = 0;
        for (int m = 0; m < effective_M; m++) {
            // 计算每个夹角，分组
            // 获取当前分区的所有向量索引
            std::vector<idx_t>& indices = cluster_indices[m];
            size_t num_in_partition = indices.size();
            float* c = centroids.data() + m * d;
            float c_dist = std::sqrt(fvec_norm_L2sqr(c, d));
            // 计算每个向量到分区中心的内积
            std::vector<std::pair<float, idx_t>> ips;
            ips.reserve(num_in_partition);
            for (idx_t idx : indices) {
                const float* nvec_norm = normalized_vectors.data() + idx * d;
                //                    const float* nvec = codes + idx * d;
                float ip_e_o = fvec_inner_product(nvec_norm, c, d) / c_dist;
                ip_e_o = std::acos(ip_e_o);
                ips.emplace_back(ip_e_o, idx);
            }

            std::sort(
                    ips.begin(),
                    ips.end());
            int v_num = ips.size();
            for (size_t s_index = 0; s_index < v_num;) {
                float start_theta = ips[s_index].first / M_PI * 180;
                float end_theta = ips[v_num - 1].first / M_PI * 180;
                float interval = 180;

                int start_bin = (int)(start_theta / interval);
                int end_index = s_index;
                bool flag = false;
                int count = 0;
                for (; end_index < v_num; end_index++) {
                    float theta_j = ips[end_index].first / M_PI * 180;
                    int j_bin = (int)(theta_j / interval);
                    if (j_bin != start_bin) {
                        flag = true;
                    }

                    if (count >= block_size && flag) {
                        break;
                    }
                    count++;
                }

                size_t start = s_index;
                size_t end = end_index;
                num_in_partition = end - start;

                partitions.emplace_back();
                PartitionData& partition = partitions.back();
                partition.centroid.resize(d);

                memcpy(partition.centroid.data(), c, sizeof(float) * d);

                //            if (cluster_indices[m].empty()) {
                //                continue;
                //            }

                // 计算每个向量到分区中心的距离
                if (num_in_partition > 0) {
                    std::vector<std::pair<float, std::pair<idx_t, float>>>
                            dist_with_index;
                    dist_with_index.reserve(num_in_partition);
                    // 计算每个向量到分区中心的内积
                    std::vector<std::pair<float, idx_t>> ip_with_index;
                    ip_with_index.reserve(num_in_partition);

                    double dist_o_e = 0.0;
                    std::vector<float> sum_var(d, 0.0);
                    std::vector<float> sum_mean(d, 0.0);
                    for (size_t s_i = start; s_i < end; s_i++) {
                        idx_t idx = ips[s_i].second;
                        const float* nvec = codes + idx * d;
                        float dist = fvec_L2sqr(nvec, centroid.data(), d);
                        const float* nvec_norm =
                                normalized_vectors.data() + idx * d;
                        //                    const float* nvec = codes + idx *
                        //                    d;
                        float ip_e_o =
                                fvec_inner_product(nvec_norm, c, d) / c_dist;
                        dist_o_e += std::sqrt(2 - 2 * ip_e_o);
                        dist_with_index.emplace_back(
                                dist, std::make_pair(idx, ip_e_o));
                        ip_with_index.emplace_back(ip_e_o, idx);
                        for (size_t j = 0; j < d; j++) {
                            float diff = nvec_norm[j] - c[j];
                            sum_var[j] += diff * diff;
                            sum_mean[j] += diff;
                        }
                    }

                    //                for (idx_t idx : indices) {
                    //                    const float* nvec = codes + idx * d;
                    //                    float dist = fvec_L2sqr(nvec,
                    //                    centroid.data(), d); const float*
                    //                    nvec_norm = normalized_vectors.data()
                    //                    + idx * d;
                    //                    //                    const float*
                    //                    nvec = codes + idx * d; float ip_e_o =
                    //                    fvec_inner_product(nvec_norm, c, d) /
                    //                    c_dist; dist_o_e += std::sqrt(2 - 2 *
                    //                    ip_e_o);
                    //                    dist_with_index.emplace_back(dist,
                    //                    std::make_pair(idx, ip_e_o));
                    //                    ip_with_index.emplace_back(ip_e_o,
                    //                    idx); for (size_t j = 0; j < d; j++) {
                    //                        float diff = nvec_norm[j] - c[j];
                    //                        sum_var[j] += diff * diff;
                    //                        sum_mean[j] += diff;
                    //                    }
                    //                }

                    std::vector<float> std_dev(d);
                    float std_mean = 0.0;
                    for (int kk = 0; kk < d; ++kk) {
                        //                    float mean = sum_mean[kk] /
                        //                    partition.num; float variance =
                        //                    (sum_var[kk] / partition.num) -
                        //                    (mean * mean);
                        float variance = (sum_var[kk] / num_in_partition);
                        if (variance < 0.0) {
                            variance = -variance;
                        }
                        variance = std::sqrt(variance);
                        std_mean += variance;
                        std_dev[kk] = variance;
                    }
                    std_mean /= (float)d;
                    partition.std_dev.resize(d);
                    memcpy(partition.std_dev.data(),
                           std_dev.data(),
                           sizeof(float) * d);
                    // 按照距离排序
                    std::sort(dist_with_index.begin(), dist_with_index.end());

                    float sum = 0.0;
                    float sum_sim = 0.0;

                    float exp_cos = 0.0;
                    float var_cos = 0.0;
                    for (size_t i = 0; i < num_in_partition; i++) {
                        exp_cos += ip_with_index[i].first;
                        float theta = std::acos(ip_with_index[i].first);
                        sum += theta;
                        sum_sim += std::sin(theta) * std::sin(theta);
                    }

                    sum_sim /= num_in_partition;
                    exp_cos /= num_in_partition;

                    for (size_t i = 0; i < num_in_partition; i++) {
                        float delta = ip_with_index[i].first - exp_cos;
                        var_cos += delta * delta;
                    }

                    var_cos /= num_in_partition;

                    partition.exp_cos_theta = exp_cos;
                    partition.var_cos_theta = var_cos;
                    partition.o_e_dist_avg = dist_o_e / num_in_partition;
                    float avg = sum / num_in_partition;
                    // 计算标准差
                    float std = 0.0;
                    for (size_t i = 0; i < num_in_partition; i++) {
                        float tmp = std::acos(ip_with_index[i].first) - avg;
                        std += tmp * tmp;
                    }

                    std = std / (float)num_in_partition;
                    //                float delta_phi = std::sqrt(std * std +
                    //                0.5 * sum_sim);
                    float delta_phi = std::sqrt(std + 0.5 * avg * avg);
                    std = std::sqrt(std);
                    partition.u = avg;
                    //                partition.std = std;
                    partition.delta_phi = delta_phi;
                    partition.avg_sin = std::sqrt(sum_sim);
                    partition.num = num_in_partition;

                    partition.std = std_mean;
                    // 按照距离排序
                    std::sort(ip_with_index.begin(), ip_with_index.end());

                    float theta_min = ip_with_index.back().first;
                    float theta_max = ip_with_index.front().first;

                    theta_min = std::max(
                            -1.0f,
                            std::min(1.0f, theta_min)); // 确保在[-1,1]范围内
                    theta_min = std::acos(theta_min);

                    theta_max = std::max(
                            -1.0f,
                            std::min(1.0f, theta_max)); // 确保在[-1,1]范围内
                    theta_max = std::acos(theta_max);

                    partition.theta_min = theta_min;
                    partition.theta_max = theta_max;

                    float u1 =
                            (partition.theta_max - partition.theta_min) / 4.0f;
                    float u2 =
                            (partition.theta_max + partition.theta_min) / 2.0f;
                    float delta = std::sqrt((u1 * u1 + u2 * u2) / 2.0f);
                    partition.gauss_delta = delta;

                    partition.min_dis = std::sqrt(dist_with_index[0].first);
                    //                partition.max_dis
                    //                =std::sqrt(dist_with_index.end()[0].first);
                    partition.max_dis = std::sqrt(
                            dist_with_index[num_in_partition - 1].first);
                    min = std::min(min, partition.min_dis);
                    max = std::max(max, partition.max_dis);

                    // 分块：每block_size个一组
                    size_t num_blocks =
                            (num_in_partition + block_size - 1) / block_size;

                    partition.blocks.resize(num_blocks);
                    block_nums += num_blocks;
                    for (size_t b = 0; b < num_blocks; b++) {
                        size_t start = b * block_size;
                        size_t end = std::min(
                                (b + 1) * block_size, num_in_partition);

                        std::vector<float> sum_var_block(d, 0.0);
                        for (size_t offset = start; offset < end; offset++) {
                            size_t idx = dist_with_index[offset].second.first;
                            const float* nvec_norm =
                                    normalized_vectors.data() + idx * d;
                            //                    const float* nvec = codes +
                            //                    idx * d;
                            for (size_t j = 0; j < d; j++) {
                                float diff = nvec_norm[j] - c[j];
                                sum_var_block[j] += diff * diff;
                            }
                        }
                        size_t num_vecs_in_block = end - start;
                        float std_mean_block = 0.0;
                        for (int kk = 0; kk < d; ++kk) {
                            //                    float mean = sum_mean[kk] /
                            //                    partition.num; float variance
                            //                    = (sum_var[kk] /
                            //                    partition.num) - (mean *
                            //                    mean);
                            float variance =
                                    (sum_var_block[kk] / num_vecs_in_block);
                            if (variance < 0.0) {
                                variance = -variance;
                            }
                            variance = std::sqrt(variance);
                            std_mean_block += variance;
                        }
                        std_mean_block /= (float)d;
                        BlockData block;
                        block.block_id = b;
                        block.std_block = std_mean_block;
                        // 记录距离范围
                        block.min_dist =
                                std::sqrt(dist_with_index[start].first);
                        block.max_dist =
                                std::sqrt(dist_with_index[end - 1].first);

                        block.theta_max = 1.0;
                        block.theta_min = -1.0;
                        // 记录theta范围
                        for (auto it = dist_with_index.begin() + (int)start;
                             it < dist_with_index.begin() + (int)end;
                             it++) {
                            block.theta_min = std::max(
                                    block.theta_min, it->second.second);
                            block.theta_max = std::min(
                                    block.theta_max, it->second.second);
                        }

                        block.theta_min = std::acos(block.theta_min);
                        block.theta_max = std::acos(block.theta_max);

                        // 记录向量在原始列表中的偏移
                        block.offsets.reserve(end - start);

                        // 为block分配对齐的内存并存储向量数据

                        // 16个向量一组，存储16个向量，SIMD
                        size_t v_id = start;
                        int bbs_16_size = 0;
                        if (num_vecs_in_block / 16 > 0) {
                            const size_t full_blocks = num_vecs_in_block / 16;
                            const size_t alloc_size =
                                    full_blocks * 16 * d * sizeof(float);
                            // 安全分配内存 (处理 size=0 情况)
                            if (alloc_size > 0) {
                                block.bbs_16 = static_cast<float*>(
                                        aligned_alloc(64, alloc_size));
                                if (!block.bbs_16) {
                                    throw std::bad_alloc();
                                }
                            } else {
                                block.bbs_16 = nullptr;
                            }
                            //                        block.bbs_16 = (float*)
                            //                        aligned_alloc(64, ((int)
                            //                        num_vecs_in_block / 16) *
                            //                        16 * d * sizeof(float));
                            for (; v_id + 15 < end; v_id += 16) {
                                float* bbs16 =
                                        block.bbs_16 + bbs_16_size * 16 * d;
                                for (int dim = 0; dim < d; dim++) {
                                    for (int i = 0; i < 16; i++) {
                                        idx_t global_idx =
                                                dist_with_index[v_id + i]
                                                        .second.first;
                                        const float* vec =
                                                codes + global_idx * d;
                                        bbs16[dim * 16 + i] = vec[dim];
                                    }
                                }

                                for (int i = 0; i < 16; i++) {
                                    idx_t global_idx = dist_with_index[v_id + i]
                                                               .second.first;
                                    block.offsets.push_back(global_idx);
                                }
                                bbs_16_size++;
                            }
                        }

                        int bbs_8_size = 0;
                        if ((num_vecs_in_block - bbs_16_size * 16) / 8 > 0) {
                            const size_t full_blocks =
                                    (num_vecs_in_block - bbs_16_size * 16) / 8;
                            const size_t alloc_size =
                                    full_blocks * 8 * d * sizeof(float);
                            // 安全分配内存 (处理 size=0 情况)
                            if (alloc_size > 0) {
                                block.bbs_8 = static_cast<float*>(
                                        aligned_alloc(64, alloc_size));
                                if (!block.bbs_8) {
                                    throw std::bad_alloc();
                                }
                            } else {
                                block.bbs_8 = nullptr;
                            }

                            //                        block.bbs_8 = (float*)
                            //                        aligned_alloc(32, ((int)
                            //                        (num_vecs_in_block -
                            //                        bbs_16_size * 16) / 8) * 8
                            //                        * d
                            //                        * sizeof(float));
                            for (; v_id + 7 < end; v_id += 8) {
                                float* bbs8 = block.bbs_8 + bbs_8_size * 8 * d;
                                for (int dim = 0; dim < d; dim++) {
                                    for (int i = 0; i < 8; i++) {
                                        idx_t global_idx =
                                                dist_with_index[v_id + i]
                                                        .second.first;
                                        const float* vec =
                                                codes + global_idx * d;
                                        bbs8[dim * 8 + i] = vec[dim];
                                    }
                                }

                                for (int i = 0; i < 8; i++) {
                                    idx_t global_idx = dist_with_index[v_id + i]
                                                               .second.first;
                                    block.offsets.push_back(global_idx);
                                }
                                bbs_8_size++;
                            }
                        }

                        size_t remaining = end - v_id;
                        if (remaining > 0) {
                            const size_t alloc_size =
                                    remaining * d * sizeof(float);
                            if (alloc_size > 0) {
                                block.vectors = static_cast<float*>(
                                        aligned_alloc(64, alloc_size));
                                if (!block.vectors) {
                                    throw std::bad_alloc();
                                }
                            } else {
                                block.vectors = nullptr;
                            }

                            for (size_t i = v_id; i < end; i++) {
                                idx_t global_idx =
                                        dist_with_index[i].second.first;
                                const float* vec = codes + global_idx * d;
                                memcpy(block.vectors + (i - v_id) * d,
                                       vec,
                                       d * sizeof(float));
                                block.offsets.push_back(
                                        dist_with_index[i].second.first);
                            }
                        }

                        //                    size_t v_id = start;
                        //                    size_t remaining =
                        //                    num_vecs_in_block; const size_t
                        //                    alloc_size = remaining * d *
                        //                    sizeof(float); if (alloc_size > 0)
                        //                    {
                        //                        block.vectors =
                        //                        static_cast<float*>(
                        //                                aligned_alloc(64,
                        //                                alloc_size));
                        //                        if (!block.vectors) {
                        //                            throw std::bad_alloc();
                        //                        }
                        //                    } else {
                        //                        block.vectors = nullptr;
                        //                    }
                        //
                        //                    for (size_t i = v_id; i < end;
                        //                    i++) {
                        //                        idx_t global_idx =
                        //                        dist_with_index[i].second.first;
                        //                        const float* vec = codes +
                        //                        global_idx * d;
                        //                        memcpy(block.vectors + (i -
                        //                        v_id) * d,
                        //                               vec,
                        //                               d * sizeof(float));
                        //                        block.offsets.push_back(dist_with_index[i].second.first);
                        //                    }

                        // 记录每个向量到分区中心的距离
                        partition.blocks[b] = std::move(block);
                        std::destroy(
                                dist_with_index.begin(), dist_with_index.end());
                        std::destroy(
                                ip_with_index.begin(), ip_with_index.end());
                    }
                }
                s_index = end_index;
            }
        }

        // 打包分组的中心
        if (partitions.size() / 16 > 0) {
            const size_t full_blocks = partitions.size() / 16;
            const size_t alloc_size = full_blocks * 16 * d * sizeof(float);
            float* p_centroids = cluster_partition_centroids[list_no];
            float* p_centroids_std = cluster_partition_centroids_std[list_no];
            // 安全分配内存 (处理 size=0 情况)
            if (alloc_size > 0) {
                p_centroids =
                        static_cast<float*>(aligned_alloc(64, alloc_size));
                p_centroids_std =
                        static_cast<float*>(aligned_alloc(64, alloc_size));
                if (!p_centroids) {
                    throw std::bad_alloc();
                }
            } else {
                p_centroids = nullptr;
                p_centroids_std = nullptr;
            }
            //                        block.bbs_16 = (float*) aligned_alloc(64,
            //                        ((int) num_vecs_in_block / 16) * 16 * d *
            //                        sizeof(float));
            size_t v_id = 0;
            size_t end = partitions.size();
            int bbs_16_size = 0;
            for (; v_id + 15 < end; v_id += 16) {
                float* bbs16 = p_centroids + bbs_16_size * 16 * d;
                float* bbs16_std = p_centroids_std + bbs_16_size * 16 * d;
                for (int dim = 0; dim < d; dim++) {
                    for (int i = 0; i < 16; i++) {
                        bbs16[dim * 16 + i] =
                                partitions[v_id + i].centroid[dim];
                        bbs16_std[dim * 16 + i] =
                                partitions[v_id + i].std_dev[dim];
                    }
                }
                bbs_16_size++;
            }
            cluster_partition_centroids[list_no] = p_centroids;
            cluster_partition_centroids_std[list_no] = p_centroids_std;
        }
        cluster_min_max[list_no].first = {min, max};
        cluster_min_max[list_no].second = block_nums;
        assignment.clear();
        cluster_indices.clear();
        normalized_vectors.clear();
        centroids.clear();
        std::destroy(assignment.begin(), assignment.end());
        std::destroy(normalized_vectors.begin(), normalized_vectors.end());
        std::destroy(cluster_indices.begin(), cluster_indices.end());
        std::destroy(centroids.begin(), centroids.end());
        partitions_per_cluster_[list_no] = std::move(partitions);
    }
}

void IndexPartitionBlockFlatSIMD::train_blocks_by_distance(int block_size) {
    FAISS_THROW_IF_NOT(is_trained);
    printf("start:------\n");
    partitions_per_cluster_.resize(nlist);
    cluster_min_max.resize(nlist);
#pragma omp parallel for schedule(dynamic)
    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        //        printf("list_size: %d\n", list_size);
        if (list_size == 0) {
            partitions_per_cluster_[list_no].clear();
            continue;
        }
        // 获取簇心
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());
        // 获取簇内所有向量
        InvertedLists::ScopedCodes scodes(invlists, list_no);
        const float* codes = (const float*)scodes.get();
        InvertedLists::ScopedIds ids(invlists, list_no);

        // 中心化并归一化向量
        std::vector<float> normalized_vectors(list_size * d);
        //        normalized_vectors.resize(list_size * d);
        //        printf("中心化并归一化向量:------\n");
        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;
            float* nvec = normalized_vectors.data() + i * d;
            // 中心化: o = (or - c)
            for (size_t j = 0; j < d; j++) {
                nvec[j] = vec[j] - centroid[j];
            }
            // 归一化: o / ||o||
            float norm = fvec_norm_L2sqr(nvec, d);
            if (norm > 0) {
                norm = std::sqrt(norm);
                for (size_t j = 0; j < d; j++) {
                    nvec[j] /= norm;
                }
            }
        }

        std::vector<std::pair<float, std::pair<size_t, const float*>>>
                min_distances;
        //        min_distances.resize(list_size);
        for (size_t idx = 0; idx < list_size; idx++) {
            const float* nvec = codes + idx * d;
            float dist = std::sqrt(fvec_L2sqr(nvec, centroid.data(), d));
            min_distances.emplace_back(dist, std::make_pair(idx, nvec));
        }
        std::sort(min_distances.begin(), min_distances.end());

        int block_num = (list_size + block_size - 1) / block_size;
        // 为每个分区创建PartitionData
        std::vector<PartitionData> partitions(1);
        PartitionData& partition = partitions[0];
        partition.blocks.resize(block_num);
        size_t num_in_partition = list_size;
        for (size_t b = 0; b < block_num; b++) {
            size_t start = b * block_size;
            size_t end = std::min((b + 1) * block_size, num_in_partition);
            BlockData block;
            block.block_id = b;
            // 记录距离范围
            block.min_dist = min_distances[start].first;
            block.max_dist = min_distances[end - 1].first;

            // 记录向量在原始列表中的偏移
            block.offsets.reserve(end - start);

            // 为block分配对齐的内存并存储向量数据
            size_t num_vecs_in_block = end - start;

            size_t v_id = start;
            int bbs_16_size = 0;
            if (num_vecs_in_block / 16 > 0) {
                const size_t full_blocks = num_vecs_in_block / 16;
                const size_t alloc_size = full_blocks * 16 * d * sizeof(float);
                // 安全分配内存 (处理 size=0 情况)
                if (alloc_size > 0) {
                    block.bbs_16 =
                            static_cast<float*>(aligned_alloc(64, alloc_size));
                    if (!block.bbs_16) {
                        throw std::bad_alloc();
                    }
                } else {
                    block.bbs_16 = nullptr;
                }
                //                        block.bbs_16 = (float*)
                //                        aligned_alloc(64, ((int)
                //                        num_vecs_in_block / 16) * 16 *
                //                        d * sizeof(float));
                for (; v_id + 15 < end; v_id += 16) {
                    float* bbs16 = block.bbs_16 + bbs_16_size * 16 * d;
                    for (int dim = 0; dim < d; dim++) {
                        for (int i = 0; i < 16; i++) {
                            const float* vec =
                                    min_distances[v_id + i].second.second;
                            bbs16[dim * 16 + i] = vec[dim];
                        }
                    }

                    for (int i = 0; i < 16; i++) {
                        idx_t global_idx = min_distances[v_id + i].second.first;
                        block.offsets.push_back(global_idx);
                    }
                    bbs_16_size++;
                }
            }

            int bbs_8_size = 0;
            if ((num_vecs_in_block - bbs_16_size * 16) / 8 > 0) {
                const size_t full_blocks =
                        (num_vecs_in_block - bbs_16_size * 16) / 8;
                const size_t alloc_size = full_blocks * 8 * d * sizeof(float);
                // 安全分配内存 (处理 size=0 情况)
                if (alloc_size > 0) {
                    block.bbs_8 =
                            static_cast<float*>(aligned_alloc(64, alloc_size));
                    if (!block.bbs_8) {
                        throw std::bad_alloc();
                    }
                } else {
                    block.bbs_8 = nullptr;
                }

                //                        block.bbs_8 = (float*)
                //                        aligned_alloc(32, ((int)
                //                        (num_vecs_in_block -
                //                        bbs_16_size * 16) / 8) * 8 * d
                //                        * sizeof(float));
                for (; v_id + 7 < end; v_id += 8) {
                    float* bbs8 = block.bbs_8 + bbs_8_size * 8 * d;
                    for (int dim = 0; dim < d; dim++) {
                        for (int i = 0; i < 8; i++) {
                            const float* vec =
                                    min_distances[v_id + i].second.second;
                            bbs8[dim * 8 + i] = vec[dim];
                        }
                    }

                    for (int i = 0; i < 8; i++) {
                        idx_t global_idx = min_distances[v_id + i].second.first;
                        block.offsets.push_back(global_idx);
                    }
                    bbs_8_size++;
                }
            }

            size_t remaining = end - v_id;
            if (remaining > 0) {
                const size_t alloc_size = remaining * d * sizeof(float);
                if (alloc_size > 0) {
                    block.vectors =
                            static_cast<float*>(aligned_alloc(64, alloc_size));
                    if (!block.vectors) {
                        throw std::bad_alloc();
                    }
                } else {
                    block.vectors = nullptr;
                }

                for (size_t i = v_id; i < end; i++) {
                    idx_t global_idx = min_distances[i].second.first;
                    const float* vec = codes + global_idx * d;
                    memcpy(block.vectors + (i - v_id) * d,
                           vec,
                           d * sizeof(float));
                    block.offsets.push_back(global_idx);
                }
            }
            // 记录每个向量到分区中心的距离
            partition.blocks[b] = std::move(block);
        }

        partitions_per_cluster_[list_no] = std::move(partitions);
        cluster_min_max[list_no].first = {
                min_distances[0].first, min_distances[list_size - 1].first};
        cluster_min_max[list_no].second = block_num;
    }
}

void IndexPartitionBlockFlatSIMD::train_blocks_by_clustering(
        int M,
        int block_size) {
    FAISS_THROW_IF_NOT(is_trained);
    printf("start:------\n");
    partitions_per_cluster_.resize(nlist);
    //    e_vec_per_cluster_.resize(nlist);
    cluster_partition_centroids.resize(nlist);
    cluster_min_max.resize(nlist);
    //    printf("start resize:------\n");
//    std::vector<float> centroids(M * d, 0);
//////    e_vects_generator_M(centroids.data(), d, M);
//    e_vects_generator_M(centroids.data(), d, M);
#pragma omp parallel for schedule(dynamic)
    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        //        printf("list_size: %d\n", list_size);
        if (list_size == 0) {
            partitions_per_cluster_[list_no].clear();
            continue;
        }
        // 获取簇心
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        // 获取簇内所有向量
        InvertedLists::ScopedCodes scodes(invlists, list_no);
        const float* codes = (const float*)scodes.get();
        InvertedLists::ScopedIds ids(invlists, list_no);

        // 中心化并归一化向量
        std::vector<float> normalized_vectors(list_size * d);
        //        normalized_vectors.resize(list_size * d);
        //        printf("中心化并归一化向量:------\n");
        for (size_t i = 0; i < list_size; i++) {
            const float* vec = codes + i * d;
            float* nvec = normalized_vectors.data() + i * d;
            // 中心化: o = (or - c)
            for (size_t j = 0; j < d; j++) {
                nvec[j] = vec[j] - centroid[j];
            }
            // 归一化: o / ||o||
            float norm = fvec_norm_L2sqr(nvec, d);
            if (norm > 0) {
                norm = std::sqrt(norm);
                for (size_t j = 0; j < d; j++) {
                    nvec[j] /= norm;
                }
            }
            //            std::vector<float> nvec_tmp(nvec, nvec + d);
            //            rotation(Q, nvec_tmp.data(), nvec, d);
        }

        //        printf("使用k-means聚类将数据分成M个组:------\n");

        // 使用k-means聚类将数据分成M个组
        int effective_M = std::min(static_cast<int>(list_size), M);
        if (effective_M <= 0) {
            partitions_per_cluster_[list_no].clear();
            continue;
        }
        //        printf("初始化聚类中心:------\n");
        std::mt19937 rng(1234);
        // 使用k-means++初始化聚类中心

        std::vector<float> min_distances(
                list_size, std::numeric_limits<float>::max());
        std::vector<idx_t> centroids_indices(effective_M);
        std::vector<idx_t> assignment(list_size, -1);
        //
        //        //        // 第一个中心点随机选择
        std::vector<float> centroids(effective_M * d, 0);
        std::uniform_int_distribution<idx_t> rand(0, list_size - 1);
        idx_t first_idx = rand(rng);
        centroids_indices[0] = first_idx;
        const float* first_vec = normalized_vectors.data() + first_idx * d;
        memcpy(centroids.data(), first_vec, sizeof(float) * d);

        // 计算每个点到最近中心的距离
        auto update_min_distances = [&](int current_centroid_count) {
            for (size_t i = 0; i < list_size; i++) {
                const float* vec = normalized_vectors.data() + i * d;
                float best_dist = -1.0f;
                for (int c = 0; c <= current_centroid_count; c++) {
                    const float* centroid = centroids.data() + c * d;
                    float dist = fvec_inner_product(vec, centroid, d);
                    if (dist > best_dist) {
                        best_dist = dist;
                    }
                }
                min_distances[i] = 1.0f - best_dist; // 转换为最小化问题
            }
        };

        update_min_distances(0);

        // 选择剩余的中心点
        for (int c = 1; c < effective_M; c++) {
            // 计算距离平方和
            float total_distance = 0.0f;
            for (size_t i = 0; i < list_size; i++) {
                total_distance += min_distances[i];
            }

            // 按概率选择下一个中心点
            std::uniform_real_distribution<float> randf(0.0f, total_distance);
            float threshold = randf(rng);
            float cumulative = 0.0f;
            idx_t next_idx = 0;
            for (size_t i = 0; i < list_size; i++) {
                cumulative += min_distances[i];
                if (cumulative >= threshold) {
                    next_idx = i;
                    break;
                }
            }

            centroids_indices[c] = next_idx;
            const float* next_vec = normalized_vectors.data() + next_idx * d;
            memcpy(centroids.data() + c * d, next_vec, sizeof(float) * d);

            // 更新最小距离
            update_min_distances(c);
        }

        // 分配向量到聚类
        int max_iter = 25;
        bool changed = true;
        int iter = 0;
        for (; iter < max_iter && changed; iter++) {
            changed = false;
            // 分配步骤：将每个点分配到最近的中心
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < list_size; i++) {
                const float* nvec = normalized_vectors.data() + i * d;
                float best_dist = -10.0f;
                int best_cluster = -1;
                for (int c = 0; c < effective_M; c++) {
                    const float* cvec = centroids.data() + c * d;
                    //                    float dist = fvec_L2sqr(nvec, cvec,
                    //                    d);
                    float dist = fvec_inner_product(nvec, cvec, d);
                    if (dist > best_dist) {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                if (assignment[i] != best_cluster) {
                    changed = true;
                    assignment[i] = best_cluster;
                }
            }

            if (!changed)
                break;

            // 更新步骤：重新计算中心
            std::vector<float> new_centroids(effective_M * d, 0);
            std::vector<int> counts(effective_M, 0);
            for (size_t i = 0; i < list_size; i++) {
                int cluster = assignment[i];
                const float* nvec = normalized_vectors.data() + i * d;
                float* new_centroid = new_centroids.data() + cluster * d;
                for (size_t j = 0; j < d; j++) {
                    new_centroid[j] += nvec[j];
                }
                counts[cluster]++;
            }

            for (int c = 0; c < effective_M; c++) {
                if (counts[c] == 0) {
                    // 随机选择一个点作为新中心
                    std::uniform_int_distribution<idx_t> rand(0, list_size - 1);
                    idx_t rand_idx = rand(rng);
                    const float* rand_vec =
                            normalized_vectors.data() + rand_idx * d;
                    memcpy(new_centroids.data() + c * d,
                           rand_vec,
                           sizeof(float) * d);
                    // 设置计数为1，避免后续归一化时除以0
                    counts[c] = 1;
                }

                float* new_centroid = new_centroids.data() + c * d;
                for (size_t j = 0; j < d; j++) {
                    new_centroid[j] /= (float)counts[c];
                }

                float norm = fvec_norm_L2sqr(new_centroid, d);
                if (norm > 0) {
                    norm = std::sqrt(norm);
                    for (size_t j = 0; j < d; j++) {
                        new_centroid[j] /= norm;
                    }
                }
            }

            centroids.swap(new_centroids);
        }

        // 利用随机向量生成中心
        //        e_vects_generator(centroids.data(), d, M);

        for (size_t i = 0; i < list_size; i++) {
            const float* nvec = normalized_vectors.data() + i * d;
            float best_dist = -10.0f;
            int best_cluster = 0;
            for (int c = 0; c < effective_M; c++) {
                const float* cvec = centroids.data() + c * d;
                //                    float dist = fvec_L2sqr(nvec, cvec,
                //                    d);
                float dist = fvec_inner_product(nvec, cvec, d);
                if (dist > best_dist) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            assignment[i] = best_cluster;
        }

        // 为每个聚类（分区）收集向量
        std::vector<std::vector<idx_t>> cluster_indices(effective_M);
        for (size_t i = 0; i < list_size; i++) {
            int cluster = assignment[i];
            cluster_indices[cluster].push_back(i);
        }

        // 为每个分区创建PartitionData
        std::vector<PartitionData> partitions(effective_M);
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        int block_nums = 0;
        for (int m = 0; m < effective_M; m++) {
            PartitionData& partition = partitions[m];
            partition.centroid.resize(d);
            float* c = centroids.data() + m * d;
            memcpy(partition.centroid.data(), c, sizeof(float) * d);

            //            if (cluster_indices[m].empty()) {
            //                continue;
            //            }

            // 获取当前分区的所有向量索引
            std::vector<idx_t>& indices = cluster_indices[m];
            size_t num_in_partition = indices.size();

            // 计算每个向量到分区中心的距离
            if (num_in_partition > 0) {
                std::vector<std::pair<float, idx_t>> dist_with_index;
                dist_with_index.reserve(num_in_partition);
                for (idx_t idx : indices) {
                    const float* nvec = codes + idx * d;
                    float dist = fvec_L2sqr(nvec, centroid.data(), d);
                    dist_with_index.emplace_back(dist, idx);
                }

                // 按照距离排序
                std::sort(dist_with_index.begin(), dist_with_index.end());

                // 计算每个向量到分区中心的内积
                std::vector<std::pair<float, idx_t>> ip_with_index;
                ip_with_index.reserve(num_in_partition);
                double dist_o_e = 0.0;
                for (idx_t idx : indices) {
                    const float* nvec = normalized_vectors.data() + idx * d;
                    //                    const float* nvec = codes + idx * d;
                    float dist = fvec_inner_product(nvec, c, d);
                    dist_o_e += std::sqrt(2 - 2 * dist);
                    ip_with_index.emplace_back(dist, idx);
                }
                float sum = 0.0;
                float sum_sim = 0.0;

                float exp_cos = 0.0;
                float var_cos = 0.0;
                for (size_t i = 0; i < num_in_partition; i++) {
                    exp_cos += ip_with_index[i].first;
                    float theta = std::acos(ip_with_index[i].first);
                    sum += theta;
                    sum_sim += std::sin(theta) * std::sin(theta);
                }

                sum_sim /= num_in_partition;
                exp_cos /= num_in_partition;

                for (size_t i = 0; i < num_in_partition; i++) {
                    float delta = ip_with_index[i].first - exp_cos;
                    var_cos += delta * delta;
                }

                var_cos /= num_in_partition;

                partition.exp_cos_theta = exp_cos;
                partition.var_cos_theta = var_cos;
                partition.o_e_dist_avg = dist_o_e / num_in_partition;
                float avg = sum / num_in_partition;
                // 计算标准差
                float std = 0.0;
                for (size_t i = 0; i < num_in_partition; i++) {
                    float tmp = std::acos(ip_with_index[i].first) - avg;
                    std += tmp * tmp;
                }

                std = std / (float)num_in_partition;
                //                float delta_phi = std::sqrt(std * std + 0.5 *
                //                sum_sim);
                float delta_phi = std::sqrt(std + 0.5 * avg * avg);
                std = std::sqrt(std);
                partition.u = avg;
                partition.std = std;
                partition.delta_phi = delta_phi;
                partition.avg_sin = std::sqrt(sum_sim);
                partition.num = num_in_partition;
                // 按照距离排序
                std::sort(ip_with_index.begin(), ip_with_index.end());

                float theta_min = ip_with_index.back().first;
                float theta_max = ip_with_index.front().first;

                theta_min = std::max(
                        -1.0f, std::min(1.0f, theta_min)); // 确保在[-1,1]范围内
                theta_min = std::acos(theta_min);

                theta_max = std::max(
                        -1.0f, std::min(1.0f, theta_max)); // 确保在[-1,1]范围内
                theta_max = std::acos(theta_max);

                partition.theta_min = theta_min;
                partition.theta_max = theta_max;

                float u1 = (partition.theta_max - partition.theta_min) / 4.0f;
                float u2 = (partition.theta_max + partition.theta_min) / 2.0f;
                float delta = std::sqrt((u1 * u1 + u2 * u2) / 2.0f);
                partition.gauss_delta = delta;

                partition.min_dis = std::sqrt(dist_with_index[0].first);
                //                partition.max_dis
                //                =std::sqrt(dist_with_index.end()[0].first);
                partition.max_dis =
                        std::sqrt(dist_with_index[num_in_partition - 1].first);
                min = std::min(min, partition.min_dis);
                max = std::max(max, partition.max_dis);

                // 分块：每block_size个一组
                size_t num_blocks =
                        (num_in_partition + block_size - 1) / block_size;

                partition.blocks.resize(num_blocks);
                block_nums += num_blocks;
                for (size_t b = 0; b < num_blocks; b++) {
                    size_t start = b * block_size;
                    size_t end =
                            std::min((b + 1) * block_size, num_in_partition);
                    BlockData block;
                    block.block_id = b;
                    // 记录距离范围
                    block.min_dist = std::sqrt(dist_with_index[start].first);
                    block.max_dist = std::sqrt(dist_with_index[end - 1].first);

                    block.theta_max = theta_max;
                    block.theta_min = theta_min;
                    // 记录向量在原始列表中的偏移
                    block.offsets.reserve(end - start);

                    // 为block分配对齐的内存并存储向量数据
                    size_t num_vecs_in_block = end - start;

                    size_t v_id = start;
                    int bbs_16_size = 0;
                    if (num_vecs_in_block / 16 > 0) {
                        const size_t full_blocks = num_vecs_in_block / 16;
                        const size_t alloc_size =
                                full_blocks * 16 * d * sizeof(float);
                        // 安全分配内存 (处理 size=0 情况)
                        if (alloc_size > 0) {
                            block.bbs_16 = static_cast<float*>(
                                    aligned_alloc(64, alloc_size));
                            if (!block.bbs_16) {
                                throw std::bad_alloc();
                            }
                        } else {
                            block.bbs_16 = nullptr;
                        }
                        //                        block.bbs_16 = (float*)
                        //                        aligned_alloc(64, ((int)
                        //                        num_vecs_in_block / 16) * 16 *
                        //                        d * sizeof(float));
                        for (; v_id + 15 < end; v_id += 16) {
                            float* bbs16 = block.bbs_16 + bbs_16_size * 16 * d;
                            for (int dim = 0; dim < d; dim++) {
                                for (int i = 0; i < 16; i++) {
                                    idx_t global_idx =
                                            dist_with_index[v_id + i].second;
                                    const float* vec = codes + global_idx * d;
                                    bbs16[dim * 16 + i] = vec[dim];
                                }
                            }

                            for (int i = 0; i < 16; i++) {
                                idx_t global_idx =
                                        dist_with_index[v_id + i].second;
                                block.offsets.push_back(global_idx);
                            }
                            bbs_16_size++;
                        }
                    }

                    int bbs_8_size = 0;
                    if ((num_vecs_in_block - bbs_16_size * 16) / 8 > 0) {
                        const size_t full_blocks =
                                (num_vecs_in_block - bbs_16_size * 16) / 8;
                        const size_t alloc_size =
                                full_blocks * 8 * d * sizeof(float);
                        // 安全分配内存 (处理 size=0 情况)
                        if (alloc_size > 0) {
                            block.bbs_8 = static_cast<float*>(
                                    aligned_alloc(64, alloc_size));
                            if (!block.bbs_8) {
                                throw std::bad_alloc();
                            }
                        } else {
                            block.bbs_8 = nullptr;
                        }

                        //                        block.bbs_8 = (float*)
                        //                        aligned_alloc(32, ((int)
                        //                        (num_vecs_in_block -
                        //                        bbs_16_size * 16) / 8) * 8 * d
                        //                        * sizeof(float));
                        for (; v_id + 7 < end; v_id += 8) {
                            float* bbs8 = block.bbs_8 + bbs_8_size * 8 * d;
                            for (int dim = 0; dim < d; dim++) {
                                for (int i = 0; i < 8; i++) {
                                    idx_t global_idx =
                                            dist_with_index[v_id + i].second;
                                    const float* vec = codes + global_idx * d;
                                    bbs8[dim * 8 + i] = vec[dim];
                                }
                            }

                            for (int i = 0; i < 8; i++) {
                                idx_t global_idx =
                                        dist_with_index[v_id + i].second;
                                block.offsets.push_back(global_idx);
                            }
                            bbs_8_size++;
                        }
                    }

                    size_t remaining = end - v_id;
                    if (remaining > 0) {
                        const size_t alloc_size = remaining * d * sizeof(float);
                        if (alloc_size > 0) {
                            block.vectors = static_cast<float*>(
                                    aligned_alloc(64, alloc_size));
                            if (!block.vectors) {
                                throw std::bad_alloc();
                            }
                        } else {
                            block.vectors = nullptr;
                        }

                        for (size_t i = v_id; i < end; i++) {
                            idx_t global_idx = dist_with_index[i].second;
                            const float* vec = codes + global_idx * d;
                            memcpy(block.vectors + (i - v_id) * d,
                                   vec,
                                   d * sizeof(float));
                            block.offsets.push_back(dist_with_index[i].second);
                        }
                    }
                    // 记录每个向量到分区中心的距离
                    partition.blocks[b] = std::move(block);
                    std::destroy(
                            dist_with_index.begin(), dist_with_index.end());
                    std::destroy(ip_with_index.begin(), ip_with_index.end());
                }
            }
        }

        // 打包分组的中心
        if (partitions.size() / 16 > 0) {
            const size_t full_blocks = partitions.size() / 16;
            const size_t alloc_size = full_blocks * 16 * d * sizeof(float);
            float* p_centroids = cluster_partition_centroids[list_no];
            // 安全分配内存 (处理 size=0 情况)
            if (alloc_size > 0) {
                p_centroids =
                        static_cast<float*>(aligned_alloc(64, alloc_size));
                if (!p_centroids) {
                    throw std::bad_alloc();
                }
            } else {
                p_centroids = nullptr;
            }
            //                        block.bbs_16 = (float*) aligned_alloc(64,
            //                        ((int) num_vecs_in_block / 16) * 16 * d *
            //                        sizeof(float));
            size_t v_id = 0;
            size_t end = partitions.size();
            int bbs_16_size = 0;
            for (; v_id + 15 < end; v_id += 16) {
                float* bbs16 = p_centroids + bbs_16_size * 16 * d;
                for (int dim = 0; dim < d; dim++) {
                    for (int i = 0; i < 16; i++) {
                        bbs16[dim * 16 + i] =
                                partitions[v_id + i].centroid[dim];
                    }
                }
                bbs_16_size++;
            }
            cluster_partition_centroids[list_no] = p_centroids;
        }
        cluster_min_max[list_no].first = {min, max};
        cluster_min_max[list_no].second = block_nums;
        assignment.clear();
        cluster_indices.clear();
        normalized_vectors.clear();
        centroids.clear();
        std::destroy(assignment.begin(), assignment.end());
        std::destroy(normalized_vectors.begin(), normalized_vectors.end());
        std::destroy(cluster_indices.begin(), cluster_indices.end());
        std::destroy(centroids.begin(), centroids.end());
        partitions_per_cluster_[list_no] = std::move(partitions);
    }
}

void IndexPartitionBlockFlatSIMD::norm_vector(
        const std::vector<float>& centroid,
        std::vector<float>& e_vec) const {
    for (size_t i = 0; i < d; i++) {
        e_vec[i] -= centroid[i];
    }

    float norm_rel = fvec_norm_L2sqr(e_vec.data(), d);
    norm_rel = std::sqrt(norm_rel);
    for (size_t j = 0; j < d; j++) {
        e_vec[j] /= norm_rel;
    }
}

size_t IndexPartitionBlockFlatSIMD::get_num_blocks(idx_t list_no) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return blocks_per_cluster_[list_no].size();
}

const std::vector<idx_t>& IndexPartitionBlockFlatSIMD::get_block_offsets(
        idx_t list_no,
        size_t block_id) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    FAISS_THROW_IF_NOT(block_id < blocks_per_cluster_[list_no].size());
    return blocks_per_cluster_[list_no][block_id].offsets;
}

const std::vector<IndexPartitionBlockFlatSIMD::BlockData>&
IndexPartitionBlockFlatSIMD::get_block_distances(idx_t list_no) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return blocks_per_cluster_[list_no];
}

std::vector<idx_t> IndexPartitionBlockFlatSIMD::get_block_ids(
        idx_t list_no,
        size_t block_id) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    FAISS_THROW_IF_NOT(block_id < blocks_per_cluster_[list_no].size());

    const auto& block = blocks_per_cluster_[list_no][block_id];
    InvertedLists::ScopedIds ids(invlists, list_no);

    std::vector<idx_t> result;
    result.reserve(block.offsets.size());
    for (idx_t offset : block.offsets) {
        result.push_back(ids[offset]);
    }
    return result;
}

void angle_bounds_simd(
        float alpha,
        float& L_j_q,
        float& U_j_q,
        float& delta,
        float theta_min,
        float theta_max) {
    if (theta_min <= alpha && alpha <= theta_max) {
        U_j_q = 1.0f;
    }
    L_j_q = std::cos(alpha + delta);
    U_j_q = std::cos(alpha - delta);
    //    if (theta_min <= alpha && alpha <= theta_max) {
    //        U_j_q = 1.0f;
    //    }
    //    if (theta_min <= M_PI - alpha && M_PI - alpha <= theta_max) {
    //        L_j_q = -1.0f;
    //    }

    //    if (theta_min <= M_PI - alpha && M_PI - alpha <= theta_max) {
    //        L_j_q = -1.0f;
    //    } else {
    //        L_j_q = std::cos(alpha + theta_min);
    //    }
    //    // 上界U_j^q
    //    if (theta_min <= alpha && alpha <= theta_max) {
    //        U_j_q = 1.0f;
    //    } else {
    ////        float cos1 = std::cos(alpha) * std::cos(theta_min) +
    ////                delata * std::sin(alpha) * std::sin(theta_min);
    ////        float cos2 = std::cos(alpha) * std::cos(theta_max) +
    ////                delata * std::sin(alpha) * std::sin(theta_max);
    //        U_j_q = std::cos(alpha - theta_max);
    ////        L_j_q = std::cos(std::min(alpha - theta_max, 0.0f));
    //        //                    float cand1 = std::fabs(alpha - theta_min);
    //        //                    float cand2 = std::fabs(alpha - theta_max);
    //        //                    float min_angle = std::min(cand1, cand2);
    //        //                    U_j_q = std::cos(min_angle);
    //    }
}
std::vector<idx_t> IndexPartitionBlockFlatSIMD::get_block_ids(
        idx_t list_no,
        int partition_id,
        size_t block_id) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    FAISS_THROW_IF_NOT(partition_id < partitions_per_cluster_[list_no].size());

    const auto& par = partitions_per_cluster_[list_no][partition_id];
    InvertedLists::ScopedIds ids(invlists, list_no);

    std::vector<idx_t> result;
    result.reserve(par.blocks[block_id].offsets.size());
    for (idx_t offset : par.blocks[block_id].offsets) {
        result.push_back(ids[offset]);
    }
    return result;
}

// 获取指定簇的所有分区
const std::vector<IndexPartitionBlockFlatSIMD::PartitionData>&
IndexPartitionBlockFlatSIMD::get_partitions(idx_t list_no) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return partitions_per_cluster_[list_no];
}

// 获取指定簇的指定分区内的所有block
const std::vector<IndexPartitionBlockFlatSIMD::BlockData>&
IndexPartitionBlockFlatSIMD::get_blocks(idx_t list_no, idx_t partition_id)
        const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    FAISS_THROW_IF_NOT(partition_id < partitions_per_cluster_[list_no].size());
    return partitions_per_cluster_[list_no][partition_id].blocks;
}

// 获取指定簇的指定分区内的指定block
const IndexPartitionBlockFlatSIMD::BlockData& IndexPartitionBlockFlatSIMD::
        get_block(idx_t list_no, idx_t partition_id, idx_t block_id) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    FAISS_THROW_IF_NOT(partition_id < partitions_per_cluster_[list_no].size());
    const auto& blocks = partitions_per_cluster_[list_no][partition_id].blocks;
    FAISS_THROW_IF_NOT(block_id < blocks.size());
    return blocks[block_id];
}

std::vector<float> IndexPartitionBlockFlatSIMD::get_block_codes(
        idx_t list_no,
        size_t block_id) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    FAISS_THROW_IF_NOT(block_id < blocks_per_cluster_[list_no].size());

    const auto& block = blocks_per_cluster_[list_no][block_id];
    InvertedLists::ScopedCodes scodes(invlists, list_no);
    const float* codes = (const float*)scodes.get();

    std::vector<float> result;
    result.reserve(block.offsets.size() * d);
    for (idx_t offset : block.offsets) {
        const float* vec = codes + offset * d;
        result.insert(result.end(), vec, vec + d);
    }
    return result;
}

std::vector<float> IndexPartitionBlockFlatSIMD::get_block_codes(
        idx_t list_no,
        int partition_id,
        size_t block_id) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    FAISS_THROW_IF_NOT(partition_id < partitions_per_cluster_[list_no].size());

    const auto& partition = partitions_per_cluster_[list_no][partition_id];
    InvertedLists::ScopedCodes scodes(invlists, list_no);
    const float* codes = (const float*)scodes.get();

    std::vector<float> result;
    result.reserve(partition.blocks[block_id].offsets.size() * d);
    for (idx_t offset : partition.blocks[block_id].offsets) {
        const float* vec = codes + offset * d;
        result.insert(result.end(), vec, vec + d);
    }
    return result;
}

void IndexPartitionBlockFlatSIMD::query_raw(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const IVFSearchParameters* params,
        std::vector<double>& times,
        std::vector<double>& ndis) const {
    FAISS_THROW_IF_NOT(k > 0);
    double t0 = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());
    // 并行处理每个查询

    // 并行处理每个查询
    int batch = 10000;
    int batch_num = (n + batch - 1) / batch;
    for (int bat = 0; bat < batch_num; bat++) {
        int start = bat * batch;
        int end = (bat + 1) * batch;
#pragma omp parallel for if (n > 10)
        for (idx_t i = start; i < std::min(end, (int)n); i++) {
            // 当前查询向量
            double t0 = getmillisecs();
            const float* xi = x + i * d;
            float* dist_i = distances + i * k;
            idx_t* label_i = labels + i * k;
            // 第二步：遍历每个簇，收集所有候选block
            using Heap = CMax<float, idx_t>;
            // 初始化结果堆
            heap_heapify<Heap>(k, dist_i, label_i);
            float* vec = new float[d];
            float current_threshold = dist_i[0]; // 当前最大距离
            long count = 0;
            for (size_t j = 0; j < nprobe; j++) {
                idx_t list_no = coarse_idx[i * nprobe + j];
                if (list_no < 0)
                    continue;
                // 提取该簇里面索引的原始数据，并计算距离
                for (int ii = 0; ii < invlists->list_size(list_no); ii++) {
                    // 更新结果堆
                    reconstruct_from_offset(list_no, ii, vec);
                    float dis = fvec_L2sqr(xi, vec, d);
                    if (dis < current_threshold) {
                        idx_t id = invlists->get_single_id(list_no, ii);
                        heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                        current_threshold = dist_i[0];
                    }
                }
                count += invlists->list_size(list_no);
            }
            delete[] vec;
            // 最终排序结果
            heap_reorder<Heap>(k, dist_i, label_i);
            std::lock_guard<std::mutex> lock(stats_mutex);
            indexIVF_stats.ndis += count;
            times[i] = getmillisecs() - t0;
            ndis[i] = count;
        }
    }
    indexIVF_stats.search_time += getmillisecs() - t0;
}

#ifdef __AVX2__
// 计算8个向量的内积
inline __m256 compute_8_ips(
        const float* query,
        const float* vecs[8],
        size_t d) {
    __m256 sum = _mm256_setzero_ps();
    for (size_t dim = 0; dim < d; dim++) {
        __m256 q = _mm256_set1_ps(query[dim]);
        __m256 v = _mm256_set_ps(
                vecs[7][dim],
                vecs[6][dim],
                vecs[5][dim],
                vecs[4][dim],
                vecs[3][dim],
                vecs[2][dim],
                vecs[1][dim],
                vecs[0][dim]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(q, v));
    }
    return sum;
}

// 计算8个向量的L2距离
inline __m256 compute_8_l2s(
        const float* query,
        const float* vecs[8],
        size_t d) {
    __m256 sum = _mm256_setzero_ps();
    for (size_t dim = 0; dim < d; dim++) {
        __m256 q = _mm256_set1_ps(query[dim]);
        __m256 v = _mm256_set_ps(
                vecs[7][dim],
                vecs[6][dim],
                vecs[5][dim],
                vecs[4][dim],
                vecs[3][dim],
                vecs[2][dim],
                vecs[1][dim],
                vecs[0][dim]);
        __m256 diff = _mm256_sub_ps(q, v);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    return sum;
}

inline __m256 compute_8_l2s_contiguous(
        const float* query,
        const float* base_ptr,
        size_t d) {
    __m256 sum = _mm256_setzero_ps();
    for (size_t dim = 0; dim < d; dim++) {
        __m256 q = _mm256_set1_ps(query[dim]);
        __m256 v_val = _mm256_load_ps(base_ptr + dim * 8);
        __m256 diff = _mm256_sub_ps(q, v_val);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    return sum;
}

#endif // __AVX2__

#ifdef __AVX512F__
// 计算8个向量的L2距离
inline __m512 compute_16_l2s(
        const float* query,
        const float* vecs[16],
        size_t d) {
    __m512 sum = _mm512_setzero_ps();
    for (size_t dim = 0; dim < d; dim++) {
        __m512 q = _mm512_set1_ps(query[dim]);
        __m512 v = _mm512_set_ps(
                vecs[15][dim],
                vecs[14][dim],
                vecs[13][dim],
                vecs[12][dim],
                vecs[11][dim],
                vecs[10][dim],
                vecs[9][dim],
                vecs[8][dim],
                vecs[7][dim],
                vecs[6][dim],
                vecs[5][dim],
                vecs[4][dim],
                vecs[3][dim],
                vecs[2][dim],
                vecs[1][dim],
                vecs[0][dim]);
        __m512 diff = _mm512_sub_ps(q, v);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    return sum;
}

// 优化后的连续内存访问版本
inline __m512 compute_16_l2s_contiguous(
        const float* query,
        const float* base_ptr,
        size_t d) {
    __m512 sum = _mm512_setzero_ps();

    for (size_t dim = 0; dim < d; dim++) {
        // 广播查询值
        __m512 q_val = _mm512_set1_ps(query[dim]);

        // 加载16个连续值
        __m512 v_val = _mm512_load_ps(base_ptr + dim * 16);

        // 计算差值
        __m512 diff = _mm512_sub_ps(q_val, v_val);

        // 累加平方
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    return sum;
}

inline float compute_one2one_16_contiguous(
        const float* query,
        const float* base_ptr,
        size_t d) {
    float sum = 0.0f;
    size_t dim = 0;
    __m512 sum_vec = _mm512_setzero_ps();
    for (; dim + 15 < d; dim += 16) {
        // 广播查询值
        __m512 q_val = _mm512_loadu_ps(query + dim);

        // 加载16个连续值
        __m512 v_val = _mm512_load_ps(base_ptr + dim);

        // 计算差值
        __m512 diff = _mm512_sub_ps(q_val, v_val);

        // 累加平方
        //        diff = _mm512_mul_ps(diff, diff);
        //        sum += _mm512_reduce_add_ps(diff);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec); // 乘加累积
    }

    sum += _mm512_reduce_add_ps(sum_vec); // 向量归约
    // 处理剩余维度
    for (; dim < d; dim++) {
        float diff = query[dim] - base_ptr[dim];
        sum += diff * diff;
    }
    return sum;
}

inline float ip_one2one_16_contiguous(
        const float* query,
        const float* p,
        size_t d) {
    float sum = 0.0f;
    size_t dim = 0;
    __m512 sum_vec = _mm512_setzero_ps();
    for (; dim + 15 < d; dim += 16) {
        // 广播查询值
        __m512 q_val = _mm512_loadu_ps(query + dim);

        // 加载16个连续值
        __m512 p_val = _mm512_loadu_ps(p + dim);

        // 计算差值
        // 累加平方
        //        diff = _mm512_mul_ps(diff, diff);
        //        sum += _mm512_reduce_add_ps(diff);
        sum_vec = _mm512_fmadd_ps(q_val, p_val, sum_vec); // 乘加累积
    }

    sum += _mm512_reduce_add_ps(sum_vec); // 向量归约
    // 处理剩余维度
    for (; dim < d; dim++) {
        sum += query[dim] * p[dim];
    }
    return sum;
}

inline __m512 ip_16_contiguous(
        const float* query,
        const float* centriod,
        const float* p,
        size_t d,
        float R_q) {
    __m512 sum = _mm512_setzero_ps();

    for (size_t dim = 0; dim < d; dim++) {
        // 广播查询值
        float diff = query[dim] - centriod[dim];
        __m512 q_val = _mm512_set1_ps(diff);
        // 加载16个连续值
        __m512 v_val = _mm512_load_ps(p + dim * 16);

        // 累加平方
        sum = _mm512_fmadd_ps(q_val, v_val, sum);
    }
    float rqv = 1.0 / R_q;
    __m512 rq = _mm512_set1_ps(rqv);
    sum = _mm512_mul_ps(sum, rq);
    return sum;
}

inline __m512 ip_16_contiguous_positive(
        const float* query,
        const float* centriod,
        const float* p,
        size_t d,
        float R_q) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> rand_dis(0.0, 1.0);
    float b = 1.0; // 你可以根据需要调整b的值
    __m512 sum = _mm512_setzero_ps();

    for (size_t dim = 0; dim < d; dim++) {
        // 广播查询值

        float diff = query[dim] - centriod[dim];
        if (query[dim] < centriod[dim]) {
            float a = rand_dis(gen);
            if (a < b) {
                diff *= -1.0f;
            }
        }
        __m512 q_val = _mm512_set1_ps(diff);
        // 加载16个连续值
        __m512 v_val = _mm512_load_ps(p + dim * 16);

        // 累加平方
        sum = _mm512_fmadd_ps(q_val, v_val, sum);
    }
    float rqv = 1.0 / R_q;
    __m512 rq = _mm512_set1_ps(rqv);
    sum = _mm512_mul_ps(sum, rq);
    return sum;
}

#endif // __AVX512F__

void IndexPartitionBlockFlatSIMD::query_by_distance(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int M,
        int block_size,
        float iter_factor,
        const IVFSearchParameters* params,
        std::vector<double>& times,
        std::vector<double>& ndis) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(M > 0);
    double t0 = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());
    using Heap = CMax<float, idx_t>;
#pragma omp parallel for if (n > 10)
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        float* dist_i = distances + i * k;
        idx_t* label_i = labels + i * k;

        // 初始化结果堆
        heap_heapify<Heap>(k, dist_i, label_i);
        float current_threshold = dist_i[0]; // 当前最大距离
        struct CandidateBlock {
            idx_t list_no;
            int partition_id;
            size_t block_id;
            float lb; // 下界
        };
        std::vector<CandidateBlock> candidate_blocks;

        // 第二步：遍历每个簇，收集所有候选block
        //        #pragma omp parallel for if (n > 10)
        int count = 0;
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;

            // 获取该簇的分区数据
            const auto& partitions = partitions_per_cluster_[list_no];
            std::vector<float> centroid(d);
            quantizer->reconstruct(list_no, centroid.data());
            // 查询向量到簇心的距离
            float R_q = 0.0f;
            if (metric_type == METRIC_INNER_PRODUCT) {
                // 对于内积，计算查询向量与簇心的内积
                R_q = fvec_inner_product(xi, centroid.data(), d);
            } else {
                // 对于L2，计算查询向量到簇心的距离
                R_q = fvec_L2sqr(xi, centroid.data(), d);
                R_q = std::sqrt(R_q);
            }
            float R_q2 = R_q * R_q;

            float min = cluster_min_max[list_no].first.first;
            float max = cluster_min_max[list_no].first.second;

            if (R_q > max) {
                float lb = R_q - max;
                lb *= lb;
                if (lb > current_threshold && count > k) {
                    break;
                }
            }

            float* xi_centroid = new float[d];
            //            std::vector<float> xi_centroid(d);
            // 计算查询向量与簇心的残差
            for (int jj = 0; jj < d; jj++) {
                xi_centroid[jj] = xi[jj] - centroid[jj];
            }
            // 计算查询向量与簇心的残差的L2平方
            float R_q_sqr = fvec_norm_L2sqr(xi_centroid, d);

            // 计算查询向量与簇心的残差的L2
            float R_q_norm = std::sqrt(R_q_sqr);

            // 计算查询向量与簇心的残差的单位向量
            float* xi_centroid_unit = new float[d];
            for (int jj = 0; jj < d; jj++) {
                xi_centroid_unit[jj] = xi_centroid[jj] / R_q_norm;
            }

            for (int m = 0; m < partitions.size(); m++) {
                // 计算分区m的下界和上界
                float L_j_q = 0.55f;
                float U_j_q = 0.55f;
                auto partition = partitions[m];

                const auto& blocks = partition.blocks;
                for (size_t b = 0; b < blocks.size(); b++) {
                    const auto& block = blocks[b];
                    float min_dist = block.min_dist;
                    float max_dist = block.max_dist;
                    CandidateBlock candidate_block;
                    candidate_block.list_no = list_no;
                    candidate_block.partition_id = m;
                    candidate_block.block_id = b;
                    //                    if (min_dist <= R_q && R_q <=
                    //                    max_dist) {
                    //
                    //                    }
                    //
                    //                    if (max_dist <= R_q) {
                    //                        float dis = R_q - max_dist;
                    //                        candidate_block.lb = dis * dis;
                    //                    } else {
                    //                        float dis = min_dist - R_q;
                    //                        candidate_block.lb = dis * dis;
                    //                    }
                    float lb =
                            getLB(R_q,
                                  R_q2,
                                  L_j_q,
                                  U_j_q,
                                  block.min_dist,
                                  block.max_dist);
                    if (lb > current_threshold) {
                        continue;
                    }
                    candidate_block.lb = lb;
                    //                    printf("lb: %.6f, min_dist: %.6f,
                    //                    max_dist: %.6f, R_q: %.6f\n",
                    //                           candidate_block.lb,
                    //                           min_dist,
                    //                           max_dist,
                    //                           R_q);
                    candidate_blocks.push_back(candidate_block);
                }
            }
        }
        // 第三步：按照下界排序
        std::sort(
                candidate_blocks.begin(),
                candidate_blocks.end(),
                [](const CandidateBlock& a, const CandidateBlock& b) {
                    return a.lb < b.lb;
                });

        // 第四步：遍历候选block
        int iter_times = 0;
        for (const auto& cand : candidate_blocks) {
            iter_times++;
            // 如果下界大于当前阈值，跳过该block
            //            float tmp = sqrt(current_threshold) / dd;
            //            printf(  "cand.lb: %.6f, current_threshold: %.6f, tmp:
            //            %.6f\n", cand.lb, current_threshold, tmp); if (cand.lb
            //            > current_threshold || ++iter_times >= max_iter) { if
            //            (cand.lb > current_threshold || ++iter_times >=
            //            max_iter) {
            if (cand.lb > current_threshold && count > k) {
                break;
            }

            // 获取block数据
            const auto& block =
                    get_block(cand.list_no, cand.partition_id, cand.block_id);

            // 获取block内的向量
            const std::vector<idx_t>& offsets = block.offsets;
            size_t num_vecs = offsets.size();

            // 计算block内每个向量的距离
            for (size_t v_idx = 0; v_idx < num_vecs; v_idx++) {
                idx_t offset = offsets[v_idx];
                float* vec = new float[d];
                reconstruct_from_offset(cand.list_no, offset, vec);

                float dis;
                if (metric_type == METRIC_INNER_PRODUCT) {
                    dis = -fvec_inner_product(xi, vec, d);
                } else {
                    dis = fvec_L2sqr(xi, vec, d);
                }
                delete[] vec;

                if (dis < current_threshold) {
                    // 获取向量的id
                    //                    printf("current_threshold: %.6f\n",
                    //                    current_threshold);
                    idx_t id = invlists->get_single_id(cand.list_no, offset);
                    heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                    current_threshold = dist_i[0];
                }
                count += num_vecs;
            }
            std::lock_guard<std::mutex> lock(stats_mutex);
            //            printf("indexIVF_stats: ndis: %zu, %zu",
            //            indexIVF_stats.ndis, num_vecs);
            indexIVF_stats.ndis += num_vecs;
        }

        //        printf("query no: %d, [block_size]: %zu, [iter_times]: %d\n",
        //               i,
        //               candidate_blocks.size(),
        //               iter_times);
        // 最终排序结果
        heap_reorder<Heap>(k, dist_i, label_i);
    }
    indexIVF_stats.search_time += getmillisecs() - t0;
}

// 定义分区候选结构
struct PartitionCandidate {
    idx_t list_no;
    int partition_id;
    float Rq_2;
    float Rq;
    float L_j_q;
    float U_j_q;
    float lb; // 分区的下界
    float alpha_cos;
    float theta;
    float R;
    // 用于优先队列比较
    bool operator<(const PartitionCandidate& other) const {
        return lb > other.lb;
    }
};

// 定义block候选结构
struct BlockCandidate {
    idx_t list_no;
    int partition_id;
    size_t block_id;
    float lb; // block的下界
    bool operator<(const BlockCandidate& other) const {
        return lb > other.lb;
    }
};

// 生成M个d维正交随机向量，结果存储在e_vects中（列优先存储）
void IndexPartitionBlockFlatSIMD::e_vects_generator_M2(
        float* e_vects,
        int d,
        int M) {
    // 验证输入参数有效性
    if (d < M) {
        throw std::invalid_argument("Dimension d must be >= M");
    }
    if (M <= 0 || d <= 0) {
        throw std::invalid_argument("Dimensions must be positive");
    }

    // 使用随机设备初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < M; i++) {
        float* e = e_vects + i * d;
        for (int dim = 0; dim < d; dim++) {
            e[dim] = dist(gen);
        }

        float dist_e = fvec_norm_L2sqr(e, d);
        if (dist_e > 0.0) {
            dist_e = 1.0f / std::sqrt(dist_e);
            for (int j = 0; j < d; j++) {
                e[j] *= dist_e;
            }
        }
    }
}

// 生成M个d维正交随机向量，结果存储在e_vects中（列优先存储）
void RP(float* e_vects, int d, int M) {
    // 验证输入参数有效性
    if (d < M) {
        throw std::invalid_argument("Dimension d must be >= M");
    }
    if (M <= 0 || d <= 0) {
        throw std::invalid_argument("Dimensions must be positive");
    }

    // 使用随机设备初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f / d);

    // 创建d×d的随机高斯矩阵
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(d, d);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            A(i, j) = dist(gen);
        }
    }

    // 进行QR分解获取正交矩阵
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
    Eigen::MatrixXf Q = qr.householderQ() * Eigen::MatrixXf::Identity(d, d);

    // 取前M列作为结果向量
    for (int vec_idx = 0; vec_idx < M; ++vec_idx) {
        for (int dim = 0; dim < d; ++dim) {
            e_vects[vec_idx * d + dim] = Q(dim, vec_idx);
        }
    }
}

// 生成M个d维正交随机向量，结果存储在e_vects中（列优先存储）
void IndexPartitionBlockFlatSIMD::e_vects_generator_M(
        float* e_vects,
        int d,
        int M) {
    // 验证输入参数有效性
    if (d < M) {
        throw std::invalid_argument("Dimension d must be >= M");
    }
    if (M <= 0 || d <= 0) {
        throw std::invalid_argument("Dimensions must be positive");
    }

    std::mt19937 rng(1234);

    std::mt19937 gen(rng);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // 创建d×d的随机高斯矩阵
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(d, d);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            A(i, j) = dist(gen);
        }
    }

    // 进行QR分解获取正交矩阵
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
    Eigen::MatrixXf Q = qr.householderQ() * Eigen::MatrixXf::Identity(d, d);

    // 取前M列作为结果向量
    for (int vec_idx = 0; vec_idx < M; ++vec_idx) {
        for (int dim = 0; dim < d; ++dim) {
            e_vects[vec_idx * d + dim] = Q(dim, vec_idx);
        }
    }

    for (int i = 0; i < M; i++) {
        float* e = e_vects + i * d;
        float dist = fvec_norm_L2sqr(e, d);
        if (dist > 0.0) {
            dist = 1.0f / std::sqrt(dist);
            for (int j = 0; j < d; j++) {
                e[j] *= dist;
            }
        }
    }
}

void IndexPartitionBlockFlatSIMD::e_vects_generator(
        float* e_vects,
        int d,
        int M) {
    // 生成M个d维的单位向量
    std::random_device rd;
    std::mt19937 gen(1234);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<float> e1(d, 1.0f / std::sqrt(d));
    for (int i = 0; i < d; i++) {
        e1[i] = dist(gen);
    }
    std::vector<float> r(d);
    r.resize(d);
    float min_angle = std::cos(M_PI / M / 2.0);
    for (int i = 0; i < M; i++) {
        float* e = e_vects + i * d;
        int max_iter = 0;
        bool flag = true;
        while (flag && max_iter < 10) {
            std::generate_n(r.begin(), d, [&] { return dist(gen); });
            float re = fvec_inner_product(r.data(), e1.data(), d);
            for (int dim = 0; dim < d; dim++) {
                e[dim] = r[dim] - re * e1[dim];
            }

            float dist = fvec_norm_L2sqr(e, d);
            bool is_ok = true;
            if (dist > 0.0) {
                dist = 1.0f / std::sqrt(dist);
                for (int j = 0; j < i; j++) {
                    float cos = fvec_inner_product(e_vects + j * d, e, d);
                    cos = cos * dist;
                    if (min_angle > cos) {
                        is_ok = false;
                        break;
                    }
                }
            }
            flag = !is_ok;
            for (int dim = 0; dim < d; dim++) {
                e[dim] *= dist;
            }
            max_iter++;
        }
    }
    printf("");
}

// 优先队列
void IndexPartitionBlockFlatSIMD::query_by_cluster_priority_queue(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int M,
        int block_size,
        float iter_factor,
        const IVFSearchParameters* params,
        std::vector<double>& times,
        std::vector<double>& ndis) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(M > 0);
    double t_start = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());
    indexIVF_stats.quantization_time += getmillisecs() - t_start;
    using Heap = CMax<float, idx_t>;
    // 并行处理每个查询
    //    int batch = 10000;
    //    int batch_num = (n + batch - 1) / batch;
    //    for (int bat = 0; bat < batch_num; bat ++) {
    //        int start = bat * batch;
    //        int end = (bat + 1) * batch;
//    auto* t_lb = new double[n];
//    auto* t_ann = new double[n];
//    auto* t_total = new double[n];
//    auto* total = new double[n];
//    for(int i =0; i <n;i ++) {
//        t_lb[i] = 0.0;
//        t_ann[i]  =0.0;
//        t_total[i] =0.0;
//        total[i]  =0.0;
//    }
    double t_compute = 0.0f;
#pragma omp parallel for if (n > 10) schedule(dynamic)
    for (idx_t i = 0; i < n; i++) {
        // 当前查询向量
        double t_s = getmillisecs();
        const float* xi = x + i * d;
        float* dist_i = distances + i * k;
        idx_t* label_i = labels + i * k;

        // 初始化结果堆
        heap_heapify<Heap>(k, dist_i, label_i);
        float current_threshold = dist_i[0]; // 当前最大距离
        //        printf("current_threshold: %.6f\n", current_threshold);
        int count = 0;

        // 分区优先队列（最小堆）
        //        std::priority_queue<PartitionCandidate,
        //        std::vector<PartitionCandidate>,
        //        std::greater<PartitionCandidate>> partition_queue;
        int block_num = 0;
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;
            block_num += cluster_min_max[list_no].second;
        }

        int max_iter = (int)((float)block_num * iter_factor);
        max_iter = std::max(1, max_iter);
        int iter_items = 0;
        // 遍历每个簇，收集所有候选分区
        for (size_t j = 0; j < nprobe; j++) {
            double t0 = getmillisecs();
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;

            // 获取该簇的分区数据
            const auto& partitions = partitions_per_cluster_[list_no];
            std::vector<float> centroid(d);
            //            auto* centroid = new float[d];
            quantizer->reconstruct(list_no, centroid.data());
            // 查询向量到簇心的距离
            float R_q = 0.0f;
            float R_q2 = coarse_dis[i * nprobe + j];
            R_q = std::sqrt(R_q2);

            float min = cluster_min_max[list_no].first.first;
            float max = cluster_min_max[list_no].first.second;

            if (R_q > max) {
                float lb = R_q - max;
                lb *= lb;
                if (lb > current_threshold && count > k) {
                    break;
                }
            }

            int m = 0;
            const float* p_c = cluster_partition_centroids[list_no];
            const float* p_c_std = cluster_partition_centroids_std[list_no];
            float* p_alpha;
            float* alphas = new float[partitions.size()];
            float* alphas_std = new float[partitions.size()];
#ifdef __AVX512F__
            int b16_size = 0;
            if (R_q > 1e-8f) {
                for (; m + 15 < partitions.size(); m += 16) {
                    __m512 distances = _mm512_setzero_ps();
                    const float* vec_ptrs = p_c + b16_size * 16 * d;
                    distances = ip_16_contiguous(
                            xi, centroid.data(), vec_ptrs, d, R_q);
                    // 处理8个距离结果
                    alignas(64) float dist_arr[16];
                    _mm512_store_ps(dist_arr, distances);
                    for (int i = m; i < m + 16; i++) {
                        alphas[i] = dist_arr[i - m];
                    }

                    // 计算alphas_std
                    //                    const float* vec_ptrs_std = p_c_std +
                    //                    b16_size * 16 * d; b16_size++;
                    //                    distances = ip_16_contiguous_positive(
                    //                            xi, centroid.data(),
                    //                            vec_ptrs_std, d, R_q);
                    //                    // 处理8个距离结果
                    //                    alignas(64) float dist_arr_2[16];
                    //                    _mm512_store_ps(dist_arr_2,
                    //                    distances); for (int i = m; i < m +
                    //                    16; i++) {
                    //                        alphas_std[i] = dist_arr_2[i - m];
                    //                    }
                }
            }
#endif

            float sum_q_positive = 0.0;
            // 计算查询向量与簇心的残差
            std::vector<float> xi_centroid_unit(d);
            std::vector<float> xi_centroid_unit_std(d);

            //            for (int jj = 0; jj < d; jj++) {
            //                xi_centroid_unit[jj] = xi[jj] - centroid[jj];
            //                if (xi_centroid_unit[jj] < 0.0) {
            //                    xi_centroid_unit_std[jj] =
            //                    -xi_centroid_unit[jj];
            //                } else {
            //                    xi_centroid_unit_std[jj] =
            //                    xi_centroid_unit[jj];
            //                }
            //                sum_q_positive += xi_centroid_unit_std[jj];
            //            }
            //            sum_q_positive /= R_q;

            for (int jj = 0; jj < d; jj++) {
                xi_centroid_unit[jj] = xi[jj] - centroid[jj];
                if (xi_centroid_unit[jj] < 0.0) {
                    xi_centroid_unit_std[jj] =
                    -xi_centroid_unit[jj];
                } else {
                    xi_centroid_unit_std[jj] =
                    xi_centroid_unit[jj];
                }
                sum_q_positive += xi_centroid_unit_std[jj];
            }
            sum_q_positive /= R_q;

            if (m < partitions.size()) {
//                for (int jj = 0; jj < d; jj++) {
//                    xi_centroid_unit[jj] = xi[jj] - centroid[jj];
//                    if (xi_centroid_unit[jj] < 0.0) {
//                        xi_centroid_unit_std[jj] = -xi_centroid_unit[jj];
//                    } else {
//                        xi_centroid_unit_std[jj] = xi_centroid_unit[jj];
//                    }
//                                        sum_q_positive +=
//                                        xi_centroid_unit_std[jj];
//                }
//                sum_q_positive /= R_q;
                //                std::vector<float>
                //                xi_centroid_unit_tmp(xi_centroid_unit.data(),
                //                xi_centroid_unit.data() + d); rotation(Q_inv,
                //                xi_centroid_unit_tmp.data(),
                //                xi_centroid_unit.data(), d);
                for (; m < partitions.size(); m++) {
                    const PartitionData& partition = partitions[m];
                    // 获取该簇的单位向量e
                    const std::vector<float>& e_vec = partition.centroid;
                    // 计算查询向量q在e方向上的投影
                    float alpha_cos = 0.0f;
                    float alpha_cos_std = 0.0f;
                    if (R_q > 1e-8f) {
#ifdef __AVX512F__
                        alpha_cos = ip_one2one_16_contiguous(
                                            xi_centroid_unit.data(),
                                            e_vec.data(),
                                            d) /
                                R_q;
//                        alpha_cos_std =
//                        ip_one2one_16_contiguous(xi_centroid_unit_std.data(),
//                        partition.std_dev.data(), d) /
//                                R_q;
#else
                        alpha_cos = fvec_inner_product(
                                            xi_centroid_unit.data(),
                                            e_vec.data(),
                                            d) /
                                R_q;

//                        alpha_cos_std = fvec_inner_product(
//                                                xi_centroid_unit_std.data(),
//                                                partition.std_dev.data(), d)
//                                                /R_q;
#endif
                    }
                    alphas[m] = alpha_cos;
                    alphas_std[m] = alpha_cos_std;
                }
            }

            std::priority_queue<PartitionCandidate> partition_queue;
            // 遍历该簇的所有分区
            for (m = 0; m < partitions.size(); m++) {
                const PartitionData& partition = partitions[m];
                // 获取该簇的单位向量e
                const std::vector<float>& e_vec = partition.centroid;
                // 计算查询向量q在e方向上的投影
                float L_j_q, U_j_q;
                float alpha_cos = 0.0f;
                float alpha_cos_std = 0.0f;
                alpha_cos = alphas[m];
                //                alpha_cos_std = alphas_std[m];
                //                alpha_cos_std = sum_q_positive *
                //                partition.std;
                ////                alpha_cos_std = sum_q_positive /
                ///std::sqrt(d); /                printf("alpha_cos:%.4f,
                ///sum_q_positive: %.4f, std: %.4f, alpha_cos_std: %.4f\n",
                ///alpha_cos, sum_q_positive, partition.std, alpha_cos_std);
                //                float phi = 0.8;
                //                L_j_q = alpha_cos - phi * alpha_cos_std;
                //                U_j_q = alpha_cos + phi * alpha_cos_std;

                float phi = std::cos(45 * M_PI / 180.0);
                float R = std::sqrt(alpha_cos * alpha_cos + phi * phi * (1 - alpha_cos * alpha_cos));
                float theta = std::acos(alpha_cos / R);
                float theta_min = partition.theta_min - theta;
                float theta_max = partition.theta_max - theta;
//                printf("theta: %.4f, p_theta_min: %.4f, p_theta_max: %.4f, theta_min: %.4f, theta_max: %.4f, R: %.4f\n", theta / M_PI * 180, partition.theta_min / M_PI * 180, partition.theta_max / M_PI * 180,
//                       theta_min / M_PI * 180, theta_max / M_PI * 180, R);
                if (theta_max < 0.0) {
                    L_j_q = R * std::cos(theta_min);
                    U_j_q = R * std::cos(theta_max);
                } else if (theta_min < 0.0) {
                    L_j_q = R * std::cos(std::min(-theta_min, theta_max));
                    U_j_q = R;
                } else {
                    L_j_q = R * std::cos(theta_max);
                    U_j_q = R * std::cos(theta_min);
                }

//                phi = 1.68;
//                L_j_q = alpha_cos + phi * std::sqrt(2.0 - 2.0 * std::cos(partition.theta_min)) / std::sqrt(d) * sum_q_positive;
//                U_j_q = alpha_cos + phi * std::sqrt(2.0 - 2.0 * std::cos(partition.theta_max)) / std::sqrt(d) * sum_q_positive;

                //                float alpha = std::acos(alpha_cos);
                ////                float dist_q_e = std::sqrt(2 - 2 *
                ///alpha_cos);

                // 基于alpha的估计
                //                float delta = (partition.theta_max -
                //                partition.theta_min) ; float delta =
                //                partition.std * 5.0;
                ////                float delta = partition.gauss_delta;
                //                L_j_q = std::cos(alpha + delta);
                //                U_j_q = std::cos(alpha  - delta);
                //                U_j_q = alpha_cos;
                //                float delta = 0.2;
                //                float delta = (partition.theta_max -
                //                partition.theta_min); float delta = 2.0 *
                //                partition.std; float delta =
                //                partition.gauss_delta; angle_bounds_simd(
                //                        alpha,
                //                        L_j_q,
                //                        U_j_q,
                //                        delta,
                //                        partition.theta_min,
                //                        partition.theta_max);

                // 直接估计, 比较重要
                ////                float mean = std::max(alpha_cos *
                ///std::cos(partition.theta_min), alpha_cos *
                ///std::cos(partition.theta_max)); /                float delta
                ///= 0.5 * std::max(std::sin(alpha) *
                ///std::sin(partition.theta_min), std::sin(alpha) *
                ///std::sin(partition.theta_max));
                //                //基于cos估计

                // 利用角度估计u=pi/2， std= pi/2
                //                float phi = std::cos( M_PI / 2.0 - 1.059);

                // 均匀分布，效果不错
                //                float phi = std::cos( M_PI / 2.0 - 0.2 *
                //                M_PI); float phi = std::sin( M_PI / 2.0
                //                - 1.322); float phi =
                //                std::sqrt(std::log(partition.num) / d);

                // N（0， 1/sqrt（2））
                //                float phi  = 0.65;
                //                float alpha = std::acos(alpha_cos);
//                phi = std::sin(0.5 * M_PI / 2.0);
//                //                phi  = 0.14;
//                //                基于phi的估计
//                float cos_u = alpha_cos * std::cos(partition.u);
//                //                //                float cos_u =
//                //std::max(alpha_cos * std::cos(partition.theta_min), alpha_cos
//                // std::cos(partition.theta_max));
//                float alpha = std::acos(alpha_cos / R);
//                float sin_i_u = std::sin(alpha) *std::sin(partition.u);
//                ////                float sin_i_u = std::sin(alpha) *
//                //std::max(std::sin(partition.theta_min),
//                //std::sin(partition.theta_max));
//                //
//                float delta = phi * sin_i_u;
//                //////                float delta = phi *
//                //std::max(std::sin(alpha) *  std::sin(partition.theta_min),
//                //std::sin(alpha) *  std::sin(partition.theta_max));
//                ////
//                L_j_q = cos_u - delta;
//                U_j_q = cos_u + delta;

                // 基于距离和角度

                //                L_j_q = std::cos(alpha + partition.max_dis /
                //                R_q); U_j_q = std::cos(alpha -
                //                partition.min_dis / R_q);

                //                L_j_q = std::cos(alpha + partition.theta_min -
                //                std::asin(partition.max_dis / R_q)); U_j_q =
                //                std::cos(alpha + partition.theta_min -
                //                std::asin(partition.max_dis / R_q));

                // 基于phi的新预测
                //                float cos_u = alpha_cos *
                //                std::cos(partition.u); float sin_i_u = 0.5 *
                //                std::sin(alpha) *  std::sin(partition.u);
                //                float delta = std::sqrt(M_PI / (partition.num
                //                + 1.0)); delta = delta * sin_i_u; L_j_q =
                //                std::cos(alpha - partition.u) - delta; U_j_q =
                //                std::cos(alpha - partition.u) - delta;

                //                U_j_q = alpha_cos * std::cos(partition.u) +
                //                phi * std::sin(partition.u); L_j_q = U_j_q;
                // 基于最短距离的估计
                //                float min_d = std::min(std::abs(alpha -
                //                partition.theta_min), std::abs(alpha -
                //                partition.theta_max)); L_j_q = std::cos(alpha
                //                + min_d); U_j_q = std::cos(alpha - min_d);

                //                float delta_p = 1.0 / (d * std::sqrt(1.0 -
                //                0.9)); L_j_q = (cos_u - sin_i_u * delta_p) /
                //                (1 + sin_i_u); U_j_q = (cos_u + sin_i_u *
                //                delta_p) / (1 - sin_i_u); L_j_q =
                //                std::max(-1.0f, L_j_q); U_j_q = std::min(1.0f,
                //                U_j_q);
                //
                ////                L_j_q = alpha_cos;
                ////                U_j_q = alpha_cos;
                //
                //                float delta = 1.96 * partition.std;
                //                angle_bounds_simd(
                //                        alpha,
                //                        L_j_q,
                //                        U_j_q,
                //                        delta,
                //                        partition.theta_min,
                //                        partition.theta_max);

                //                float cos_tmp =
                //                std::max(std::cos(partition.theta_min),
                //                std::cos(partition.theta_max)); float
                //                cos_tmp_min =
                //                std::min(std::cos(partition.theta_min),
                //                std::cos(partition.theta_max));

                //                float cos_u = std::cos(partition.u);
                //                float radius = partition.theta_max -
                //                partition.theta_min;
                ////                float delta = dist_q_e *
                ///partition.o_e_dist_avg * cos(45.0 / 180.0 * M_PI);
                //                float delta = dist_q_e *
                //                partition.o_e_dist_avg * cos(M_PI/ 2.0 -
                //                radius);
                ////                float delta = 1.0;
                //
                //                L_j_q = alpha_cos + cos_u;
                //
                //                U_j_q = alpha_cos + cos_u - 1.0 + delta;

                //                U_j_q = alpha_cos + cos_u;

                //                U_j_q = 0.6;
                //               printf("m: %d, alpha: %f, theta_min: %f
                //               theta_max: %f, u:%f, std:%f, r: %f, L: %f,
                //               U:%f\n",
                //                                       m,
                //                                       alpha / M_PI * 180,
                //                                       partition.theta_min /
                //                                       M_PI * 180,
                //                                       partition.theta_max /
                //                                       M_PI * 180, partition.u
                //                                       / M_PI * 180,
                //                                       partition.std / M_PI *
                //                                       180,
                ////                                       delta / M_PI * 180,
                //                       delta_p,
                //                                       L_j_q,
                //                                       U_j_q);

                //                                float delta =
                //                                partition.gauss_delta;
                //                                angle_bounds_simd(alpha,
                //                                L_j_q, U_j_q,
                //                                delta,partition.theta_min,
                //                                partition.theta_max);

                // 计算delta和角度边界
                //                                float u1 =
                //                                (partition.theta_max -
                //                                partition.theta_min) / 4.0f;
                //                                float u2 =
                //                                (partition.theta_max +
                //                                partition.theta_min) / 2.0f;
                //                                float delta = std::sqrt((u1 *
                //                                u1 + u2 * u2) / 2.0f);
                //                                angle_bounds_simd(alpha,
                //                                L_j_q, U_j_q, delta,
                //                                partition.theta_min,
                //                                partition.theta_max);

                //                float delta = partition.gauss_delta;
                //                angle_bounds_simd(alpha, L_j_q, U_j_q, delta,
                //                partition.theta_min, partition.theta_max);

                //                float u = alpha_cos * std::cos(partition.u);
                //                float delta_u = 0.8 * std::sin(alpha) *
                //                std::sin(partition.u) / std::sqrt(2.0f);
                ////                float delta_u = std::sin(alpha) *
                ///partition.avg_sin  / std::sqrt(2.0f);
                //                //                delta = std::cos(alpha) *
                //                std::cos(partition.u) + 1.64 * std::sin(alpha)
                //                * std::sin(partition.u) / std::sqrt(2.0f);
                //                L_j_q = u - delta_u;
                //                U_j_q = u + delta_u;

                ////
                //                float exp = alpha_cos *
                //                partition.exp_cos_theta; float var =
                //                std::sqrt(alpha_cos * alpha_cos *
                //                partition.var_cos_theta  + 0.5 *
                //                std::sin(alpha) * std::sin(alpha) *
                //                partition.avg_sin *  partition.avg_sin); L_j_q
                //                = exp - var; U_j_q = exp + var;

                //                printf("L_j_q: %.6f, U_j_q: %.6f, alpha: %.6f,
                //                theta_min: %.6f, theta_max: %.6f, u:%0.6f,
                //                delta: %.6f, r: %.6f\n",
                //                       L_j_q, U_j_q, alpha / M_PI * 180,
                //                       partition.theta_min / M_PI * 180,
                //                       partition.theta_max / M_PI * 180,
                //                       exp,
                //                       var,
                //                       (partition.theta_max -
                //                       partition.theta_min) / M_PI * 180);

                //                L_j_q = alpha_cos * std::cos(partition.u) +
                //                std::cos(80) * std::sin(alpha) *
                //                std::sin(partition.u); U_j_q = L_j_q;

                // 计算整个分区的下界
                float partition_lb =
                        getLB(R_q,
                              R_q2,
                              L_j_q,
                              U_j_q,
                              partition.min_dis,
                              partition.max_dis);

                if (partition_lb > current_threshold) {
                    continue;
                }
                // 将分区加入优先队列
                partition_queue.push(
                        {list_no,
                         m,
                         R_q2,
                         R_q,
                         L_j_q,
                         U_j_q,
                         partition_lb,
                         alpha_cos, theta, R});
            }
            //            t_lb[i] += getmillisecs() - t0;
            double t1 = getmillisecs();
            bool flag = true;
            // 处理分区优先队列
            std::priority_queue<BlockCandidate> block_queue;
            // 一个簇一个簇的计算
            while (!partition_queue.empty() && flag) {
                // 弹出当前最小的分区
                PartitionCandidate part_cand = partition_queue.top();
                partition_queue.pop();

                if (part_cand.lb > current_threshold && count > k) {
                    flag = false;
                    break;
                }
                // 获取该分区
                const PartitionData& partition =
                        partitions_per_cluster_[part_cand.list_no]
                                               [part_cand.partition_id];

                // 创建block优先队列（最小堆）
                //            std::priority_queue<BlockCandidate,
                //            std::vector<BlockCandidate>,
                //            std::greater<BlockCandidate>> block_queue;
                // 计算分区内所有block的LB并放入block队列
                for (size_t b = 0; b < partition.blocks.size(); b++) {
                    const BlockData& block = partition.blocks[b];
                    //                    float alpha_cos_std_block =
                    //                    sum_q_positive * block.std_block;
                    //                    float phi = 0.8f;
                    ////
                    //                    float L_j_q = part_cand.alpha_cos -
                    //                    phi * alpha_cos_std_block; float U_j_q
                    //                    = part_cand.alpha_cos + phi *
                    //                    alpha_cos_std_block;
                    float theta = part_cand.theta;

                    float theta_min = block.theta_min - theta;
                    float theta_max = block.theta_max - theta;
                    //                printf("theta: %.4f, p_theta_min: %.4f, p_theta_max: %.4f, theta_min: %.4f, theta_max: %.4f, R: %.4f\n", theta / M_PI * 180, partition.theta_min / M_PI * 180, partition.theta_max / M_PI * 180,
                    //                       theta_min / M_PI * 180, theta_max / M_PI * 180, R);
                    float L_j_q = 0.0;
                    float U_j_q = 0.0;
                    if (theta_max < 0.0) {
                        L_j_q = part_cand.R * std::cos(theta_min);
                        U_j_q = part_cand.R * std::cos(theta_max);
                    } else if (theta_min < 0.0) {
                        L_j_q = part_cand.R * std::cos(std::min(-theta_min, theta_max));
                        U_j_q = part_cand.R;
                    } else {
                        L_j_q = part_cand.R * std::cos(theta_max);
                        U_j_q = part_cand.R * std::cos(theta_min);
                    }

//                   float phi = 0.68;
//                    L_j_q = part_cand.alpha_cos +  phi * std::sqrt(2.0 - 2.0 * std::cos(block.theta_min)) / std::sqrt(d) * sum_q_positive;
//                    U_j_q = part_cand.alpha_cos +  phi * std::sqrt(2.0 - 2.0 * std::cos(block.theta_max)) / std::sqrt(d) * sum_q_positive;

                    float block_lb =
                            getLB(part_cand.Rq, // 使用分区下界作为参考
                                  part_cand.Rq_2,
//                                  part_cand.L_j_q,
//                                  part_cand.U_j_q,
                                                                    L_j_q,
                                                                    U_j_q,
                                  block.min_dist,
                                  block.max_dist);

                    if (block_lb > current_threshold && count > k) {
                        continue;
                    }
                    double t = getmillisecs();
                    int num_vecs = block.offsets.size();

                    //                    #ifdef __AVX2__
                    ////                    printf("AVX2 is available\n");
                    //                    const float* block_data =
                    //                    block.vectors; size_t v_idx = 0;
                    //
                    //                    // 使用AVX2处理8个向量一组
                    //                    for (; v_idx + 7 < num_vecs; v_idx +=
                    //                    8) {
                    //                        __m256 distances =
                    //                        _mm256_setzero_ps(); const float*
                    //                        vec_ptrs[8] = {
                    //                                block_data + (v_idx + 0) *
                    //                                d, block_data + (v_idx +
                    //                                1) * d, block_data +
                    //                                (v_idx + 2) * d,
                    //                                block_data + (v_idx + 3) *
                    //                                d, block_data + (v_idx +
                    //                                4) * d, block_data +
                    //                                (v_idx + 5) * d,
                    //                                block_data + (v_idx + 6) *
                    //                                d, block_data + (v_idx +
                    //                                7) * d
                    //                        };
                    //
                    ////                        const float* vec_ptrs =
                    /// block_data
                    //                        distances = compute_8_l2s(xi,
                    //                        vec_ptrs, d);
                    //
                    //                        // 处理8个距离结果
                    //                        alignas(32) float dist_arr[8];
                    //                        _mm256_store_ps(dist_arr,
                    //                        distances);
                    //
                    //                        for (int j = 0; j < 8; j++) {
                    //                            if (dist_arr[j] <=
                    //                            current_threshold) {
                    //                                idx_t offset =
                    //                                block.offsets[v_idx + j];
                    //                                idx_t id =
                    //                                invlists->get_single_id(part_cand.list_no,
                    //                                offset);
                    //                                heap_replace_top<Heap>(k,
                    //                                dist_i, label_i,
                    //                                dist_arr[j], id);
                    //                                current_threshold =
                    //                                dist_i[0];
                    //                            }
                    //                        }
                    //                    }
                    //
                    //                    // 处理剩余向量
                    //                    for (; v_idx < num_vecs; v_idx++) {
                    //                        idx_t offset =
                    //                        block.offsets[v_idx]; const float*
                    //                        vec_ptr = block_data + v_idx * d;
                    ////                        const float* vec_ptr = (float*)
                    /// invlists->get_single_code(part_cand.list_no, offset);
                    //                        float dis;
                    //                        if (metric_type ==
                    //                        METRIC_INNER_PRODUCT) {
                    //                            dis = -fvec_inner_product(xi,
                    //                            vec_ptr, d);
                    //                        } else {
                    //                            dis = fvec_L2sqr(xi, vec_ptr,
                    //                            d);
                    //                        }
                    //
                    //                        if (dis <= current_threshold) {
                    //                            idx_t id =
                    //                            invlists->get_single_id(part_cand.list_no,
                    //                            offset);
                    //                            heap_replace_top<Heap>(k,
                    //                            dist_i, label_i, dis, id);
                    //                            current_threshold = dist_i[0];
                    //                        }
                    //                    }
                    //                    #else
#ifdef __AVX512F__
                    //                    printf("__AVX512F__\n");
                    const float* block_data = block.vectors;
                    const float* block_16_data = block.bbs_16;
                    const float* block_8_data = block.bbs_8;
                    size_t v_idx = 0;

                    // 使用AVX2处理8个向量一组
                    int b16_size = 0;
                    for (; v_idx + 15 < num_vecs; v_idx += 16) {
                        __m512 distances = _mm512_setzero_ps();
                        const float* vec_ptrs =
                                block_16_data + b16_size * 16 * d;
                        b16_size++;
                        distances = compute_16_l2s_contiguous(xi, vec_ptrs, d);

                        // 处理8个距离结果
                        alignas(64) float dist_arr[16];
                        _mm512_store_ps(dist_arr, distances);

                        for (int j = 0; j < 16; j++) {
                            if (dist_arr[j] <= current_threshold) {
                                idx_t offset = block.offsets[v_idx + j];
                                idx_t id = invlists->get_single_id(
                                        part_cand.list_no, offset);
                                heap_replace_top<Heap>(
                                        k, dist_i, label_i, dist_arr[j], id);
                                current_threshold = dist_i[0];
                            }
                        }
                    }
                    int b8_size = 0;
                    // avx2处理剩余向量
                    for (; v_idx + 7 < num_vecs; v_idx += 8) {
                        __m256 distances = _mm256_setzero_ps();
                        const float* vec_ptrs = block_8_data + b8_size * 8 * d;
                        distances = compute_8_l2s_contiguous(xi, vec_ptrs, d);
                        b8_size++;
                        // 处理8个距离结果
                        alignas(32) float dist_arr[8];
                        _mm256_store_ps(dist_arr, distances);

                        for (int j = 0; j < 8; j++) {
                            if (dist_arr[j] <= current_threshold) {
                                idx_t offset = block.offsets[v_idx + j];
                                idx_t id = invlists->get_single_id(
                                        part_cand.list_no, offset);
                                heap_replace_top<Heap>(
                                        k, dist_i, label_i, dist_arr[j], id);
                                current_threshold = dist_i[0];
                            }
                        }
                    }
                    int reminder = 0;
                    // 处理剩余向量
                    for (; v_idx < num_vecs; v_idx++) {
                        idx_t offset = block.offsets[v_idx];
                        const float* vec_ptr = block_data + reminder * d;
                        //                        const float* vec_ptr =
                        //                        (float*)
                        //                        invlists->get_single_code(part_cand.list_no,
                        //                        offset);
                        //                            float dis;
                        //                            if (metric_type ==
                        //                            METRIC_INNER_PRODUCT) {
                        //                                dis =
                        //                                -fvec_inner_product(xi,
                        //                                vec_ptr, d);
                        //                            } else {
                        //                                dis = fvec_L2sqr(xi,
                        //                                vec_ptr, d);
                        //                            }
                        float dis =
                                compute_one2one_16_contiguous(xi, vec_ptr, d);
                        reminder++;
                        if (dis <= current_threshold) {
                            idx_t id = invlists->get_single_id(
                                    part_cand.list_no, offset);
                            heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                            current_threshold = dist_i[0];
                        }
                    }
#else
                    //                    printf("AVX2 is not available\n");
                    for (size_t v_idx = 0; v_idx < num_vecs; v_idx++) {
                        idx_t offset = block.offsets[v_idx];
                        //                    idx_t offset = offsets[v_idx];
                        const float* vec_ptr =
                                (float*)invlists->get_single_code(
                                        part_cand.list_no, offset);
                        //                    const float* vec_ptr = (const
                        //                    float*)invlists->get_single_code(block_cand.list_no,
                        //                    offset);
                        float dis;
                        if (metric_type == METRIC_INNER_PRODUCT) {
                            dis = -fvec_inner_product(xi, vec_ptr, d);
                        } else {
                            dis = fvec_L2sqr(xi, vec_ptr, d);
                        }

                        if (dis <= current_threshold) {
                            idx_t id = invlists->get_single_id(
                                    part_cand.list_no, offset);
                            heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                            current_threshold = dist_i[0];
                        }
                    }
#endif
                    //                    #endif
                    count += num_vecs;
                    t_compute += getmillisecs() - t;
                    if (++iter_items > max_iter) {
                        flag = false;
                        break;
                    }
                }
                //                t_total[i] = t_ann[i] + t_lb[i];
            }
            //            t_ann[i] += getmillisecs() - t1;
            //            t_total[i] += getmillisecs() - t0;
        }
        //        ndis[i] = count;
        //        times[i] =  getmillisecs() - t0;
        // 最终排序结果
        heap_reorder<Heap>(k, dist_i, label_i);
#pragma omp atomic
        indexIVF_stats.ndis += count;
        indexIVF_stats.search_time += t_compute;
        //        total[i] = getmillisecs() - t_s;
    }

    double ratio_lb = 0.0;
    double ratio_ann = 0.0;
    double ratio_lb_ann = 0.0;

    double sum_lb = 0.0;
    double sum_ann = 0.0;
    double sum_lb_ann = 0.0;
    double sum = 0.0;
    //    for (int i = 0; i < n; i++) {
    //        ratio_lb += t_lb[i] / t_total[i];
    //        ratio_ann += t_ann[i] / t_total[i];
    //        ratio_lb_ann += t_lb[i] / t_ann[i];
    //
    //        sum_lb += t_lb[i];
    //        sum_ann += t_ann[i];
    //        sum_lb_ann += t_total[i];
    //        sum += total[i];
    //    }

    //    printf("ratio_lb:%f, ratio_ann:%f, ratio_lb_ann:%f\n",
    //           ratio_lb / n,
    //           ratio_ann / n,
    //           ratio_lb_ann / n);
    //    printf("sum_lb:%f, sum_ann:%f, t_total:%f, t_total_2:%f\n",
    //           sum_lb,
    //           sum_ann,
    //           sum_lb_ann,
    //           sum);
//    indexIVF_stats.search_time += getmillisecs() - t_start;
}

void IndexPartitionBlockFlatSIMD::query_by_mini_blocks(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int M,
        int block_size,
        float iter_factor,
        const IVFSearchParameters* params,
        std::vector<double>& times,
        std::vector<double>& ndis) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(M > 0);
    double t_start = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());
    indexIVF_stats.quantization_time += getmillisecs() - t_start;
    using Heap = CMax<float, idx_t>;
    // 并行处理每个查询
    //    int batch = 10000;
    //    int batch_num = (n + batch - 1) / batch;
    //    for (int bat = 0; bat < batch_num; bat ++) {
    //        int start = bat * batch;
    //        int end = (bat + 1) * batch;
//    double* t_lb = new double[n];
//    double* t_sort = new double[n];
//    double* t_ann = new double[n];
//    double* t_total = new double[n];
#pragma omp parallel for if (n > 10) schedule(dynamic)
    for (idx_t i = 0; i < n; i++) {
        // 当前查询向量
        double t0 = getmillisecs();
        const float* xi = x + i * d;
        float* dist_i = distances + i * k;
        idx_t* label_i = labels + i * k;

        // 初始化结果堆
        heap_heapify<Heap>(k, dist_i, label_i);
        float current_threshold = dist_i[0]; // 当前最大距离
        //        printf("current_threshold: %.6f\n", current_threshold);
        int count = 0;

        // 分区优先队列（最小堆）
        //        std::priority_queue<PartitionCandidate,
        //        std::vector<PartitionCandidate>,
        //        std::greater<PartitionCandidate>> partition_queue;
        int block_num = 0;
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;
            block_num += cluster_min_max[list_no].second;
        }

        int max_iter = (int)((float)block_num * iter_factor);
        max_iter = std::max(1, max_iter);
        int iter_items = 0;
        // 遍历每个簇，收集所有候选分区
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;

            // 获取该簇的分区数据
            const auto& partitions = partitions_per_cluster_[list_no];
            std::vector<float> centroid(d);
            //            auto* centroid = new float[d];
            quantizer->reconstruct(list_no, centroid.data());
            // 查询向量到簇心的距离
            float R_q = 0.0f;
            float R_q2 = coarse_dis[i * nprobe + j];
            R_q = std::sqrt(R_q2);

            float min = cluster_min_max[list_no].first.first;
            float max = cluster_min_max[list_no].first.second;

            if (R_q > max) {
                float lb = R_q - max;
                lb *= lb;
                if (lb > current_threshold && count > k) {
                    break;
                }
            }

            int m = 0;
            const float* p_c = cluster_partition_centroids[list_no];
            float* p_alpha;
            float* alphas = new float[partitions.size()];
#ifdef __AVX512F__
            int b16_size = 0;
            if (R_q > 1e-8f) {
                for (; m + 15 < partitions.size(); m += 16) {
                    __m512 distances = _mm512_setzero_ps();
                    const float* vec_ptrs = p_c + b16_size * 16 * d;
                    b16_size++;
                    distances = ip_16_contiguous(
                            xi, centroid.data(), vec_ptrs, d, R_q);
                    // 处理8个距离结果
                    alignas(64) float dist_arr[16];
                    _mm512_store_ps(dist_arr, distances);
                    for (int i = m; i < m + 16; i++) {
                        alphas[i] = dist_arr[i - m] / R_q;
                    }
                }
            }
#endif

            if (m < partitions.size()) {
                // 计算查询向量与簇心的残差
                std::vector<float> xi_centroid_unit(d);
                for (int jj = 0; jj < d; jj++) {
                    xi_centroid_unit[jj] = xi[jj] - centroid[jj];
                }

                for (; m < partitions.size(); m++) {
                    const PartitionData& partition = partitions[m];
                    // 获取该簇的单位向量e
                    const std::vector<float>& e_vec = partition.centroid;
                    // 计算查询向量q在e方向上的投影
                    float alpha_cos = 0.0f;
                    if (R_q > 1e-8f) {
#ifdef __AVX512F__
                        alpha_cos =
                                ip_one2one_16_contiguous(xi, e_vec.data(), d) /
                                R_q;
#else
                        alpha_cos = fvec_inner_product(
                                            xi_centroid_unit.data(),
                                            e_vec.data(),
                                            d) /
                                R_q;
#endif
                    }
                }
            }

            std::priority_queue<PartitionCandidate> partition_queue;
            // 遍历该簇的所有分区
            for (int m = 0; m < partitions.size(); m++) {
                const PartitionData& partition = partitions[m];
                // 获取该簇的单位向量e
                const std::vector<float>& e_vec = partition.centroid;
                // 计算查询向量q在e方向上的投影
                float alpha_cos = 0.0f;
                //                if (R_q > 1e-8f) {
                ////                    #ifdef __AVX512F__
                ////                    alpha_cos = ip_one2one_16_contiguous(xi,
                /// centroid.data(), e_vec.data(), d) / R_q; / #endif
                //                    alpha_cos =
                //                    fvec_inner_product(xi_centroid_unit.data(),
                //                    e_vec.data(), d) / R_q;
                //                }

                //                alpha_cos = std::max(-1.0f, std::min(1.0f,
                //                alpha_cos));
                alpha_cos = alphas[m];
                float alpha = std::acos(alpha_cos);
                float L_j_q, U_j_q;

                float delta = (partition.theta_max - partition.theta_min);
                angle_bounds_simd(
                        alpha,
                        L_j_q,
                        U_j_q,
                        delta,
                        partition.theta_min,
                        partition.theta_max);

                //                float delta = partition.gauss_delta;
                //                angle_bounds_simd(alpha, L_j_q, U_j_q, delta,
                //                delta);

                // 计算delta和角度边界
                //                float u1 = (partition.theta_max -
                //                partition.theta_min) / 4.0f; float u2 =
                //                (partition.theta_max + partition.theta_min)
                //                / 2.0f; float delta = std::sqrt((u1 * u1 + u2
                //                * u2) / 2.0f); angle_bounds_simd(alpha, L_j_q,
                //                U_j_q, delta, delta);

                //                float delta = partition.gauss_delta;
                //                angle_bounds_simd(alpha, L_j_q, U_j_q, delta,
                //                partition.theta_min, partition.theta_max);

                //                  float u = alpha_cos * std::cos(partition.u);
                //                  float delta_u = std::sin(alpha) *
                //                  partition.avg_sin / std::sqrt(2.0f);
                //                //                delta = std::cos(alpha) *
                //                std::cos(partition.u) + 1.64 * std::sin(alpha)
                //                * std::sin(partition.u) / std::sqrt(2.0f);
                //                  L_j_q = u - delta_u;
                //                  U_j_q = u + delta_u;

                //                printf("L_j_q: %.6f, U_j_q: %.6f, alpha: %.6f,
                //                theta_min: %.6f, theta_max: %.6f, delta:
                //                %.6f\n", L_j_q, U_j_q, alpha / M_PI * 180,
                //                partition.theta_min / M_PI * 180,
                //                partition.theta_max / M_PI * 180,
                //                (partition.theta_max - partition.theta_min) /
                //                M_PI * 180);
                // 计算整个分区的下界
                float partition_lb =
                        getLB(R_q,
                              R_q2,
                              L_j_q,
                              U_j_q,
                              partition.min_dis,
                              partition.max_dis);

                if (partition_lb > current_threshold) {
                    continue;
                }
                // 将分区加入优先队列
                partition_queue.push(
                        {list_no, m, R_q2, R_q, L_j_q, U_j_q, partition_lb});
            }
            bool flag = true;
            // 处理分区优先队列
            std::priority_queue<BlockCandidate> block_queue;
            // 一个簇一个簇的计算
            while (!partition_queue.empty() && flag) {
                // 弹出当前最小的分区
                PartitionCandidate part_cand = partition_queue.top();
                partition_queue.pop();

                if (part_cand.lb > current_threshold && count > k) {
                    flag = false;
                    break;
                }
                // 获取该分区
                const PartitionData& partition =
                        partitions_per_cluster_[part_cand.list_no]
                                               [part_cand.partition_id];

                // 创建block优先队列（最小堆）
                //            std::priority_queue<BlockCandidate,
                //            std::vector<BlockCandidate>,
                //            std::greater<BlockCandidate>> block_queue;
                // 计算分区内所有block的LB并放入block队列
                for (size_t b = 0; b < partition.blocks.size(); b++) {
                    const BlockData& block = partition.blocks[b];
                    float block_lb =
                            getLB(part_cand.Rq, // 使用分区下界作为参考
                                  part_cand.Rq_2,
                                  part_cand.L_j_q,
                                  part_cand.U_j_q,
                                  block.min_dist,
                                  block.max_dist);

                    if (block_lb > current_threshold && count > k) {
                        break;
                    }

                    int num_vecs = block.offsets.size();
                    for (size_t v_idx = 0; v_idx < num_vecs; v_idx++) {
                        idx_t offset = block.offsets[v_idx];
                        idx_t id = invlists->get_single_id(
                                part_cand.list_no, offset);
                        heap_replace_top<Heap>(
                                k, dist_i, label_i, block_lb, id);
                    }
                    current_threshold = dist_i[0];
                    //                    #endif
                    count += num_vecs;
                    if (++iter_items > max_iter) {
                        flag = false;
                        break;
                    }
                }
            }
        }
        //        ndis[i] = count;
        //        times[i] =  getmillisecs() - t0;
        // 最终排序结果
        heap_reorder<Heap>(k, dist_i, label_i);
#pragma omp atomic
        indexIVF_stats.ndis += count;
    }
    indexIVF_stats.search_time += getmillisecs() - t_start;
}

// 一个簇一个簇的搜索
void IndexPartitionBlockFlatSIMD::query_by_cluster_nprobe(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int M,
        int block_size,
        float iter_factor,
        const IVFSearchParameters* params,
        std::vector<double>& times,
        std::vector<double>& ndis) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(M > 0);
    double t_start = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());
    indexIVF_stats.quantization_time += getmillisecs() - t_start;
    using Heap = CMax<float, idx_t>;
    // 并行处理每个查询
    //    int batch = 10000;
    //    int batch_num = (n + batch - 1) / batch;
    //    for (int bat = 0; bat < batch_num; bat ++) {
    //        int start = bat * batch;
    //        int end = (bat + 1) * batch;
    double* t_lb = new double[n];
    double* t_sort = new double[n];
    double* t_ann = new double[n];
    double* t_total = new double[n];
#pragma omp parallel for if (n > 10)
    for (idx_t i = 0; i < n; i++) {
        // 当前查询向量
        double t0 = getmillisecs();
        const float* xi = x + i * d;
        float* dist_i = distances + i * k;
        idx_t* label_i = labels + i * k;

        // 初始化结果堆
        heap_heapify<Heap>(k, dist_i, label_i);
        float current_threshold = dist_i[0]; // 当前最大距离
        //        printf("current_threshold: %.6f\n", current_threshold);

        // 存储候选block
        struct CandidateBlock {
            idx_t list_no;
            int partition_id;
            size_t block_id;
            float lb; // 下界
        };
        std::vector<CandidateBlock> candidate_blocks;
        // 第二步：遍历每个簇，收集所有候选block
        //        #pragma omp parallel for if (n > 10)
        int count = 0;
        int filter_count = 0;
        int iter_times = 0;
        int block_num = 0;
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;
            block_num += cluster_min_max[list_no].second;
        }

        int max_iter = (int)((float)block_num * iter_factor);
        max_iter = std::max(1, max_iter);
        bool flag = true;
        for (size_t j = 0; j < nprobe && flag; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;

            // 获取该簇的分区数据
            const auto& partitions = partitions_per_cluster_[list_no];
            std::vector<float> centroid(d);
            quantizer->reconstruct(list_no, centroid.data());
            // 查询向量到簇心的距离
            float R_q = 0.0f;
            float R_q2 = 0.0f;
            R_q2 = coarse_dis[i * nprobe + j];
            R_q = std::sqrt(R_q2);

            float min = cluster_min_max[list_no].first.first;
            float max = cluster_min_max[list_no].first.second;

            if (R_q > max) {
                float lb = R_q - max;
                lb *= lb;
                if (lb > current_threshold && count > k) {
                    filter_count++;
                    break;
                }
            }

            float* xi_centroid_unit = new float[d];
            for (int jj = 0; jj < d; jj++) {
                xi_centroid_unit[jj] = xi[jj] - centroid[jj];
            }

            // 这里计算每个partition的边界，并放入优先队列中。

            for (int m = 0; m < partitions.size() && flag; m++) {
                const PartitionData& partition = partitions[m];
                // 获取该簇的单位向量e
                const std::vector<float>& e_vec = partition.centroid;
                // 计算查询向量q在e方向上的投影
                float alpha = 0.0f;
                float alpha_cos = 0.0f;
                if (R_q > 1e-8f) {
                    alpha_cos = fvec_inner_product(
                                        xi_centroid_unit, e_vec.data(), d) /
                            R_q;
                }

                alpha_cos = std::max(
                        -1.0f, std::min(1.0f, alpha_cos)); // 确保在[-1,1]范围内
                alpha = std::acos(alpha_cos);
                float L_j_q, U_j_q;
                // block_min
                //                float delta = partition.theta_min;
                // block_radius
                float delta = (partition.theta_max - partition.theta_min);
                angle_bounds_simd(alpha, L_j_q, U_j_q, delta, delta, delta);

                //                float u1 = (partition.theta_max -
                //                partition.theta_min) / 4.0f; float u2 =
                //                (partition.theta_max + partition.theta_min)
                //                / 2.0f; delta = std::sqrt((u1 * u1 + u2 * u2)
                //                / 2.0f); float delta = partition.gauss_delta;
                //                angle_bounds_simd(alpha, L_j_q, U_j_q, delta,
                //                delta);

                //

                //
                //                float delta_d = std::abs(alpha -
                //                std::acos(alpha_cos * std::cos(partition.u)));
                //                //float delta_d = std::acos(alpha_cos *
                //                std::cos(partition.u) + std::sin(alpha) *
                //                std::sin(partition.u) * std::cos(M_PI * (1.0 -
                //                0.8))); delta = delta_d + std::sin(alpha) *
                //                partition.delta_phi;
                //                //                delta = delta_d +
                //                partition.delta_phi; angle_bounds_simd(alpha,
                //                L_j_q, U_j_q, delta, delta);

                // 高斯分布 sin
                //                float phi_1 = 1.0;
                //                float mean = std::cos(alpha) *
                //                std::cos(partition.u); delta = phi_1 *
                //                std::sin(alpha) * partition.avg_sin /
                //                std::sqrt(2.0f); L_j_q = mean - delta; U_j_q =
                //                mean + delta;

                //                float g = 0.8;
                //                delta = std::sin(alpha) * partition.avg_sin *
                //                std::sqrt((1 - std::cos(M_PI * g)) / 2); L_j_q
                //                = alpha_cos * std::cos(partition.u) - delta;
                //                U_j_q = alpha_cos * std::cos(partition.u) +
                //                delta;

                float theta_min = partition.min_dis;
                float theta_max = partition.max_dis;
                float lb = getLB(R_q, R_q2, L_j_q, U_j_q, theta_min, theta_max);

                if (lb > current_threshold && count > k) {
                    continue;
                }

                const auto& blocks = partition.blocks;
                for (size_t b = 0; b < blocks.size() && flag; b++) {
                    const auto& block = blocks[b];
                    lb =
                            getLB(R_q,
                                  R_q2,
                                  L_j_q,
                                  U_j_q,
                                  block.min_dist,
                                  block.max_dist);
                    // 添加到候选列表
                    candidate_blocks.push_back({list_no, m, b, lb});
                }

                std::sort(
                        candidate_blocks.begin(),
                        candidate_blocks.end(),
                        [](const CandidateBlock& a, const CandidateBlock& b) {
                            return a.lb < b.lb;
                        });

                for (const auto& cand : candidate_blocks) {
                    //            iter_times++;
                    // 如果下界大于当前阈值，跳过该block
                    //            float tmp = sqrt(current_threshold) / dd;
                    //            printf(  "cand.lb: %.6f, current_threshold:
                    //            %.6f, tmp:
                    //            %.6f\n", cand.lb, current_threshold, tmp); if
                    //            (cand.lb > current_threshold || ++iter_times
                    //            >= max_iter) {
                    if (cand.lb > current_threshold) {
                        //            if (cand.lb > current_threshold) {
                        break;
                    }

                    // 获取block数据
                    const auto& block = get_block(
                            cand.list_no, cand.partition_id, cand.block_id);

                    // 获取block内的向量
                    const std::vector<idx_t>& offsets = block.offsets;
                    size_t num_vecs = offsets.size();

                    // 计算block内每个向量的距离
                    //            #pragma omp parallel for schedule(dynamic)
                    for (size_t v_idx = 0; v_idx < num_vecs; v_idx++) {
                        idx_t offset = offsets[v_idx];
                        const float* vec_ptr =
                                (float*)invlists->get_single_code(
                                        cand.list_no, offset);
                        //                reconstruct_from_offset(cand.list_no,
                        //                offset, vec);
                        float dis;
                        if (metric_type == METRIC_INNER_PRODUCT) {
                            dis = -fvec_inner_product(xi, vec_ptr, d);
                        } else {
                            dis = fvec_L2sqr(xi, vec_ptr, d);
                        }
                        //                block_dis[v_idx] = dis;
                        //                idx_t id =
                        //                invlists->get_single_id(cand.list_no,
                        //                offset); block_ids[v_idx] = id;
                        if (dis <= current_threshold) {
                            idx_t id = invlists->get_single_id(
                                    cand.list_no, offset);
                            heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                            current_threshold = dist_i[0];
                            //                        printf("dis: %.6f,
                            //                        current_threshold:
                            //                        %.6f\n", dis,
                            //                        current_threshold);
                        }
                    }
                    //            heap_addn<Heap>(k, dist_i, label_i,
                    //            block_dis.data(), block_ids.data(), num_vecs);
                    //            current_threshold = dist_i[0];
                    count += num_vecs;
                    //            heap_addn<>()
                    if (++iter_times >= max_iter) {
                        flag = false;
                        break;
                    }
                }
                candidate_blocks.clear();
            }
            //            double t1 = getmillisecs() - t0;
            //
            //            max_iter = candidate_blocks.size() * iter_factor;
            //            max_iter = std::max(1, max_iter);
            //            // 第三步：按照下界排序
            //            if (iter_factor <= 0.6) {
            //                const size_t top_k =
            //                        std::min(candidate_blocks.size(),
            //                        size_t(max_iter));
            //                auto cmp = [](const CandidateBlock& a, const
            //                CandidateBlock& b) {
            //                    return a.lb < b.lb; //
            //                    根据实际需求调整比较逻辑
            //                };
            //                std::partial_sort(
            //                        candidate_blocks.begin(),
            //                        candidate_blocks.begin() + top_k,
            //                        candidate_blocks.end(),
            //                        cmp);
            //            } else {
            //                std::sort(
            //                        candidate_blocks.begin(),
            //                        candidate_blocks.end(),
            //                        [](const CandidateBlock& a, const
            //                        CandidateBlock& b) {
            //                            return a.lb < b.lb;
            //                        });
            //            }
            //
            //            double t2 = getmillisecs() - t0;
            //            // 第四步：遍历候选block
            //            //        std::vector<float> block_dis(block_size);
            //            //        std::vector<idx_t> block_ids(block_size);
            //            for (const auto& cand : candidate_blocks) {
            //                //            iter_times++;
            //                // 如果下界大于当前阈值，跳过该block
            //                //            float tmp = sqrt(current_threshold)
            //                / dd;
            //                //            printf(  "cand.lb: %.6f,
            //                current_threshold: %.6f, tmp:
            //                //            %.6f\n", cand.lb, current_threshold,
            //                tmp); if (cand.lb
            //                //            > current_threshold || ++iter_times
            //                >= max_iter) { if (cand.lb > current_threshold ) {
            //                    //            if (cand.lb > current_threshold)
            //                    { break;
            //                }
            //
            //                // 获取block数据
            //                const auto& block =
            //                        get_block(cand.list_no, cand.partition_id,
            //                        cand.block_id);
            //
            //                // 获取block内的向量
            //                const std::vector<idx_t>& offsets = block.offsets;
            //                size_t num_vecs = offsets.size();
            //
            //
            //                // 计算block内每个向量的距离
            //
            //                //            #pragma omp parallel for
            //                schedule(dynamic) for (size_t v_idx = 0; v_idx <
            //                num_vecs; v_idx++) {
            //                    idx_t offset = offsets[v_idx];
            //                    const float* vec_ptr = (float*)
            //                    invlists->get_single_code(cand.list_no,
            //                    offset);
            //                    // reconstruct_from_offset(cand.list_no,
            //                    offset, vec); float dis; if (metric_type ==
            //                    METRIC_INNER_PRODUCT) {
            //                        dis = -fvec_inner_product(xi, vec_ptr, d);
            //                    } else {
            //                        dis = fvec_L2sqr(xi, vec_ptr, d);
            //                    }
            //                    //                block_dis[v_idx] = dis;
            //                    //                idx_t id =
            //                    invlists->get_single_id(cand.list_no, offset);
            //                    //                block_ids[v_idx] = id;
            //                    if (dis <= current_threshold) {
            //                        idx_t id =
            //                        invlists->get_single_id(cand.list_no,
            //                        offset); heap_replace_top<Heap>(k, dist_i,
            //                        label_i, dis, id); current_threshold =
            //                        dist_i[0];
            ////                        printf("dis: %.6f, current_threshold:
            ///%.6f\n", dis, current_threshold);
            //                    }
            //                }
            //                //            heap_addn<Heap>(k, dist_i, label_i,
            //                block_dis.data(), block_ids.data(), num_vecs);
            //                //            current_threshold = dist_i[0];
            //                count += num_vecs;
            //                //            heap_addn<>()
            //            }
            //            candidate_blocks.clear();
            delete[] xi_centroid_unit;
            // xi_centroid = nullptr;  // 可选：防止野指针
        }
        double t3 = getmillisecs() - t0;
        //        times[i] = t3;
        //        ndis[i] = count;
        // 最终排序结果
        heap_reorder<Heap>(k, dist_i, label_i);
#pragma omp atomic
        indexIVF_stats.ndis += count;
    }
    indexIVF_stats.search_time += getmillisecs() - t_start;
}

void IndexPartitionBlockFlatSIMD::query_by_cluster(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int M,
        int block_size,
        float iter_factor,
        const IVFSearchParameters* params,
        std::vector<double> times,
        std::vector<double> ndis) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(M > 0);
    double t_start = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());
    indexIVF_stats.quantization_time += getmillisecs() - t_start;
    using Heap = CMax<float, idx_t>;
    // 并行处理每个查询
    //    int batch = 10000;
    //    int batch_num = (n + batch - 1) / batch;
    //    for (int bat = 0; bat < batch_num; bat ++) {
    //        int start = bat * batch;
    //        int end = (bat + 1) * batch;
    double* t_lb = new double[n];
    double* t_sort = new double[n];
    double* t_ann = new double[n];
    double* t_total = new double[n];
#pragma omp parallel for if (n > 10)
    for (idx_t i = 0; i < n; i++) {
        int max_iter = 500;
        // 当前查询向量
        const float* xi = x + i * d;
        float* dist_i = distances + i * k;
        idx_t* label_i = labels + i * k;

        // 初始化结果堆
        heap_heapify<Heap>(k, dist_i, label_i);
        float current_threshold = dist_i[0]; // 当前最大距离
        //        printf("current_threshold: %.6f\n", current_threshold);
        // 存储候选block
        struct CandidateBlock {
            idx_t list_no;
            int partition_id;
            size_t block_id;
            float lb; // 下界
        };
        std::vector<CandidateBlock> candidate_blocks;

        // 第二步：遍历每个簇，收集所有候选block
        double t0 = getmillisecs();
        //        #pragma omp parallel for if (n > 10)
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;

            // 获取该簇的分区数据
            const auto& partitions = partitions_per_cluster_[list_no];
            std::vector<float> centroid(d);
            quantizer->reconstruct(list_no, centroid.data());
            // 查询向量到簇心的距离
            float R_q = 0.0f;
            float R_q2 = 0.0f;
            R_q2 = coarse_dis[i * nprobe + j];
            R_q = std::sqrt(R_q2);

            float* xi_centroid_unit = new float[d];
            for (int jj = 0; jj < d; jj++) {
                xi_centroid_unit[jj] = xi[jj] - centroid[jj];
            }
            for (int m = 0; m < partitions.size(); m++) {
                const PartitionData& partition = partitions[m];
                // 获取该簇的单位向量e
                const std::vector<float>& e_vec = partition.centroid;
                // 计算查询向量q在e方向上的投影
                //                float alpha =
                //                        fvec_inner_product(xi_centroid_unit,
                //                        e_vec.data(), d);
                float alpha = 0.0f;
                float alpha_cos = 0.0f;
                if (R_q > 1e-8f) {
                    alpha_cos = fvec_inner_product(
                                        xi_centroid_unit, e_vec.data(), d) /
                            R_q;
                }
                //                float alpha =
                //                        fvec_inner_product(xi_centroid_unit,
                //                        e_vec.data(), d);

                alpha_cos = std::max(
                        -1.0f, std::min(1.0f, alpha_cos)); // 确保在[-1,1]范围内
                alpha = std::acos(alpha_cos);
                float L_j_q, U_j_q;
                // block_min
                float delta = partition.theta_min;
                //                angle_bounds_simd(alpha, delata, L_j_q, U_j_q,
                //                delta, delta); float delta =
                //                partition.theta_max;
                //                float u1 = (partition.theta_max
                //                -partition.theta_min) / 4.0f; float u2
                //                =(partition.theta_max +
                //                partition.theta_min)/ 2.0f; delta =
                //                std::sqrt((u1 * u1 + u2 * u2)/ 2.0f);
                //                angle_bounds_simd(alpha, L_j_q,U_j_q, delta,
                //                partition.theta_min, partition.theta_max);

                //                 block_radius
                delta = (partition.theta_max - partition.theta_min);
                angle_bounds_simd(
                        alpha,
                        L_j_q,
                        U_j_q,
                        delta,
                        partition.theta_min,
                        partition.theta_max);
                //
                //                float delta_d = std::abs(alpha -
                //                std::acos(alpha_cos * std::cos(partition.u)));
                ////                float delta_d = std::acos(alpha_cos *
                /// std::cos(partition.u) + std::sin(alpha) *
                /// std::sin(partition.u) * std::cos(M_PI * (1.0 - 0.8)));
                //                delta = delta_d + std::sin(alpha) *
                //                partition.delta_phi;
                ////                delta = delta_d + partition.delta_phi;

                // 高斯分布 sin
                //                float phi_1 = 1.28;
                //                float mean = std::cos(alpha) *
                //                std::cos(partition.u); delta = phi_1 *
                //                std::sin(alpha) * partition.avg_sin /
                //                std::sqrt(2.0f); L_j_q = mean - delta; U_j_q =
                //                mean + delta;

                //                float g = 0.8;
                //                delta = std::sin(alpha) * partition.avg_sin *
                //                std::sqrt((1 - std::cos(M_PI * g)) / 2); L_j_q
                //                = alpha_cos * std::cos(partition.u) - delta;
                //                U_j_q = alpha_cos * std::cos(partition.u) +
                //                delta;

                //                angle_bounds_simd(alpha, delata, L_j_q, U_j_q,
                //                delta, delta); float u = std::cos(alpha) *
                //                std::cos(partition.u); float delta_u =
                //                std::sin(alpha) * partition.avg_sin /
                //                std::sqrt(2.0f);
                ////                delta = std::cos(alpha) *
                /// std::cos(partition.u) + 1.64 * std::sin(alpha) *
                /// std::sin(partition.u) / std::sqrt(2.0f);
                //                L_j_q = u - delta_u;
                //                U_j_q = u + delta_u;

                //                printf("alpha: %.6f, u: %.6f, delta_u: %.6f,
                //                L_j_q: %.6f, U_j_q: %.6f\n", alpha, u,
                //                delta_u, L_j_q, U_j_q); delta = delta_d;
                //                printf("delta_d: %.6f,  delta_d%.6f,
                //                delta_phi: %.6f delta_min:%.6f,
                //                delta_radius:%.6f, delta_gauss1:%.6f\n",
                //                delta, delta_d, partition.delta_phi,
                //                partition.theta_min, (partition.theta_max -
                //                partition.theta_min), std::sqrt((u1 * u1 + u2
                //                * u2) / 2.0f)); delta =
                //                std::sin(std::sin(partition.u) * 1.64 /
                //                std::sqrt(2.0f));

                //                delta = std::sqrt(0.5) * partition.avg_sin;

                //                delta = 1.28 * std::sqrt(-2.0 *
                //                std::log(std::cos(partition.u)));

                //                delta = 20.0 / 180 * M_PI;
                //                angle_bounds_simd(alpha, delata, L_j_q, U_j_q,
                //                delta, delta);
                //
                float theta_min = partition.theta_min;
                float theta_max = partition.theta_max;
                //                angle_bounds_4(alpha, L_j_q, U_j_q, theta_min,
                //                theta_max);
                const auto& blocks = partition.blocks;

                for (size_t b = 0; b < blocks.size(); b++) {
                    const auto& block = blocks[b];
                    // 计算s的边界L_j^q和U_j^q
                    //                    ee = 0.0f + block.theta_min;
                    //                    angle_bounds_simd(
                    //                            alpha, delata, L_j_q, U_j_q,
                    //                            ee, ee);

                    float lb =
                            getLB(R_q,
                                  R_q2,
                                  L_j_q,
                                  U_j_q,
                                  block.min_dist,
                                  block.max_dist);

                    // 添加到候选列表
                    //                    #pragma omp critical
                    candidate_blocks.push_back({list_no, m, b, lb});
                }
            }
            delete[] xi_centroid_unit;
            // xi_centroid = nullptr;  // 可选：防止野指针
        }
        double t1 = getmillisecs() - t0;

        max_iter = candidate_blocks.size() * iter_factor;
        max_iter = std::max(1, max_iter);
        // 第三步：按照下界排序
        if (iter_factor <= 0.6) {
            const size_t top_k =
                    std::min(candidate_blocks.size(), size_t(max_iter));
            auto cmp = [](const CandidateBlock& a, const CandidateBlock& b) {
                return a.lb < b.lb; // 根据实际需求调整比较逻辑
            };
            std::partial_sort(
                    candidate_blocks.begin(),
                    candidate_blocks.begin() + top_k,
                    candidate_blocks.end(),
                    cmp);
        } else {
            std::sort(
                    candidate_blocks.begin(),
                    candidate_blocks.end(),
                    [](const CandidateBlock& a, const CandidateBlock& b) {
                        return a.lb < b.lb;
                    });
        }

        double t2 = getmillisecs() - t0;
        // 第四步：遍历候选block
        int iter_times = 0;
        int count = 0;
        //        std::vector<float> block_dis(block_size);
        //        std::vector<idx_t> block_ids(block_size);
        for (const auto& cand : candidate_blocks) {
            //            iter_times++;
            // 如果下界大于当前阈值，跳过该block
            //            float tmp = sqrt(current_threshold) / dd;
            //            printf(  "cand.lb: %.6f, current_threshold: %.6f, tmp:
            //            %.6f\n", cand.lb, current_threshold, tmp); if (cand.lb
            //            > current_threshold || ++iter_times >= max_iter) {
            if (cand.lb > current_threshold || iter_times++ > max_iter) {
                //            if (cand.lb > current_threshold) {
                break;
            }

            // 获取block数据
            const auto& block =
                    get_block(cand.list_no, cand.partition_id, cand.block_id);

            // 获取block内的向量
            const std::vector<idx_t>& offsets = block.offsets;
            size_t num_vecs = offsets.size();

            // 计算block内每个向量的距离

            //            #pragma omp parallel for schedule(dynamic)
            for (size_t v_idx = 0; v_idx < num_vecs; v_idx++) {
                idx_t offset = offsets[v_idx];
                const float* vec_ptr =
                        (float*)invlists->get_single_code(cand.list_no, offset);
                //                reconstruct_from_offset(cand.list_no, offset,
                //                vec);
                float dis;
                if (metric_type == METRIC_INNER_PRODUCT) {
                    dis = -fvec_inner_product(xi, vec_ptr, d);
                } else {
                    dis = fvec_L2sqr(xi, vec_ptr, d);
                }

                //                block_dis[v_idx] = dis;
                //                idx_t id =
                //                invlists->get_single_id(cand.list_no, offset);
                //                block_ids[v_idx] = id;
                if (dis <= current_threshold) {
                    idx_t id = invlists->get_single_id(cand.list_no, offset);
                    heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                    current_threshold = dist_i[0];
                }
            }

            //            heap_addn<Heap>(k, dist_i, label_i, block_dis.data(),
            //            block_ids.data(), num_vecs); current_threshold =
            //            dist_i[0];
            count += num_vecs;
            //            heap_addn<>()
        }

        double t3 = getmillisecs() - t0;
        //        printf("query no: %d, block_size: %d, iter_times: %d,  t1: %f,
        //        t2: %f, t3: %f\n", i, candidate_blocks.size(), iter_times, t1,
        //        t2 - t1, t3 - t2);
        //        *(t_lb + i) =  t1;
        //        *(t_sort + i) = t2 - t1;
        //        *(t_ann + i) = t3 - t2;
        //        *(t_total + i) = t3;
        //        t_lb += t1;
        //        t_sort += t2 - t1;
        //        t_ann += t3 - t2;
        // 最终排序结果
        heap_reorder<Heap>(k, dist_i, label_i);
#pragma omp atomic
        //        std::lock_guard<std::mutex> lock(stats_mutex);
        indexIVF_stats.ndis += count;
        //                printf("query no: %d, [block_size]: %zu, [iter_times]:
        //                %d\n",
        //                       i,
        //                       candidate_blocks.size(),
        //                       iter_times);
    }

    //    for (int i = 0; i < n; i++) {
    //        printf("t_lb:  %lf, t_sort: %lf, t_ann: %lf, t_total: %lf, rate:
    //        %lf\n", t_lb[i], t_sort[i], t_ann[i], t_total[i], (t_lb[i] +
    //        t_sort[i]) / t_ann[i]);
    //    }
    //    printf("t_lb: %f, t_sort: %f, t_ann: %f, rate: %f\n", t_lb, t_sort,
    //    t_ann, (t_lb + t_sort) / t_ann); printf("t_lb: %f, t_sort: %f, t_ann:
    //    %f, rate: %f\n", t_lb, t_sort, t_ann, (t_lb + t_sort) / t_ann);
    indexIVF_stats.search_time += getmillisecs() - t_start;
}

void IndexPartitionBlockFlatSIMD::query_by_cluster_raw(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int M,
        int block_size,
        float iter_factor,
        const IVFSearchParameters* params,
        std::vector<double> times,
        std::vector<double> ndis) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(M > 0);
    double t0 = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());

    using Heap = CMax<float, idx_t>;
    // 并行处理每个查询
    float t_lb = 0.0;
    float t_sort = 0.0;
    float t_ann = 0.0;
#pragma omp parallel for if (n > 10) reduction(+ : t_lb, t_sort, t_ann)
    for (idx_t i = 0; i < n; i++) {
        int max_iter = 500;
        // 当前查询向量
        const float* xi = x + i * d;
        float* dist_i = distances + i * k;
        idx_t* label_i = labels + i * k;

        // 初始化结果堆
        heap_heapify<Heap>(k, dist_i, label_i);
        float current_threshold = dist_i[0]; // 当前最大距离
        //        printf("current_threshold: %.6f\n", current_threshold);
        // 存储候选block
        struct CandidateBlock {
            idx_t list_no;
            int partition_id;
            size_t block_id;
            float lb; // 下界
        };
        std::vector<CandidateBlock> candidate_blocks;

        // 第二步：遍历每个簇，收集所有候选block
        double t0 = getmillisecs();
        //        #pragma omp parallel for if (n > 10)
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;

            // 获取该簇的分区数据
            const auto& partitions = partitions_per_cluster_[list_no];
            std::vector<float> centroid(d);
            quantizer->reconstruct(list_no, centroid.data());
            // 查询向量到簇心的距离
            float R_q = 0.0f;
            float R_q2 = 0.0f;
            R_q2 = fvec_norm_L2sqr(xi, d);
            R_q = std::sqrt(R_q2);

            float delata = 1;
            for (int m = 0; m < partitions.size(); m++) {
                const PartitionData& partition = partitions[m];
                // 获取该簇的单位向量e
                const std::vector<float>& e_vec = partition.centroid;
                // 计算查询向量q在e方向上的投影
                //                float alpha =
                //                        fvec_inner_product(xi_centroid_unit,
                //                        e_vec.data(), d);
                float alpha = 0.0f;
                float alpha_cos = 0.0f;
                if (R_q > 1e-8f) {
                    alpha_cos = fvec_inner_product(xi, e_vec.data(), d) / R_q;
                }

                alpha_cos = std::max(
                        -1.0f, std::min(1.0f, alpha_cos)); // 确保在[-1,1]范围内
                alpha = std::acos(alpha_cos);
                float L_j_q, U_j_q;
                // block_min
                float delta = partition.theta_min;
                //                angle_bounds_simd(alpha, delata, L_j_q, U_j_q,
                //                delta, delta); float delta =
                //                partition.theta_max;

                //                float u1 = (partition.theta_max -
                //                partition.theta_min) / 4.0f; float u2 =
                //                (partition.theta_max + partition.theta_min)
                //                / 2.0f; delta = std::sqrt((u1 * u1 + u2 * u2)
                //                / 2.0f); angle_bounds_simd(alpha, delata,
                //                L_j_q, U_j_q, delta, delta);
                // block_radius
                delta = (partition.theta_max - partition.theta_min);
                angle_bounds_simd(alpha, L_j_q, U_j_q, delta, delta, delta);
                //                printf("alpha: %.6f, L_j_q: %.6f, U_j_q: %.6f,
                //                theta_min: %.6f, theta_max: %.6f, delta:
                //                %.6f\n", alpha / M_PI * 180, L_j_q, U_j_q,
                //                partition.theta_min/ M_PI * 180,
                //                partition.theta_max/ M_PI * 180, delta / M_PI
                //                * 180);
                //
                //                float delta_d = std::abs(alpha -
                //                std::acos(alpha_cos * std::cos(partition.u)));
                ////                float delta_d = std::acos(alpha_cos *
                /// std::cos(partition.u) + std::sin(alpha) *
                /// std::sin(partition.u) * std::cos(M_PI * (1.0 - 0.8)));
                //                delta = delta_d + std::sin(alpha) *
                //                partition.delta_phi;
                ////                delta = delta_d + partition.delta_phi;

                // 高斯分布 sin
                //                 float phi_1 = 1.28;
                //                 float mean = std::cos(alpha) *
                //                 std::cos(partition.u); delta = phi_1 *
                //                 std::sin(alpha) * partition.avg_sin /
                //                 std::sqrt(2.0f); L_j_q = mean - delta; U_j_q
                //                 = mean + delta;

                //                float g = 0.8;
                //                delta = std::sin(alpha) * partition.avg_sin *
                //                std::sqrt((1 - std::cos(M_PI * g)) / 2); L_j_q
                //                = alpha_cos * std::cos(partition.u) - delta;
                //                U_j_q = alpha_cos * std::cos(partition.u) +
                //                delta;

                //                angle_bounds_simd(alpha, delata, L_j_q, U_j_q,
                //                delta, delta); float u = std::cos(alpha) *
                //                std::cos(partition.u); float delta_u =
                //                std::sin(alpha) * partition.avg_sin /
                //                std::sqrt(2.0f);
                ////                delta = std::cos(alpha) *
                /// std::cos(partition.u) + 1.64 * std::sin(alpha) *
                /// std::sin(partition.u) / std::sqrt(2.0f);
                //                L_j_q = u - delta_u;
                //                U_j_q = u + delta_u;

                //                printf("alpha: %.6f, u: %.6f, delta_u: %.6f,
                //                L_j_q: %.6f, U_j_q: %.6f\n", alpha, u,
                //                delta_u, L_j_q, U_j_q); delta = delta_d;
                //                printf("delta_d: %.6f,  delta_d%.6f,
                //                delta_phi: %.6f delta_min:%.6f,
                //                delta_radius:%.6f, delta_gauss1:%.6f\n",
                //                delta, delta_d, partition.delta_phi,
                //                partition.theta_min, (partition.theta_max -
                //                partition.theta_min), std::sqrt((u1 * u1 + u2
                //                * u2) / 2.0f)); delta =
                //                std::sin(std::sin(partition.u) * 1.64 /
                //                std::sqrt(2.0f));

                //                delta = std::sqrt(0.5) * partition.avg_sin;

                //                delta = 1.28 * std::sqrt(-2.0 *
                //                std::log(std::cos(partition.u)));

                //                delta = 20.0 / 180 * M_PI;
                //                angle_bounds_simd(alpha, delata, L_j_q, U_j_q,
                //                delta, delta);
                //
                float theta_min = partition.theta_min;
                float theta_max = partition.theta_max;
                //                angle_bounds_4(alpha, L_j_q, U_j_q, theta_min,
                //                theta_max);
                const auto& blocks = partition.blocks;

                for (size_t b = 0; b < blocks.size(); b++) {
                    const auto& block = blocks[b];
                    float lb =
                            getLB(R_q,
                                  R_q2,
                                  L_j_q,
                                  U_j_q,
                                  block.min_dist,
                                  block.max_dist);

                    // 添加到候选列表
                    //                    #pragma omp critical
                    candidate_blocks.push_back({list_no, m, b, lb});
                }
            }
            // xi_centroid = nullptr;  // 可选：防止野指针
        }
        double t1 = getmillisecs() - t0;

        max_iter = candidate_blocks.size() * iter_factor;
        max_iter = std::max(1, max_iter);
        // 第三步：按照下界排序
        if (iter_factor <= 0.6) {
            const size_t top_k =
                    std::min(candidate_blocks.size(), size_t(max_iter));
            auto cmp = [](const CandidateBlock& a, const CandidateBlock& b) {
                return a.lb < b.lb; // 根据实际需求调整比较逻辑
            };
            std::partial_sort(
                    candidate_blocks.begin(),
                    candidate_blocks.begin() + top_k,
                    candidate_blocks.end(),
                    cmp);
        } else {
            std::sort(
                    candidate_blocks.begin(),
                    candidate_blocks.end(),
                    [](const CandidateBlock& a, const CandidateBlock& b) {
                        return a.lb < b.lb;
                    });
        }

        double t2 = getmillisecs() - t0;
        // 第四步：遍历候选block
        int iter_times = 0;
        int count = 0;
        for (const auto& cand : candidate_blocks) {
            //            iter_times++;
            // 如果下界大于当前阈值，跳过该block
            //            float tmp = sqrt(current_threshold) / dd;
            //            printf(  "cand.lb: %.6f, current_threshold: %.6f, tmp:
            //            %.6f\n", cand.lb, current_threshold, tmp); if (cand.lb
            //            > current_threshold || ++iter_times >= max_iter) {
            if (cand.lb > current_threshold || iter_times++ > max_iter) {
                //            if (cand.lb > current_threshold) {
                break;
            }

            // 获取block数据
            const auto& block =
                    get_block(cand.list_no, cand.partition_id, cand.block_id);

            // 获取block内的向量
            const std::vector<idx_t>& offsets = block.offsets;
            size_t num_vecs = offsets.size();

            //            std::vector<float> block_dis(num_vecs);
            //            std::vector<idx_t> block_ids(num_vecs);

            // 计算block内每个向量的距离

            //            #pragma omp parallel for schedule(dynamic)

            for (size_t v_idx = 0; v_idx < num_vecs; v_idx++) {
                idx_t offset = offsets[v_idx];
                const float* vec_ptr =
                        (float*)invlists->get_single_code(cand.list_no, offset);
                //                reconstruct_from_offset(cand.list_no, offset,
                //                vec);
                float dis;
                if (metric_type == METRIC_INNER_PRODUCT) {
                    dis = -fvec_inner_product(xi, vec_ptr, d);
                } else {
                    dis = fvec_L2sqr(xi, vec_ptr, d);
                }

                //                block_dis[v_idx] = dis;
                //                block_ids[v_idx] = id;
                if (dis <= current_threshold) {
                    // 获取向量的id
                    //                    printf("current_threshold:
                    //                    %.6f\n", current_threshold);
                    idx_t id = invlists->get_single_id(cand.list_no, offset);
                    heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                    current_threshold = dist_i[0];
                }
            }
            count += num_vecs;
            //            heap_addn<>()
        }

        double t3 = getmillisecs() - t0;
        //        printf("query no: %d, block_size: %d, iter_times: %d,  t1: %f,
        //        t2: %f, t3: %f\n", i, candidate_blocks.size(), iter_times, t1,
        //        t2 - t1, t3 - t2);
#pragma omp atomic
        //        std::lock_guard<std::mutex> lock(stats_mutex);
        indexIVF_stats.ndis += count;
        t_lb += t1;
        t_sort += t2 - t1;
        t_ann += t3 - t2;

        //        printf("query no: %d, [block_size]: %zu, [iter_times]: %d\n",
        //               i,
        //               candidate_blocks.size(),
        //               iter_times);
        // 最终排序结果
        heap_reorder<Heap>(k, dist_i, label_i);
    }

    printf("t_lb: %f, t_sort: %f, t_ann: %f, rate: %f\n",
           t_lb,
           t_sort,
           t_ann,
           (t_lb + t_sort) / t_ann);
    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexPartitionBlockFlatSIMD::query(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int M,
        int block_size,
        float iter_factor,
        const IVFSearchParameters* params,
        std::vector<double> times,
        std::vector<double> ndis) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(M > 0);
    double t0 = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());
    using Heap = CMax<float, idx_t>;
// 并行处理每个查询
#pragma omp parallel for if (n > 10)
    for (idx_t i = 0; i < n; i++) {
        //        int max_iter = 500;
        // 当前查询向量
        const float* xi = x + i * d;
        float* dist_i = distances + i * k;
        idx_t* label_i = labels + i * k;

        // 初始化结果堆
        heap_heapify<Heap>(k, dist_i, label_i);
        float current_threshold = dist_i[0]; // 当前最大距离
        //        printf("current_threshold: %.6f\n", current_threshold);
        // 存储候选block
        struct CandidateBlock {
            idx_t list_no;
            int partition_id;
            size_t block_id;
            float lb; // 下界
        };
        std::vector<CandidateBlock> candidate_blocks;

        // 第二步：遍历每个簇，收集所有候选block
        //        #pragma omp parallel for if (n > 10)

        int block_num = 0;
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;
            block_num += cluster_min_max[list_no].second;
        }
        int max_iter = (int)((float)block_num * iter_factor);
        max_iter = std::max(1, max_iter);
        int iter_items = 0;
        int count = 0;
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;

            // 获取该簇的分区数据
            const auto& partitions = partitions_per_cluster_[list_no];
            std::vector<float> centroid(d);
            quantizer->reconstruct(list_no, centroid.data());
            // 查询向量到簇心的距离
            // 查询向量到簇心的距离
            float R_q = 0.0f;
            float R_q2 = coarse_dis[i * nprobe + j];
            R_q = std::sqrt(R_q2);

            float min = cluster_min_max[list_no].first.first;
            float max = cluster_min_max[list_no].first.second;

            if (R_q > max) {
                float lb = R_q - max;
                lb *= lb;
                if (lb > current_threshold && count > k) {
                    break;
                }
            }

            float* xi_centroid = new float[d];
            float* xi_centroid_positive = new float[d];
            //            std::vector<float> xi_centroid(d);
            // 计算查询向量与簇心的残差
            float R_q_sqr = 0.0f;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> rand_dis(0.0, 1.0);
            float std_b = 1.0;
            float xi_sum_std = 0.0;
            for (int jj = 0; jj < d; jj++) {
                const float diff = xi[jj] - centroid[jj];
                xi_centroid[jj] = diff;
                xi_centroid_positive[jj] = diff;
                if (diff < 0.0) {
                    //                    float a = rand_dis(gen);
                    //                    if (a < std_b) {
                    //
                    //                    }
                    xi_centroid_positive[jj] = -diff;
                }
                xi_sum_std += xi_centroid_positive[jj];
                R_q_sqr += diff * diff;
            }

            // 计算查询向量与簇心的残差的L2平方
            // 计算查询向量与簇心的残差的L2
            float R_q_norm = std::sqrt(R_q_sqr);

            xi_sum_std /= R_q_norm;
            // 计算查询向量与簇心的残差的单位向量
            //            float* xi_centroid_unit = new float[d];
            //            for (int jj = 0; jj < d; jj++) {
            //                xi_centroid_unit[jj] = xi_centroid[jj] / R_q_norm;
            //            }

            //            printf("alpha: %.6f, theta: %.6f, pi - theta:
            //            %.6f, angle: %.6f\n",
            //            fvec_inner_product(xi_centroid_unit, e_vec.data(),
            //            d), alpha, M_PI - alpha, alpha / M_PI * 180);
            //            float delata = 3 * sqrt(this->d - 3);
            float delata = 1;
            const std::vector<float>& e_vec = e_vec_per_cluster_[list_no];
            if (e_vec.empty()) {
                continue;
            }
            // 计算查询向量q在e方向上的投影
            float alpha_cos =
                    fvec_inner_product(xi_centroid, e_vec.data(), d) / R_q_norm;
            //            alpha_cos = std::max(
            //                    -1.0f, std::min(1.0f, alpha_cos)); //
            //                    确保在[-1,1]范围内
            //            float alpha = std::acos(alpha_cos);
            float dist_q_e = std::sqrt(2 - 2 * alpha_cos);

            int partition_size = cluster_partitions[list_no];
            for (int m = 0; m < partitions.size(); m++) {
                PartitionData partition = partitions[m];
                //                float alpha_cos_std =
                //                        fvec_inner_product(xi_centroid_positive,
                //                        partition.std_dev.data(), d) /
                //                        R_q_norm;

                //                float delta_x_o = 0.0;
                const float* std_dev = partition.std_dev.data();
                float sum_std = 0.0;
                for (int jj = 0; jj < d; jj++) {
                    sum_std += xi_centroid_positive[jj] * std_dev[jj];
                }
                float alpha_cos_std = sum_std / R_q_norm;
                //                float alpha_cos_std = xi_sum_std *
                //                partition.std;
                const auto& blocks = partition.blocks;
                // 获取该簇的单位向量e
                // 计算s的边界L_j^q和U_j^q
                float L_j_q, U_j_q;
                float phi = 0.9;
                L_j_q = alpha_cos - phi * alpha_cos_std;
                U_j_q = alpha_cos + phi * alpha_cos_std;
                L_j_q = std::max(U_j_q, -1.0f);
                U_j_q = std::min(U_j_q, 1.0f);

                //                float phi = std::cos(40.0 * M_PI / 180.0);
                //                float R = std::sqrt(alpha_cos * alpha_cos +
                //                phi * phi * (1 - alpha_cos * alpha_cos));
                //                float theta = std::acos(alpha_cos / R);
                //                float theta_min = partition.theta_min - theta;
                //                float theta_max = partition.theta_max - theta;
                //                if (theta_max < 0.0) {
                //                    L_j_q = R * std::cos(theta_min);
                //                    U_j_q = R * std::cos(theta_max);
                //                } else if (theta_min < 0.0) {
                //                    L_j_q = R * std::cos(std::min(-theta_min,
                //                    theta_max)); U_j_q = R;
                //                } else {
                //                    L_j_q = R * std::cos(theta_max);
                //                    U_j_q = R * std::cos(theta_min);
                //                }

                //                printf("list_no:%d, L: %lf, U: %lf, alpha_cos:
                //                %f, alpha_cos_std: %lf, alpha: %f, t_min: %f,
                //                t_max: %f, dist_q_e: %f,
                //                partition.o_e_dist_avg: %f, avg_sin: %f\n",
                //                       list_no, L_j_q / M_PI * 180, U_j_q /
                //                       M_PI * 180, alpha_cos, alpha_cos_std,
                //                       alpha / M_PI * 180,
                //                       partition.theta_min / M_PI * 180,
                //                       partition.theta_max / M_PI * 180,
                //                       dist_q_e,
                //                       partition.o_e_dist_avg,
                //                       partition.avg_sin);

                //                float mean = alpha_cos *
                //                std::cos(partition.u); float mean_sin =
                //                std::sin(alpha) * std::sin(partition.u); phi =
                //                std::sin(0.4 * M_PI / 2.0); L_j_q = mean - phi
                //                * mean_sin; U_j_q = mean + phi * mean_sin;
                //                float delta = partition.theta_min;
                //                if (alpha < partition.theta_min) {
                //                    // 处理alpha小于最小角度的情况
                //                    delta = partition.theta_max;
                //                }
                //                angle_bounds_simd(alpha, L_j_q, U_j_q, delta,
                //                delta, delta);

                //                U_j_q = std::cos(alpha - partition.theta_min);
                //                float u = partition.u;
                //                float avg_sin = partition.avg_sin;
                //                float u = (partition.theta_min +
                //                partition.theta_max) / 2.0; float avg_sin =
                //                std::sin(u); float mean = std::cos(alpha) *
                //                std::cos(u);
                //
                //                delta = std::sin(alpha) * avg_sin /
                //                std::sqrt(2.0f); L_j_q = mean - delta; U_j_q =
                //                mean + delta;

                // 高斯估计
                //                float exp = alpha_cos *
                //                partition.exp_cos_theta;
                ////                float var = std::sqrt(alpha_cos * alpha_cos
                ///* partition.var_cos_theta  + 0.5 * std::sin(alpha) *
                ///std::sin(alpha) * partition.avg_sin *  partition.avg_sin);
                //                float var = std::sqrt(alpha_cos * alpha_cos *
                //                partition.var_cos_theta  + 0.5 *
                //                std::sin(alpha) * std::sin(alpha) *
                //                partition.avg_sin *  partition.avg_sin);
                ////                float var = std::sqrt(0.5 * std::sin(alpha)
                ///* std::sin(alpha) * partition.avg_sin *  partition.avg_sin);
                //                L_j_q = exp - var;
                //                U_j_q = exp + var;

                //                float cos_u = std::max(alpha_cos *
                //                std::cos(partition.theta_min), alpha_cos *
                //                std::cos(partition.theta_max)); float cos_u =
                //                alpha_cos * std::cos(partition.u);
                //                //                float mean =
                //                std::max(alpha_cos *
                //                std::cos(partition.theta_min), alpha_cos *
                //                std::cos(partition.theta_max));
                //                //                float delta = 0.5 *
                //                std::max(std::sin(alpha) *
                //                std::sin(partition.theta_min), std::sin(alpha)
                //                *  std::sin(partition.theta_max));
                //                //基于cos估计
                //                float phi = std::sin(0.4 * M_PI / 2.0);

                // 利用角度估计u=pi/2， std= pi/2
                //                 float phi = std::cos( M_PI / 2.0 - 1.059);

                // 均匀分布，效果不错
                //                 float phi = std::cos( M_PI / 2.0 - 0.2 *
                //                 M_PI); float phi = std::sin( M_PI / 2.0
                //                 - 1.322); float phi =
                //                 std::sqrt(std::log(partition.num) / d);

                // N（0， 1/sqrt（2））
                //                 float phi  = 0.6;

                //                phi  = 0.14;

                //                                float sin_i_u =
                //                                std::sin(alpha) *
                //                                std::sin(partition.u);
                //                float sin_i_u = std::sin(alpha) *
                //                std::max(std::sin(partition.theta_min),
                //                std::sin(partition.theta_max));
                //
                //                float delta_p = 1.0 / (d * std::sqrt(1.0 -
                //                0.8)); L_j_q = (cos_u - sin_i_u * delta_p) /
                //                (1 + sin_i_u); U_j_q = (cos_u + sin_i_u *
                //                delta_p) / (1 - sin_i_u); L_j_q =
                //                std::max(-1.0f, L_j_q); U_j_q = std::min(1.0f,
                //                U_j_q);

                //                printf("L: %lf, U: %lf, cos_u: %lf, alpha: %f,
                //                t_min: %f, t_max: %f, sin_i_u: %f, delta_p:
                //                %f, avg_sin: %f\n",
                //                       L_j_q, U_j_q, cos_u, alpha,
                //                       partition.theta_min,
                //                       partition.theta_max,
                //                       sin_i_u,
                //                       delta_p,
                //                       partition.avg_sin);

                //                float mean = alpha_cos *
                //                std::cos(partition.theta_min);
                ////                //                float mean =
                ///std::max(alpha_cos * std::cos(partition.theta_min), alpha_cos
                ///* std::cos(partition.theta_max)); /                // float
                ///delta = 0.5 * std::max(std::sin(alpha) *
                ///std::sin(partition.theta_min), std::sin(alpha) *
                ///std::sin(partition.theta_max)); /                float phi =
                ///std::sin(0.45 * M_PI / 2.0); /                float phi =
                ///std::cos(M_PI / 2.0 * (1- 0.4)); /                float phi
                ///= 1.0;
                //                float phi =
                //                std::sqrt(partition.second_eigenvalue); delta
                //                = phi * std::sin(alpha) ; L_j_q = mean -
                //                delta; U_j_q = mean + delta;

                //                delta = alpha_cos +
                //                std::cos(partition.theta_min) - 1.0f; float
                //                q_o_e_r = std::cos(35.0 / 180.0 * M_PI) *
                //                dist_q_e * partition.o_e_dist_avg; L_j_q =
                //                delta - q_o_e_r; U_j_q = delta + q_o_e_r;
                //                L_j_q = std::max(-1.0f, L_j_q);
                //                U_j_q = std::min(1.0f, U_j_q);

                //                printf("L: %lf, U: %lf, delta: %f, q_o_e_r:
                //                %lf, alpha: %f, t_min: %f, t_max: %f,
                //                dist_q_e: %f, partition.o_e_dist_avg: %f,
                //                avg_sin: %f\n",
                //                       L_j_q, U_j_q, delta, q_o_e_r, alpha,
                //                       partition.theta_min,
                //                       partition.theta_max,
                //                       dist_q_e,
                //                       partition.o_e_dist_avg,
                //                       partition.avg_sin);
                // 遍历该分区内的所有block
                for (size_t b = 0; b < blocks.size(); b++) {
                    const auto& block = blocks[b];
                    float lb =
                            getLB(R_q,
                                  R_q2,
                                  L_j_q,
                                  U_j_q,
                                  block.min_dist,
                                  block.max_dist);

                    // 添加到候选列表
                    candidate_blocks.push_back({list_no, m, b, lb});
                }
            }
            delete[] xi_centroid;
            // xi_centroid = nullptr;  // 可选：防止野指针
        }

        // 第三步：按照下界排序
        std::sort(
                candidate_blocks.begin(),
                candidate_blocks.end(),
                [](const CandidateBlock& a, const CandidateBlock& b) {
                    return a.lb < b.lb;
                });

        // 第四步：遍历候选block
        int iter_times = 0;
        max_iter = iter_factor * candidate_blocks.size();
        max_iter = std::max(1, max_iter);
        for (const auto& cand : candidate_blocks) {
            iter_times++;
            // 如果下界大于当前阈值，跳过该block
            //            float tmp = sqrt(current_threshold) / dd;
            //            printf(  "cand.lb: %.6f, current_threshold: %.6f,
            //            tmp:
            //            %.6f\n", cand.lb, current_threshold, tmp); if
            //            (cand.lb > current_threshold || ++iter_times >=
            //            max_iter) {
            if (cand.lb > current_threshold || iter_times > max_iter) {
                //            if (cand.lb > current_threshold) {
                break;
            }

            // 获取block数据
            const auto& block =
                    get_block(cand.list_no, cand.partition_id, cand.block_id);

            // 获取block内的向量
            const std::vector<idx_t>& offsets = block.offsets;
            size_t num_vecs = offsets.size();

            // 计算block内每个向量的距离
            for (size_t v_idx = 0; v_idx < num_vecs; v_idx++) {
                idx_t offset = offsets[v_idx];
                float* vec = new float[d];
                reconstruct_from_offset(cand.list_no, offset, vec);

                float dis;
                if (metric_type == METRIC_INNER_PRODUCT) {
                    dis = -fvec_inner_product(xi, vec, d);
                } else {
                    dis = fvec_L2sqr(xi, vec, d);
                }
                delete[] vec;

                if (dis < current_threshold) {
                    // 获取向量的id
                    //                    printf("current_threshold:
                    //                    %.6f\n", current_threshold);
                    idx_t id = invlists->get_single_id(cand.list_no, offset);
                    heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                    current_threshold = dist_i[0];
                }
            }
            std::lock_guard<std::mutex> lock(stats_mutex);
            //            printf("indexIVF_stats: ndis: %zu, %zu",
            //            indexIVF_stats.ndis, num_vecs);
            indexIVF_stats.ndis += num_vecs;
        }

        //        printf("query no: %d, [block_size]: %zu, [iter_times]: %d\n",
        //               i,
        //               candidate_blocks.size(),
        //               iter_times);
        // 最终排序结果
        heap_reorder<Heap>(k, dist_i, label_i);
    }
    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexPartitionBlockFlatSIMD::query_perpendicular(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int M,
        int block_size,
        float iter_factor,
        const IVFSearchParameters* params,
        std::vector<double> times,
        std::vector<double> ndis) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(M > 0);
    double t0 = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());
    using Heap = CMax<float, idx_t>;
// 并行处理每个查询
#pragma omp parallel for if (n > 10)
    for (idx_t i = 0; i < n; i++) {
        //        int max_iter = 500;
        // 当前查询向量
        const float* xi = x + i * d;
        float* dist_i = distances + i * k;
        idx_t* label_i = labels + i * k;

        // 初始化结果堆
        heap_heapify<Heap>(k, dist_i, label_i);
        float current_threshold = dist_i[0]; // 当前最大距离
        //        printf("current_threshold: %.6f\n", current_threshold);
        // 存储候选block
        struct CandidateBlock {
            idx_t list_no;
            int partition_id;
            size_t block_id;
            float lb; // 下界
        };
        std::vector<CandidateBlock> candidate_blocks;

        // 第二步：遍历每个簇，收集所有候选block
        //        #pragma omp parallel for if (n > 10)

        int block_num = 0;
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;
            block_num += cluster_min_max[list_no].second;
        }
        int max_iter = (int)((float)block_num * iter_factor);
        max_iter = std::max(1, max_iter);
        int iter_items = 0;
        int count = 0;
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;

            // 获取该簇的分区数据
            const auto& partitions = partitions_per_cluster_[list_no];
            std::vector<float> centroid(d);
            quantizer->reconstruct(list_no, centroid.data());
            // 查询向量到簇心的距离
            // 查询向量到簇心的距离
            float R_q = 0.0f;
            float R_q2 = coarse_dis[i * nprobe + j];
            R_q = std::sqrt(R_q2);

            float min = cluster_min_max[list_no].first.first;
            float max = cluster_min_max[list_no].first.second;

            if (R_q > max) {
                float lb = R_q - max;
                lb *= lb;
                if (lb > current_threshold && count > k) {
                    break;
                }
            }

            float* xi_centroid = new float[d];
            float* xi_centroid_positive = new float[d];
            float* xi_pert_centroid = new float[d];
            //            std::vector<float> xi_centroid(d);
            // 计算查询向量与簇心的残差
            float R_q_sqr = 0.0f;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> rand_dis(0.0, 1.0);
            float std_b = 1.0;
            for (int jj = 0; jj < d; jj++) {
                const float diff = xi[jj] - centroid[jj];
                xi_centroid[jj] = diff;
                R_q_sqr += diff * diff;
            }

            const std::vector<float>& e_vec = e_vec_per_cluster_[list_no];

            if (e_vec.empty()) {
                continue;
            }
            // 计算查询向量与簇心的残差的L2平方
            // 计算查询向量与簇心的残差的L2
            float R_q_norm = std::sqrt(R_q_sqr);
            float alpha_cos = 1.0f;
            //            float xi_sum_std = 0.0;
            //            if (R_q_norm > 0.0) {
            //                alpha_cos = fvec_inner_product(xi_centroid,
            //                e_vec.data(), d) / R_q_norm; for (int jj = 0; jj <
            //                d; jj++) {
            //                    xi_centroid[jj] /= R_q_norm;
            //                    xi_centroid_positive[jj] = xi_centroid[jj];
            //                    if ( xi_centroid_positive[jj] < 0.0) {
            //                        xi_centroid_positive[jj] =
            //                        -xi_centroid_positive[jj];
            //                    }
            //                    xi_sum_std += xi_centroid_positive[jj];
            //                }
            //            }

            // 计算查询向量与簇心的残差的L2平方，垂向量
            float xi_sum_std = 0.0;
            if (R_q_norm > 0.0) {
                alpha_cos = fvec_inner_product(xi_centroid, e_vec.data(), d) /
                        R_q_norm;
                for (int jj = 0; jj < d; jj++) {
                    xi_centroid[jj] /= R_q_norm;
                    float diff = xi_centroid[jj] - alpha_cos * e_vec[jj];
                    xi_pert_centroid[jj] = diff;
                    xi_centroid_positive[jj] = diff;
                    if (diff < 0.0) {
                        xi_centroid_positive[jj] = -diff;
                    }
                    xi_sum_std += xi_centroid_positive[jj];
                }

                float d_x_pert =
                        std::sqrt(fvec_norm_L2sqr(xi_pert_centroid, d));
                if (d_x_pert > 0.0) {
                    xi_sum_std /= d_x_pert;
                    for (int jj = 0; jj < d; jj++) {
                        xi_pert_centroid[jj] /= d_x_pert;
                        xi_centroid_positive[jj] /= d_x_pert;
                    }
                }
            }
            //

            // 计算查询向量q在e方向上的投影

            //            alpha_cos = std::max(
            //                    -1.0f, std::min(1.0f, alpha_cos)); //
            //                    确保在[-1,1]范围内
            //            float alpha = std::acos(alpha_cos);
            float dist_q_e = std::sqrt(2 - 2 * alpha_cos);

            int partition_size = cluster_partitions[list_no];

            std::priority_queue<PartitionCandidate> partition_queue;

            for (int m = 0; m < partitions.size(); m++) {
                PartitionData partition = partitions[m];
                //                float alpha_cos_std =
                //                        fvec_inner_product(xi_centroid_positive,
                //                        partition.std_dev.data(), d) /
                //                        R_q_norm;

                //                float delta_x_o = 0.0;
                const float* std_dev = partition.std_dev.data();
                float sum_std = 0.0;
                for (int jj = 0; jj < d; jj++) {
                    sum_std += xi_centroid_positive[jj] * std_dev[jj];
                }
                float alpha_cos_std = sum_std;
                //                                float alpha_cos_std =
                //                                xi_sum_std * partition.std;
                //                                float alpha_cos_std =
                //                                xi_sum_std  / std::sqrt(d);
                const auto& blocks = partition.blocks;
                // 获取该簇的单位向量e
                // 计算s的边界L_j^q和U_j^q
                float L_j_q, U_j_q;
                //                float phi = 0.9;
                //                std::vector<float>& p_centroid =
                //                partition.centroid; float d_x =
                //                fvec_norm_L2sqr(xi_centroid, d); float d_c =
                //                fvec_norm_L2sqr(p_centroid.data(), d); float
                //                alpha_cos_p =
                //                fvec_inner_product(p_centroid.data(),
                //                xi_centroid, d); L_j_q = alpha_cos_p - phi *
                //                alpha_cos_std; U_j_q = alpha_cos_p + phi *
                //                alpha_cos_std;

                //                printf("L_j_q: %.6f, U_j_q: %.6f, d_x: %.4f,
                //                d_c: %.4f, alpha_cos_p: %.4f, phi: %.4f,
                //                alpha_cos_std: %.4f\n",
                //                       L_j_q, U_j_q, d_x,  d_c, alpha_cos_p,
                //                       phi, alpha_cos_std);
                //                phi = std::cos(40 * M_PI / 180);
                //                L_j_q = alpha_cos - phi * alpha_cos_std;
                //                U_j_q = alpha_cos + phi * alpha_cos_std;
                //
                //
                //                L_j_q = std::max(U_j_q, -1.0f);
                //                U_j_q = std::min(U_j_q, 1.0f);
                //
                //                float cos_theta = alpha_cos *
                //                std::cos(partition.u); float sin_theta =
                //                std::sqrt(1.0 - alpha_cos * alpha_cos) *
                //                std::sin(partition.u); L_j_q = cos_theta - phi
                //                * sin_theta; U_j_q = cos_theta + phi *
                //                sin_theta;

                //                基于垂向量
                std::vector<float>& perpendicular_centroid = partition.centroid;
                float alpha_cos_perpendicular = fvec_inner_product(
                        perpendicular_centroid.data(), xi_pert_centroid, d);
                float phi = 0.85;
                phi = alpha_cos_perpendicular + phi * alpha_cos_std;
                // 基于角度预估
                //                phi = std::cos(55 * M_PI / 180);
                float R = std::sqrt(
                        alpha_cos * alpha_cos +
                        phi * phi * (1 - alpha_cos * alpha_cos));
                float cos_theta = alpha_cos / R;
                cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
                float sin_theta = std::max(
                        -1.0f,
                        std::min(
                                1.0f,
                                phi * std::sin(std::acos(alpha_cos)) / R));
                float theta = std::acos(cos_theta);
                //                float theta = std::asin(sin_theta);
                float theta_min = partition.theta_min - theta;
                float theta_max = partition.theta_max - theta;
                if (theta_max < 0.0) {
                    L_j_q = R * std::cos(theta_min);
                    U_j_q = R * std::cos(theta_max);
                } else if (theta_min < 0.0) {
                    L_j_q = R * std::cos(std::min(-theta_min, theta_max));
                    U_j_q = R;
                } else {
                    L_j_q = R * std::cos(theta_max);
                    U_j_q = R * std::cos(theta_min);
                }

                //                printf("list_no:%d, p_id:%d, num: %d, std:
                //                %.4f, L_j_q: %.4f, U_j_q: %.4f, R: %.4f, phi:
                //                %.4f, alpha_cos: %.4f, cos_theta: %.4f, "
                //                       "p_theta_min: %.4f, p_theta_max: %.4f,
                //                       theta: %.4f, theta_min: %.4f,
                //                       theta_max: %.4f\n",
                //                        list_no, m, partition.num,
                //                        partition.std, L_j_q, U_j_q, R, phi,
                //                        std::acos(alpha_cos) / M_PI * 180,
                //                        std::acos(cos_theta)/ M_PI * 180,
                //                       partition.theta_min / M_PI * 180,
                //                       partition.theta_max / M_PI * 180,
                //                       theta/ M_PI * 180, theta_min/ M_PI *
                //                       180, theta_max/ M_PI * 180);

                L_j_q = std::max(U_j_q, -1.0f);
                U_j_q = std::min(U_j_q, 1.0f);
                float p_lb =
                        getLB(R_q,
                              R_q2,
                              L_j_q,
                              U_j_q,
                              partition.min_dis,
                              partition.max_dis);

                if (p_lb > current_threshold && count > k) {
                    continue;
                }
                // 将分区加入优先队列
                partition_queue.push(
                        {list_no, m, R_q2, R_q, L_j_q, U_j_q, p_lb, alpha_cos});
            }

            bool flag = true;
            while (!partition_queue.empty() && flag) {
                PartitionCandidate candidate = partition_queue.top();
                partition_queue.pop();
                if (candidate.lb > current_threshold && count > k) {
                    flag = false;
                    break;
                }
                // 获取该分区
                const PartitionData& partition =
                        partitions_per_cluster_[candidate.list_no]
                                               [candidate.partition_id];
                for (size_t b = 0; b < partition.blocks.size(); b++) {
                    const BlockData& block = partition.blocks[b];
                    float block_lb =
                            getLB(candidate.Rq, // 使用分区下界作为参考
                                  candidate.Rq_2,
                                  candidate.L_j_q,
                                  candidate.U_j_q,
                                  //                                  L_j_q,
                                  //                                  U_j_q,
                                  block.min_dist,
                                  block.max_dist);

                    if (block_lb > current_threshold && count > k) {
                        continue;
                    }
                    // 获取block内的向量
                    const std::vector<idx_t>& offsets = block.offsets;
                    size_t num_vecs = offsets.size();

                    // 计算block内每个向量的距离
                    for (size_t v_idx = 0; v_idx < num_vecs; v_idx++) {
                        idx_t offset = offsets[v_idx];
                        float* vec = new float[d];
                        reconstruct_from_offset(candidate.list_no, offset, vec);

                        float dis;
                        if (metric_type == METRIC_INNER_PRODUCT) {
                            dis = -fvec_inner_product(xi, vec, d);
                        } else {
                            dis = fvec_L2sqr(xi, vec, d);
                        }
                        delete[] vec;

                        if (dis < current_threshold) {
                            // 获取向量的id
                            idx_t id = invlists->get_single_id(
                                    candidate.list_no, offset);
                            heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                            current_threshold = dist_i[0];
                        }
                    }
                    count += num_vecs;
                    if (++iter_items > max_iter) {
                        flag = false;
                        break;
                    }
                }
            }
        }
        heap_reorder<Heap>(k, dist_i, label_i);
#pragma omp atomic
        indexIVF_stats.ndis += count;
    }
    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexPartitionBlockFlatSIMD::query_by_e_vect(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int M,
        int block_size,
        const IVFSearchParameters* params,
        std::vector<double> times,
        std::vector<double> ndis) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(M > 0);
    double t0 = getmillisecs();
    // 获取查询参数
    const IVFSearchParameters* ivf_params = nullptr;
    if (params) {
        ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
        FAISS_THROW_IF_NOT_MSG(
                ivf_params, "IndexIVF params have incorrect type");
    }

    const size_t nprobe =
            std::min(nlist, ivf_params ? ivf_params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // 分配内存
    std::vector<idx_t> coarse_idx(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);

    // 第一步：寻找nprobe个最近的簇
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_idx.data());

    using Heap = CMax<float, idx_t>;
    // 并行处理每个查询
    //    int batch = 10000;
    //    int batch_num = (n + batch - 1) / batch;
    //    for (int bat = 0; bat < batch_num; bat ++) {
    //        int start = bat * batch;
    //        int end = (bat + 1) * batch;
    float t_lb = 0.0;
    float t_sort = 0.0;
    float t_ann = 0.0;
    // 定义四个基向量
    std::vector<float> e1(d, 1.0f / std::sqrt(d));
    std::vector<float> e2(d, -1.0f / std::sqrt(d));
    std::vector<float> e3(d);
    std::vector<float> e4(d);

    for (int i = 0; i < d; i++) {
        if (i % 2 == 0) {
            e3[i] = -1.0f / std::sqrt(d);
            e4[i] = 1.0f / std::sqrt(d);
        } else {
            e3[i] = 1.0f / std::sqrt(d);
            e4[i] = -1.0f / std::sqrt(d);
        }
    }
    std::vector<std::vector<float>> base_vectors = {e1, e2, e3, e4};

    // 存储所有基向量
    //    std::vector<std::vector<float>> base_vectors = {e1, e2, e3, e4};

#pragma omp parallel for if (n > 10) reduction(+ : t_lb, t_sort, t_ann)
    for (idx_t i = 0; i < n; i++) {
        int max_iter = 500;
        float iter_factor = 0.8;
        // 当前查询向量
        const float* xi = x + i * d;
        float* dist_i = distances + i * k;
        idx_t* label_i = labels + i * k;

        // 初始化结果堆
        heap_heapify<Heap>(k, dist_i, label_i);
        float current_threshold = dist_i[0]; // 当前最大距离
        //        printf("current_threshold: %.6f\n", current_threshold);
        // 存储候选block
        struct CandidateBlock {
            idx_t list_no;
            int partition_id;
            size_t block_id;
            float lb; // 下界
        };
        std::vector<CandidateBlock> candidate_blocks;

        // 第二步：遍历每个簇，收集所有候选block
        double t0 = getmillisecs();
        // #pragma omp parallel for if (n > 10)
        for (size_t j = 0; j < nprobe; j++) {
            idx_t list_no = coarse_idx[i * nprobe + j];
            if (list_no < 0)
                continue;

            // 获取该簇的分区数据
            const auto& partitions = partitions_per_cluster_[list_no];
            std::vector<float> centroid(d);
            quantizer->reconstruct(list_no, centroid.data());

            // 存储所有基向量
            //            std::vector<float> centroid_e1(d);
            //            std::vector<float> centroid_e2(d);
            //            std::vector<float> centroid_e3(d);
            //            std::vector<float> centroid_e4(d);
            //            norm_unit_vector(centroid, e1.data(), centroid_e1, d);
            //            norm_unit_vector(centroid, e2.data(), centroid_e2, d);
            //            norm_unit_vector(centroid, e3.data(), centroid_e3, d);
            //            norm_unit_vector(centroid, e4.data(), centroid_e4, d);

            // 查询向量到簇心的距离
            float R_q = 0.0f;
            if (metric_type == METRIC_INNER_PRODUCT) {
                // 对于内积，计算查询向量与簇心的内积
                R_q = fvec_inner_product(xi, centroid.data(), d);
            } else {
                // 对于L2，计算查询向量到簇心的距离
                R_q = fvec_L2sqr(xi, centroid.data(), d);
                R_q = std::sqrt(R_q);
            }
            float R_q2 = R_q * R_q;

            float* xi_centroid_unit = new float[d];
            //            std::vector<float> xi_centroid(d);
            // 计算查询向量与簇心的残差
            for (int jj = 0; jj < d; jj++) {
                xi_centroid_unit[jj] = xi[jj] - centroid[jj];
            }
            // 计算查询向量与簇心的残差的L2平方
            float R_q_sqr = fvec_norm_L2sqr(xi_centroid_unit, d);

            // 计算查询向量与簇心的残差的L2
            float R_q_norm = std::sqrt(R_q_sqr);

            // 计算查询向量与簇心的残差的单位向量
            //            float* xi_centroid_unit = new float[d];
            for (int jj = 0; jj < d; jj++) {
                xi_centroid_unit[jj] = xi_centroid_unit[jj] / R_q_norm;
            }

            //            printf("alpha: %.6f, theta: %.6f, pi - theta: %.6f,
            //            angle: %.6f\n", fvec_inner_product(xi_centroid_unit,
            //            e_vec.data(), d), alpha, M_PI - alpha, alpha / M_PI *
            //            180); float delata = 3 * sqrt(this->d - 3);
            for (int m = 0; m < partitions.size(); m++) {
                PartitionData partition = partitions[m];
                // 获取该簇的单位向量e
                //                const std::vector<float>& e_vec =
                //                partition.centroid;
                // 计算查询向量q在e方向上的投影
                float alpha = fvec_inner_product(
                        xi_centroid_unit, base_vectors[m].data(), d);
                alpha = std::max(
                        -1.0f, std::min(1.0f, alpha)); // 确保在[-1,1]范围内
                alpha = std::acos(alpha);
                float L_j_q, U_j_q;

                //                float delta = partition.theta_max -
                //                partition.theta_min; angle_bounds_simd(alpha,
                //                L_j_q, U_j_q, delta, delta);

                float phi_1 = 1.28;
                float u = (partition.theta_max + partition.theta_min) / 2.0;
                float mean = std::cos(alpha) * std::cos(u);
                float delta =
                        phi_1 * std::sin(alpha) * std::sin(u) / std::sqrt(2.0f);
                L_j_q = mean - delta;
                U_j_q = mean + delta;

                //                printf("L_j_q: %.6f, U_j_q: %.6f, alpha: %.6f,
                //                theta_min: %.6f, theta_max: %.6f, delta:
                //                %.6f\n", L_j_q, U_j_q, alpha / M_PI * 180,
                //                partition.theta_min  / M_PI * 180,
                //                partition.theta_max  / M_PI * 180, delta  /
                //                M_PI * 180);
                const auto& blocks = partition.blocks;

                float theta_min = partition.theta_min;
                float theta_max = partition.theta_max;
                for (size_t b = 0; b < blocks.size(); b++) {
                    const auto& block = blocks[b];
                    // 计算s的边界L_j^q和U_j^q
                    //                    ee = 0.0f + block.theta_min;
                    //                    angle_bounds_simd(
                    //                            alpha, delata, L_j_q, U_j_q,
                    //                            ee, ee);

                    float lb =
                            getLB(R_q,
                                  R_q2,
                                  L_j_q,
                                  U_j_q,
                                  block.min_dist,
                                  block.max_dist);

                    // 添加到候选列表
                    candidate_blocks.push_back({list_no, m, b, lb});
                }
            }
            //            delete[] xi_centroid_unit;
            // xi_centroid = nullptr;  // 可选：防止野指针
        }
        double t1 = getmillisecs() - t0;
        // 第三步：按照下界排序
        std::sort(
                candidate_blocks.begin(),
                candidate_blocks.end(),
                [](const CandidateBlock& a, const CandidateBlock& b) {
                    return a.lb < b.lb;
                });

        max_iter = candidate_blocks.size() * iter_factor;
        max_iter = std::max(1, max_iter);
        double t2 = getmillisecs() - t0;
        // 第四步：遍历候选block
        int iter_times = 0;
        float* vec = new float[d];
        for (const auto& cand : candidate_blocks) {
            if (cand.lb > current_threshold || iter_times++ > max_iter) {
                //            if (cand.lb > current_threshold) {
                break;
            }

            // 获取block数据
            const auto& block =
                    get_block(cand.list_no, cand.partition_id, cand.block_id);

            // 获取block内的向量
            const std::vector<idx_t>& offsets = block.offsets;
            size_t num_vecs = offsets.size();
            for (size_t v_idx = 0; v_idx < num_vecs; v_idx++) {
                idx_t offset = offsets[v_idx];
                reconstruct_from_offset(cand.list_no, offset, vec);

                float dis;
                if (metric_type == METRIC_INNER_PRODUCT) {
                    dis = -fvec_inner_product(xi, vec, d);
                } else {
                    dis = fvec_L2sqr(xi, vec, d);
                }
                if (dis < current_threshold) {
                    idx_t id = invlists->get_single_id(cand.list_no, offset);
                    heap_replace_top<Heap>(k, dist_i, label_i, dis, id);
                    current_threshold = dist_i[0];
                }
            }

//            heap_addn<>()
#pragma omp atomic
            indexIVF_stats.ndis += num_vecs;
        }
        double t3 = getmillisecs() - t0;
        //        printf("query no: %d, block_size: %d, iter_times: %d,  t1: %f,
        //        t2: %f, t3: %f\n", i, candidate_blocks.size(), iter_times, t1,
        //        t2 - t1, t3 - t2);
        t_lb += t1;
        t_sort += t2 - t1;
        t_ann += t3 - t2;
        //        delete[] vec;
        //        candidate_blocks.clear();

        //        printf("query no: %d, [block_size]: %zu, [iter_times]: %d\n",
        //               i,
        //               candidate_blocks.size(),
        //               iter_times);
        // 最终排序结果
        heap_reorder<Heap>(k, dist_i, label_i);
    }
    printf("t_lb: %f, t_sort: %f, t_ann: %f\n", t_lb, t_sort, t_ann);
    indexIVF_stats.search_time += getmillisecs() - t0;
}

float IndexPartitionBlockFlatSIMD::getLB(
        float R_q,
        float R_q2,
        float L_j_q,
        float U_j_q,
        float r_min,
        float r_max) const {
    //    float r_min = block.min_dist;
    //    float r_max = block.max_dist;
    // 计算距离函数f(r,s)的边界
    //    float p1 = r_min * r_min + R_q2 - 2 * R_q * r_min * L_j_q;
    float p2 = r_min * r_min + R_q2 - 2 * R_q * r_min * U_j_q;
    //    float p3 = r_max * r_max + R_q2 - 2 * R_q * r_max * L_j_q;
    float p4 = r_max * r_max + R_q2 - 2 * R_q * r_max * U_j_q;
    float lb = 0.0f;
    float p5 = p2;
    if (L_j_q > 0.0f) {
        float r0 = R_q * L_j_q;
        if (r0 <= r_max && r0 >= r_min) {
            p5 = R_q2 * (1 - L_j_q * L_j_q);
        }
    }

    float p6 = p2;
    float r0 = R_q * U_j_q;
    if (r0 <= r_max && r0 >= r_min) {
        p6 = R_q2 * (1 - U_j_q * U_j_q);
    }

    // 计算下界
    if (R_q > 0.0) {
        float s0_min = r_min / R_q;
        float s0_max = r_max / R_q;
        float r0_min = R_q * L_j_q;
        float r0_max = R_q * U_j_q;
        // 存在交界
        if (s0_min <= U_j_q && s0_max >= L_j_q && r0_min <= r_max &&
            r0_max >= r_min) {
            float s_max = std::min(s0_max, U_j_q);
            float p7 = R_q2 * (1 - s_max * s_max);
            lb = std::min({p2, p4, p5, p6, p7});
        } else {
            lb = std::min({p2, p4, p5, p6});
        }
    } else {
        lb = std::min({p2, p4, p5, p6});
    }

    //    lb = std::min({p2, p4, p5, p6});
    return lb;
}

void IndexPartitionBlockFlatSIMD::angle_bounds(
        float alpha,
        float delata,
        PartitionData& partition,
        float& L_j_q,
        float& U_j_q,
        float& theta_min,
        float& theta_max) const {
    theta_min = partition.theta_min;
    theta_max = partition.theta_min; // 下界L_j^q
    angle_bounds_simd(alpha, L_j_q, U_j_q, theta_min, theta_min, theta_max);
}

const std::vector<float>& IndexPartitionBlockFlatSIMD::get_e_vec(
        idx_t list_no) const {
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return e_vec_per_cluster_[list_no];
}

namespace {

template <MetricType metric, class C, bool use_sel>
struct IVFFlatScanner : InvertedListScanner {
    size_t d;

    IVFFlatScanner(size_t d, bool store_pairs, const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), d(d) {
        keep_max = is_similarity_metric(metric);
        code_size = d * sizeof(float);
    }

    const float* xi;
    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* code) const override {
        const float* yj = (float*)code;
        float dis = metric == METRIC_INNER_PRODUCT
                ? fvec_inner_product(xi, yj, d)
                : fvec_L2sqr(xi, yj, d);
        return dis;
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        const float* list_vecs = (const float*)codes;
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + d * j;
            if (use_sel && !sel->is_member(ids[j])) {
                continue;
            }
            float dis = metric == METRIC_INNER_PRODUCT
                    ? fvec_inner_product(xi, yj, d)
                    : fvec_L2sqr(xi, yj, d);
            if (C::cmp(simi[0], dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                heap_replace_top<C>(k, simi, idxi, dis, id);
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const float* list_vecs = (const float*)codes;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + d * j;
            if (use_sel && !sel->is_member(ids[j])) {
                continue;
            }
            float dis = metric == METRIC_INNER_PRODUCT
                    ? fvec_inner_product(xi, yj, d)
                    : fvec_L2sqr(xi, yj, d);
            if (C::cmp(radius, dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(dis, id);
            }
        }
    }
};

template <bool use_sel>
InvertedListScanner* get_InvertedListScanner1(
        const IndexPartitionBlockFlatSIMD* ivf,
        bool store_pairs,
        const IDSelector* sel) {
    if (ivf->metric_type == METRIC_INNER_PRODUCT) {
        return new IVFFlatScanner<
                METRIC_INNER_PRODUCT,
                CMin<float, int64_t>,
                use_sel>(ivf->d, store_pairs, sel);
    } else if (ivf->metric_type == METRIC_L2) {
        return new IVFFlatScanner<METRIC_L2, CMax<float, int64_t>, use_sel>(
                ivf->d, store_pairs, sel);
    } else {
        FAISS_THROW_MSG("metric type not supported");
    }
}

} // anonymous namespace

InvertedListScanner* IndexPartitionBlockFlatSIMD::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    if (sel) {
        return get_InvertedListScanner1<true>(this, store_pairs, sel);
    } else {
        return get_InvertedListScanner1<false>(this, store_pairs, sel);
    }
}

void IndexPartitionBlockFlatSIMD::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}

/*****************************************
 * IndexPartitionBlockFlatSIMDDedup implementation
 ******************************************/

IndexPartitionBlockFlatSIMDDedup::IndexPartitionBlockFlatSIMDDedup(
        Index* quantizer,
        size_t d,
        size_t nlist_,
        MetricType metric_type,
        bool own_invlists)
        : IndexPartitionBlockFlatSIMD(
                  quantizer,
                  d,
                  nlist_,
                  metric_type,
                  own_invlists) {}

void IndexPartitionBlockFlatSIMDDedup::train(idx_t n, const float* x) {
    std::unordered_map<uint64_t, idx_t> map;
    std::unique_ptr<float[]> x2(new float[n * d]);

    int64_t n2 = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t hash = hash_bytes((uint8_t*)(x + i * d), code_size);
        if (map.count(hash) &&
            !memcmp(x2.get() + map[hash] * d, x + i * d, code_size)) {
            // is duplicate, skip
        } else {
            map[hash] = n2;
            memcpy(x2.get() + n2 * d, x + i * d, code_size);
            n2++;
        }
    }
    if (verbose) {
        printf("IndexPartitionBlockFlatSIMDDedup::train: train on %" PRId64
               " points after dedup "
               "(was %" PRId64 " points)\n",
               n2,
               n);
    }
    IndexPartitionBlockFlatSIMD::train(n2, x2.get());
}

void IndexPartitionBlockFlatSIMDDedup::add_with_ids(
        idx_t na,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);
    assert(invlists);
    FAISS_THROW_IF_NOT_MSG(
            direct_map.no(), "IVFFlatDedup not implemented with direct_map");
    std::unique_ptr<int64_t[]> idx(new int64_t[na]);
    quantizer->assign(na, x, idx.get());

    int64_t n_add = 0, n_dup = 0;

#pragma omp parallel reduction(+ : n_add, n_dup)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < na; i++) {
            int64_t list_no = idx[i];

            if (list_no < 0 || list_no % nt != rank) {
                continue;
            }

            idx_t id = xids ? xids[i] : ntotal + i;
            const float* xi = x + i * d;

            // search if there is already an entry with that id
            InvertedLists::ScopedCodes codes(invlists, list_no);

            int64_t n = invlists->list_size(list_no);
            int64_t offset = -1;
            for (int64_t o = 0; o < n; o++) {
                if (!memcmp(codes.get() + o * code_size, xi, code_size)) {
                    offset = o;
                    break;
                }
            }

            if (offset == -1) { // not found
                invlists->add_entry(list_no, id, (const uint8_t*)xi);
            } else {
                // mark equivalence
                idx_t id2 = invlists->get_single_id(list_no, offset);
                std::pair<idx_t, idx_t> pair(id2, id);

#pragma omp critical
                // executed by one thread at a time
                instances.insert(pair);

                n_dup++;
            }
            n_add++;
        }
    }
    if (verbose) {
        printf("IndexPartitionBlockFlatSIMD::add_with_ids: added %" PRId64
               " / %" PRId64
               " vectors"
               " (out of which %" PRId64 " are duplicates)\n",
               n_add,
               na,
               n_dup);
    }
    ntotal += n_add;
}

void IndexPartitionBlockFlatSIMDDedup::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT_MSG(
            !store_pairs, "store_pairs not supported in IVFDedup");

    IndexPartitionBlockFlatSIMD::search_preassigned(
            n, x, k, assign, centroid_dis, distances, labels, false, params);

    std::vector<idx_t> labels2(k);
    std::vector<float> dis2(k);

    for (int64_t i = 0; i < n; i++) {
        idx_t* labels1 = labels + i * k;
        float* dis1 = distances + i * k;
        int64_t j = 0;
        for (; j < k; j++) {
            if (instances.find(labels1[j]) != instances.end()) {
                // a duplicate: special handling
                break;
            }
        }
        if (j < k) {
            // there are duplicates, special handling
            int64_t j0 = j;
            int64_t rp = j;
            while (j < k) {
                auto range = instances.equal_range(labels1[rp]);
                float dis = dis1[rp];
                labels2[j] = labels1[rp];
                dis2[j] = dis;
                j++;
                for (auto it = range.first; j < k && it != range.second; ++it) {
                    labels2[j] = it->second;
                    dis2[j] = dis;
                    j++;
                }
                rp++;
            }
            memcpy(labels1 + j0,
                   labels2.data() + j0,
                   sizeof(labels1[0]) * (k - j0));
            memcpy(dis1 + j0, dis2.data() + j0, sizeof(dis2[0]) * (k - j0));
        }
    }
}

size_t IndexPartitionBlockFlatSIMDDedup::remove_ids(const IDSelector& sel) {
    std::unordered_map<idx_t, idx_t> replace;
    std::vector<std::pair<idx_t, idx_t>> toadd;
    for (auto it = instances.begin(); it != instances.end();) {
        if (sel.is_member(it->first)) {
            // then we erase this entry
            if (!sel.is_member(it->second)) {
                // if the second is not erased
                if (replace.count(it->first) == 0) {
                    replace[it->first] = it->second;
                } else { // remember we should add an element
                    std::pair<idx_t, idx_t> new_entry(
                            replace[it->first], it->second);
                    toadd.push_back(new_entry);
                }
            }
            it = instances.erase(it);
        } else {
            if (sel.is_member(it->second)) {
                it = instances.erase(it);
            } else {
                ++it;
            }
        }
    }

    instances.insert(toadd.begin(), toadd.end());

    // mostly copied from IndexIVF.cpp

    FAISS_THROW_IF_NOT_MSG(
            direct_map.no(), "direct map remove not implemented");

    std::vector<int64_t> toremove(nlist);

#pragma omp parallel for
    for (int64_t i = 0; i < nlist; i++) {
        int64_t l0 = invlists->list_size(i), l = l0, j = 0;
        InvertedLists::ScopedIds idsi(invlists, i);
        while (j < l) {
            if (sel.is_member(idsi[j])) {
                if (replace.count(idsi[j]) == 0) {
                    l--;
                    invlists->update_entry(
                            i,
                            j,
                            invlists->get_single_id(i, l),
                            InvertedLists::ScopedCodes(invlists, i, l).get());
                } else {
                    invlists->update_entry(
                            i,
                            j,
                            replace[idsi[j]],
                            InvertedLists::ScopedCodes(invlists, i, j).get());
                    j++;
                }
            } else {
                j++;
            }
        }
        toremove[i] = l0 - l;
    }
    // this will not run well in parallel on ondisk because of possible
    // shrinks
    int64_t nremove = 0;
    for (int64_t i = 0; i < nlist; i++) {
        if (toremove[i] > 0) {
            nremove += toremove[i];
            invlists->resize(i, invlists->list_size(i) - toremove[i]);
        }
    }
    ntotal -= nremove;
    return nremove;
}

void IndexPartitionBlockFlatSIMDDedup::range_search(
        idx_t,
        const float*,
        float,
        RangeSearchResult*,
        const SearchParameters*) const {
    FAISS_THROW_MSG("not implemented");
}

void IndexPartitionBlockFlatSIMDDedup::update_vectors(
        int,
        const idx_t*,
        const float*) {
    FAISS_THROW_MSG("not implemented");
}

void IndexPartitionBlockFlatSIMDDedup::reconstruct_from_offset(
        int64_t,
        int64_t,
        float*) const {
    FAISS_THROW_MSG("not implemented");
}

} // namespace faiss
