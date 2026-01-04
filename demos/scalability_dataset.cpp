//
// Created by root on 12/28/25.
//
// 使用指定数据集进行可扩展性测试
// 支持通过随机旋转矩阵扩展数据量
//
#include <H5Cpp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include <omp.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexHNSW.h>
#include <faiss/ABQSIMD.h>
#include <faiss/ABQ_SQ.h>
#include <faiss/ABQ_PQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <thread>
#include <faiss/index_factory.h>
#include <faiss/MetricType.h>
#include <faiss/index_io.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <fstream>
#include <map>
#include <utility>
#include <faiss/utils/utils.h>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#include "demos/data_loader.h"

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 计算召回率（使用ground truth向量格式）
// 注意：扩展数据后，原始向量的id会映射到 id % nb_orig
double calculate_recall_expanded(
        const std::vector<std::vector<int>>& ground_truth,
        const faiss::idx_t* query_results,
        int nq,
        int top_k,
        size_t nb_orig) {
    int count = 0;
    for (size_t i = 0; i < nq; i++) {
        for (int j = 0; j < top_k; j++) {
            // 将扩展后的id映射回原始id
            faiss::idx_t v1 = query_results[i * top_k + j] % nb_orig;
            for (int k = 0; k < top_k; k++) {
                int v2 = ground_truth[i][k];
                if (v1 == v2) {
                    count++;
                    break;
                }
            }
        }
    }
    double recall = (float)count / (float)(nq * top_k);
    printf("total_recall: %d ", count);
    return recall;
}

// 计算 Recall@1（扩展数据版本）
double calculate_recall_r1_expanded(
        const std::vector<std::vector<int>>& ground_truth,
        const faiss::idx_t* query_results,
        int nq,
        int top_k,
        size_t nb_orig) {
    int count = 0;
    for (size_t i = 0; i < nq; i++) {
        int gt_first = ground_truth[i][0];
        for (int j = 0; j < top_k; j++) {
            // 将扩展后的id映射回原始id
            faiss::idx_t mapped_id = query_results[i * top_k + j] % nb_orig;
            if (mapped_id == gt_first) {
                count++;
                break;
            }
        }
    }
    printf("total_recall_r1: %d ", count);
    return (float)count / (float)nq;
}

std::vector<float> horizontal_avg(const std::vector<std::vector<float>>& data) {
    size_t rows = data.size();
    size_t cols = data[0].size();

    std::vector<float> avg(cols, 0);
    for (auto& row : data) {
        for (size_t j = 0; j < cols; j++) {
            avg[j] += row[j];
        }
    }

    for (size_t j = 0; j < cols; j++) {
        avg[j] /= rows;
    }
    return avg;
}

// 对向量进行L2归一化
void normalize_vectors(float* data, size_t n, size_t d) {
    for (size_t i = 0; i < n; i++) {
        float* vec = data + i * d;
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += vec[j] * vec[j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (size_t j = 0; j < d; j++) {
                vec[j] /= norm;
            }
        }
    }
}

// 生成随机正交矩阵（使用Gram-Schmidt正交化）
std::vector<float> generate_random_orthogonal_matrix(size_t d, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // 生成随机矩阵 d x d
    std::vector<float> matrix(d * d);
    for (size_t i = 0; i < d * d; i++) {
        matrix[i] = dist(rng);
    }
    
    // Gram-Schmidt正交化（按列进行）
    for (size_t j = 0; j < d; j++) {
        // 减去之前列的投影
        for (size_t k = 0; k < j; k++) {
            float dot = 0.0f;
            for (size_t i = 0; i < d; i++) {
                dot += matrix[i * d + j] * matrix[i * d + k];
            }
            for (size_t i = 0; i < d; i++) {
                matrix[i * d + j] -= dot * matrix[i * d + k];
            }
        }
        
        // 归一化当前列
        float norm = 0.0f;
        for (size_t i = 0; i < d; i++) {
            norm += matrix[i * d + j] * matrix[i * d + j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (size_t i = 0; i < d; i++) {
                matrix[i * d + j] /= norm;
            }
        }
    }
    
    return matrix;
}

// 通过随机旋转矩阵扩展数据量
// 输入: n个d维向量
// 输出: n*multiplier个d维向量（维度不变，数据量变为multiplier倍）
// 方法: 原始数据复制multiplier次，每次应用不同的随机旋转矩阵
std::vector<float> expand_data_with_rotation(
        const float* data, 
        size_t n, 
        size_t d, 
        size_t multiplier,
        unsigned int seed) {
    
    size_t new_n = n * multiplier;
    std::vector<float> expanded(new_n * d);
    
    // 为每个扩展块生成一个随机旋转矩阵
    std::vector<std::vector<float>> rotation_matrices(multiplier);
    for (size_t m = 0; m < multiplier; m++) {
        if (m == 0) {
            // 第一份数据保持不变（单位矩阵）
            rotation_matrices[m].resize(d * d, 0.0f);
            for (size_t i = 0; i < d; i++) {
                rotation_matrices[m][i * d + i] = 1.0f;
            }
        } else {
            // 其他份数据应用随机旋转
            rotation_matrices[m] = generate_random_orthogonal_matrix(d, seed + m * 1000);
        }
    }
    
    // 对每份数据应用旋转
    for (size_t m = 0; m < multiplier; m++) {
        const float* R = rotation_matrices[m].data();
        
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            const float* src = data + i * d;
            float* dst = expanded.data() + (m * n + i) * d;
            
            // 应用旋转: dst = R * src (矩阵向量乘法)
            for (size_t j = 0; j < d; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < d; k++) {
                    sum += R[j * d + k] * src[k];
                }
                dst[j] = sum;
            }
        }
    }
    
    printf("数据量扩展完成: %zu -> %zu (倍数: %zu), 维度保持: %zu\n", n, new_n, multiplier, d);
    return expanded;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        printf("用法: %s <数据集名称> <数据量扩展倍数> <z值> <结果路径前缀> <索引类型> [block_distance] [ordered_block]\n", argv[0]);
        printf("数据集名称: SIFT1M, ImageNet, AgNews, Laion, GooAQ, Gist, Glove 等\n");
        printf("数据量扩展倍数: 1, 2, 4, 6, 8, 10 等（维度不变，数据量变为x倍）\n");
        printf("示例: %s AgNews 2 0.095 /data/results/ ABQ 1 1\n", argv[0]);
        return 1;
    }

    printf("程序: %s\n", argv[0]);
    
    // 解析参数
    std::string dataset_name = argv[1];               // 数据集名称
    size_t data_multiplier = std::stoi(argv[2]);      // 数据量扩展倍数
    float z = std::stof(argv[3]);                     // z值
    std::string result_path_prefix = argv[4];         // 结果路径前缀
    std::string index_type = argv[5];                 // 索引类型
    
    bool block_distance = true;
    bool ordered_block = true;
    if (argc > 6) {
        block_distance = std::stoi(argv[6]);
    }
    if (argc > 7) {
        ordered_block = std::stoi(argv[7]);
    }

    printf("数据集名称: %s\n", dataset_name.c_str());
    printf("数据量扩展倍数: %zu\n", data_multiplier);
    printf("z值: %f\n", z);
    printf("索引类型: %s\n", index_type.c_str());
    printf("block_distance: %d\n", block_distance);
    printf("ordered_block: %d\n", ordered_block);

    // 初始化数据路径映射
    init_data_path_map();

    // 加载数据集（使用data_loader.h的方式）
    size_t d;
    size_t nb_orig;
    size_t nt;
    size_t nq;
    std::unique_ptr<float[]> database;
    std::unique_ptr<float[]> train;
    std::unique_ptr<float[]> queries;
    std::vector<std::vector<int>> ground_truth;
    
    load_ivecs_from_h5(dataset_name, database, train, queries, ground_truth, nb_orig, nt, nq, d);

    printf("原始数据量 nb_orig: %zu\n", nb_orig);
    printf("原始训练数据量 nt: %zu\n", nt);
    printf("查询数量 nq: %zu\n", nq);
    printf("维度 d: %zu\n", d);

    // 通过随机旋转矩阵扩展数据量（维度不变）
    printf("正在通过随机旋转矩阵扩展数据量...\n");
    std::vector<float> expanded_base = expand_data_with_rotation(
            database.get(), nb_orig, d, data_multiplier, 12345);
    
    // 扩展后的数据量
    size_t nb = nb_orig * data_multiplier;
    
    // 对扩展后的向量进行L2归一化（旋转后范数可能有微小变化）
    printf("正在对扩展后的向量进行L2归一化...\n");
//    normalize_vectors(expanded_base.data(), nb, d);

    // 限制查询向量数量
    if (nq > 10000) {
        nq = 10000;
    }
    
    // 训练数据量：从扩展数据集中采样10%
    size_t nt_new = nb / 10;
    if (nt_new < 10000) {
        nt_new = std::min(nb, (size_t)10000);
    }
    if (nt_new > 500000) {
        nt_new = 500000;  // 最大100万
    }

    printf("\n=== 数据集配置 ===\n");
    printf("维度 d: %zu\n", d);
    printf("原始数据量 nb_orig: %zu (%.2fW)\n", nb_orig, nb_orig / 10000.0);
    printf("扩展后数据量 nb: %zu (%.2fW, %zux)\n", nb, nb / 10000.0, data_multiplier);
    printf("训练数据量 nt: %zu (%.2fW, 占比%.1f%%)\n", nt_new, nt_new / 10000.0, nt_new * 100.0 / nb);
    printf("查询数量 nq: %zu\n", nq);

    // 计算 ncentroids
    int ncentroids = 4 * (int)std::sqrt(nb);
    if (ncentroids > 100000) {
        ncentroids = 100000;
    }
    printf("ncentroids: %d\n", ncentroids);

    int test_round = 2;
    int k = 10;

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(1));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(1));
    std::vector<std::vector<float>> all_ndis(test_round, std::vector<float>(1));

    // 构造结果文件路径
    std::string result_path_str = result_path_prefix + dataset_name + "_x" + 
                                  std::to_string(data_multiplier) + "_" + 
                                  index_type + "_scalability.txt";
    FILE* result_file = fopen(result_path_str.c_str(), "w");
    if (!result_file) {
        printf("无法打开结果文件: %s\n", result_path_str.c_str());
        return 1;
    }

    // 创建索引
    faiss::IndexFlatL2 coarse_quantizer(d);
    std::unique_ptr<faiss::Index> index;
    std::string index_name = "IVF";
    
    if (index_type == "ABQ" || index_type == "ABQ_A") {
        bool bd = (index_type == "ABQ_A") ? false : block_distance;
        index = std::make_unique<faiss::ABQSIMD>(&coarse_quantizer, d, ncentroids, bd, ordered_block);
    } else if (index_type == "ABQ_SQ" || index_type == "ABQ_D") {
        bool bd = (index_type == "ABQ_D") ? true : block_distance;
        index = std::make_unique<faiss::ABQ_SQ>(&coarse_quantizer, d, ncentroids, bd, ordered_block);
    } else if (index_type == "PQ") {
        index_name += std::to_string(ncentroids);
        index_name += ",PQ";
        index_name += std::to_string(d);
        index_name += "x8";
    } else if (index_type == "OPQ") {
        index_name = "OPQ16,IVF";
        index_name += std::to_string(ncentroids);
        index_name += ",PQ";
        index_name += std::to_string(d);
        index_name += "x8";
    } else if (index_type == "SQ") {
        index_name += std::to_string(ncentroids);
        index_name += ",SQ8";
    } else if (index_type == "PQFS") {
        index_name += std::to_string(ncentroids);
        index_name += ",PQ";
        index_name += std::to_string(d);
        index_name += "x4fs";
    } else if (index_type == "FLAT") {
        index_name += std::to_string(ncentroids);
        index_name += ",Flat";
    } else {
        printf("不支持的索引类型: %s\n", index_type.c_str());
        fclose(result_file);
        return 1;
    }

    faiss::IndexIVF* index_ivf = nullptr;
    if (index_type == "ABQ" || index_type == "ABQ_SQ" || index_type == "ABQ_A" || index_type == "ABQ_D") {
        index_ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
    } else {
        index = std::unique_ptr<faiss::Index>(faiss::index_factory(d, index_name.c_str()));
        if (index_type == "OPQ") {
            auto pre_transform = dynamic_cast<faiss::IndexPreTransform*>(index.get());
            if (pre_transform) {
                index_ivf = dynamic_cast<faiss::IndexIVF*>(pre_transform->index);
            }
        } else {
            index_ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
        }
    }

    // 训练索引：从扩展数据集中采样10%
    printf("开始训练索引...\n");
    {
        index_ivf->verbose = true;
        // 从数据库中均匀采样训练数据
        std::vector<float> train_data(nt_new * d);
        std::vector<size_t> train_indices(nb);
        for (size_t i = 0; i < nb; i++) {
            train_indices[i] = i;
        }
        // 使用固定种子打乱索引
        std::mt19937 rng_train(789);
        std::shuffle(train_indices.begin(), train_indices.end(), rng_train);
        // 取前nt_new个作为训练数据
        for (size_t i = 0; i < nt_new; i++) {
            std::copy(expanded_base.data() + train_indices[i] * d, 
                      expanded_base.data() + (train_indices[i] + 1) * d, 
                      train_data.data() + i * d);
        }
        index->train(nt_new, train_data.data());
    }
    printf("索引训练完成\n");

    // 添加数据到索引
    printf("开始添加数据到索引...\n");
    {
        index->add(nb, expanded_base.data());
        index_ivf->make_direct_map(true);
    }
    printf("数据添加完成\n");

    // 对ABQ类型索引使用从数据库采样的向量重新拟合z值参数
    if (index_type == "ABQ" || index_type == "ABQ_A") {
        const size_t refit_nq = 5000;
        std::unique_ptr<float[]> refit_queries(new float[refit_nq * d]);
        
        std::vector<size_t> refit_indices(nb);
        for (size_t i = 0; i < nb; i++) {
            refit_indices[i] = i;
        }
        std::mt19937 rng_refit(456);
        std::shuffle(refit_indices.begin(), refit_indices.end(), rng_refit);
        for (size_t i = 0; i < refit_nq && i < nb; i++) {
            std::copy(expanded_base.data() + refit_indices[i] * d, 
                      expanded_base.data() + (refit_indices[i] + 1) * d, 
                      refit_queries.get() + i * d);
        }
        
        printf("开始使用 %zu 个从数据库采样的向量重新拟合z值参数...\n", refit_nq);
        faiss::ABQSIMD* abq_index = dynamic_cast<faiss::ABQSIMD*>(index.get());
        if (abq_index) {
            // abq_index->refit_z_params(refit_nq, refit_queries.get());
            printf("z值参数拟合完成\n");
        }
    }

    // 休眠一段时间让系统稳定
    timespec delay = {10, 0};
    nanosleep(&delay, NULL);

    // 设置 nprobe
    int nprobe = ncentroids;
    printf("nprobe: %d\n", nprobe);

    long ndis = 0;
    double compute_time = 0.0f;
    double pruning_time = 0.0f;
    double q_time = 0.0f;
    double bound_time = 0.0f;
    double bound_time_ip = 0.0f;
    double search_time_2 = 0.0f;
    int n_scanned = 0;
    int c_scanned = 0;

    faiss::idx_t* nns = new faiss::idx_t[k * nq];
    float* dis = new float[k * nq];

    for (int r = 0; r < test_round; r++) {
        printf("\n=== 测试轮次 %d/%d ===\n", r + 1, test_round);
        
        faiss::SearchParametersIVF params;
        params.nprobe = nprobe;
        
        double t1 = elapsed();
#pragma omp parallel for schedule(dynamic)
        for (int ii = 0; ii < nq; ii++) {
            faiss::idx_t* nns1 = nns + ii * k;
            float* dis1 = dis + ii * k;
            index_ivf->query(
                    1,
                    queries.get() + ii * d,
                    k,
                    dis1,
                    nns1,
                    z,
                    &params);
        }
        double search_time = elapsed() - t1;

        // 使用扩展数据版本的召回率计算（将id映射回原始id）
        double recall = calculate_recall_expanded(ground_truth, nns, nq, k, nb_orig);
        double recall_1 = calculate_recall_r1_expanded(ground_truth, nns, nq, k, nb_orig);
        
        printf("--------------\n");
        printf("[recall@%d]: %.4f\n", k, recall);
        printf("[recall_R1@%d]: %.4f\n", k, recall_1);
        
        auto stats = index_ivf->get_ivf_stats();
        printf("ncentroids: %ld\n", index_ivf->nlist);
        printf("nprobe: %ld\n", params.nprobe);
        printf("quantization_time: %lf\n", stats.quantization_time - q_time);
        printf("compute_time: %lf\n", stats.compute_time - compute_time);
        printf("pruning_time: %lf\n", stats.pruning_time - pruning_time);
        printf("bound_time: %lf\n", stats.bound_time - bound_time);
        printf("bound_time_ip: %lf\n", stats.bound_time_ip - bound_time_ip);
        printf("search_time_ivf: %lf\n", stats.search_time - search_time_2);
        printf("n_scanned: %d\n", stats.nlist - n_scanned);
        printf("c_scanned: %d\n", stats.c_list - c_scanned);
        printf("search_time: %lf\n", search_time);
        printf("ndis: %ld\n", stats.ndis - ndis);
        printf("QPS: %f\n", nq / search_time);

        all_qps[r][0] = nq / search_time;
        all_recall[r][0] = recall;
        all_ndis[r][0] = stats.ndis - ndis;

        ndis = stats.ndis;
        compute_time = stats.compute_time;
        pruning_time = stats.pruning_time;
        q_time = stats.quantization_time;
        search_time_2 = stats.search_time;
        bound_time = stats.bound_time;
        bound_time_ip = stats.bound_time_ip;
        n_scanned = stats.nlist;
        c_scanned = stats.c_list;

        timespec delay = {5, 0};
        nanosleep(&delay, NULL);
    }

    delete[] nns;
    delete[] dis;

    auto avg_qps = horizontal_avg(all_qps);
    auto avg_recall = horizontal_avg(all_recall);
    auto avg_ndis = horizontal_avg(all_ndis);

    printf("\n\n========== 最终结果 ==========\n");
    printf("数据集\t维度\t扩展倍数\t数据量(万)\tQPS\t召回率\tndis\n");
    printf("%s\t%zu\t%zu\t%.2f\t%.2f\t%.4f\t%.0f\n", 
           dataset_name.c_str(), d, data_multiplier, nb / 10000.0, 
           avg_qps[0], avg_recall[0], avg_ndis[0]);

    // 写入结果文件
    fprintf(result_file, "# 配置信息\n");
    fprintf(result_file, "dataset\t%s\n", dataset_name.c_str());
    fprintf(result_file, "dimension\t%zu\n", d);
    fprintf(result_file, "original_database_size\t%zu\n", nb_orig);
    fprintf(result_file, "expanded_database_size\t%zu\n", nb);
    fprintf(result_file, "data_multiplier\t%zu\n", data_multiplier);
    fprintf(result_file, "train_size\t%zu\n", nt_new);
    fprintf(result_file, "query_size\t%zu\n", nq);
    fprintf(result_file, "ncentroids\t%d\n", ncentroids);
    fprintf(result_file, "nprobe\t%d\n", nprobe);
    fprintf(result_file, "index_type\t%s\n", index_type.c_str());
    fprintf(result_file, "\n# 结果\n");
    fprintf(result_file, "QPS\t%.2f\n", avg_qps[0]);
    fprintf(result_file, "recall@%d\t%.4f\n", k, avg_recall[0]);
    fprintf(result_file, "ndis\t%.0f\n", avg_ndis[0]);

    fclose(result_file);
    printf("结果已保存到: %s\n", result_path_str.c_str());
    
    return 0;
}
