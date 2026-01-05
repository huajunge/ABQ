//
// Created by root on 11/13/25.
//
// Experiments with different nprobe values
// ABQ，不同nprobe值的性能, 默认的nlist为 sqrt(n)
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

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 计算召回率（使用暴力搜索结果作为ground truth）
double calculate_recall(
        const faiss::idx_t* ground_truth,
        const faiss::idx_t* query_results,
        int nq,
        int top_k) {
    int count = 0;
    for (size_t i = 0; i < nq; i++) {
        for (int j = 0; j < top_k; j++) {
            faiss::idx_t v1 = query_results[i * top_k + j];
            for (int k = 0; k < top_k; k++) {
                faiss::idx_t v2 = ground_truth[i * top_k + k];
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

// 计算 Recall@1
double calculate_recall_r1(
        const faiss::idx_t* ground_truth,
        const faiss::idx_t* query_results,
        int nq,
        int top_k) {
    int count = 0;
    for (size_t i = 0; i < nq; i++) {
        faiss::idx_t gt_first = ground_truth[i * top_k];
        for (int j = 0; j < top_k; j++) {
            if (query_results[i * top_k + j] == gt_first) {
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

// 随机生成向量数据
void generate_random_vectors(float* data, size_t n, size_t d, unsigned int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n * d; i++) {
        data[i] = dist(rng);
    }
    
    // 对每个向量进行L2归一化
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

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printf("用法: %s <维度d> <数据量n(万)> <z值> <结果路径前缀> <索引类型> [block_distance] [ordered_block]\n", argv[0]);
        printf("示例: %s 128 1000 0.095 /data/results/ ABQ 1 1\n", argv[0]);
        return 1;
    }

    printf("程序: %s\n", argv[0]);
    
    // 解析参数
    size_t d = std::stoi(argv[1]);                    // 维度
    size_t n_wan = std::stoll(argv[2]);               // 数据量（万）
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

    // 计算实际数据量
    size_t nb = n_wan * 10000;                        // 数据库向量数量
    size_t nq = 1000;                               // 查询向量数量（10万，用于refit）
    
    // 训练数据量：小于100M时5M，大于等于100M时10M
    size_t nt;
    if (nb < 1000000) {  // 100M = 10000万
        nt = 50000;      // 5M
    } else {
        nt = 100000;     // 10M
    }
    // 训练数据量不能超过数据库大小
    if (nt > nb) {
        nt = nb;
    }

    printf("维度 d: %zu\n", d);
    printf("数据库大小 nb: %zu (%.2fW)\n", nb, nb / 10000.0);
    printf("训练数据量 nt: %zu (%.2fW)\n", nt, nt / 10000.0);
    printf("查询数量 nq: %zu\n", nq);
    printf("索引类型: %s\n", index_type.c_str());
    printf("block_distance: %d\n", block_distance);
    printf("ordered_block: %d\n", ordered_block);

    // 分配并生成随机数据
    printf("正在生成随机数据...\n");
    std::unique_ptr<float[]> database(new float[nb * d]);
    std::unique_ptr<float[]> queries(new float[nq * d]);
    
    generate_random_vectors(database.get(), nb, d, 42);
    
    // 从数据库中均匀采样查询向量
    std::vector<size_t> query_indices(nb);
    for (size_t i = 0; i < nb; i++) {
        query_indices[i] = i;
    }
    std::mt19937 rng_query(123);  // 使用不同的seed
    std::shuffle(query_indices.begin(), query_indices.end(), rng_query);
    for (size_t i = 0; i < nq && i < nb; i++) {
        std::copy(database.get() + query_indices[i] * d, 
                  database.get() + (query_indices[i] + 1) * d, 
                  queries.get() + i * d);
    }
    printf("数据生成完成（查询向量从数据库采样）\n");

    // 计算 ncentroids
    int ncentroids = (int)std::sqrt(nb);
    printf("ncentroids: %d\n", ncentroids);

    int test_round = 2;
    int k = 10;

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(1));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(1));
    std::vector<std::vector<float>> all_ndis(test_round, std::vector<float>(1));

    // 构造结果文件路径
    std::string result_path_str = result_path_prefix + "d" + std::to_string(d) + 
                                  "_n" + std::to_string(n_wan) + "w_" + 
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
        // ABQ_A 表示 block_distance = 0
        bool bd = (index_type == "ABQ_A") ? false : block_distance;
        index = std::make_unique<faiss::ABQSIMD>(&coarse_quantizer, d, ncentroids, bd, ordered_block);
    } else if (index_type == "ABQ_SQ" || index_type == "ABQ_D") {
        // ABQ_D 表示 block_distance = 1
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

    // 训练索引
    printf("开始训练索引...\n");
    {
        index_ivf->verbose = true;
        // 从数据库中均匀采样训练数据
        std::vector<float> train_data(nt * d);
        std::vector<size_t> train_indices(nb);
        for (size_t i = 0; i < nb; i++) {
            train_indices[i] = i;
        }
        // 使用固定种子打乱索引
        std::mt19937 rng_train(789);
        std::shuffle(train_indices.begin(), train_indices.end(), rng_train);
        // 取前nt个作为训练数据
        for (size_t i = 0; i < nt; i++) {
            std::copy(database.get() + train_indices[i] * d, 
                      database.get() + (train_indices[i] + 1) * d, 
                      train_data.data() + i * d);
        }
        index->train(nt, train_data.data());
    }
    printf("索引训练完成\n");

    // 添加数据到索引
    printf("开始添加数据到索引...\n");
    {
        index->add(nb, database.get());
        index_ivf->make_direct_map(true);
    }
    printf("数据添加完成\n");

    // 对ABQ类型索引使用从数据库采样的向量重新拟合z值参数
    if (index_type == "ABQ" || index_type == "ABQ_A") {
        const size_t refit_nq = 5000;  // 用于refit的向量数量
        std::unique_ptr<float[]> refit_queries(new float[refit_nq * d]);
        
        // 从数据库中均匀采样向量用于refit
        std::vector<size_t> refit_indices(nb);
        for (size_t i = 0; i < nb; i++) {
            refit_indices[i] = i;
        }
        std::mt19937 rng_refit(456);  // 使用不同的seed
        std::shuffle(refit_indices.begin(), refit_indices.end(), rng_refit);
        for (size_t i = 0; i < refit_nq && i < nb; i++) {
            std::copy(database.get() + refit_indices[i] * d, 
                      database.get() + (refit_indices[i] + 1) * d, 
                      refit_queries.get() + i * d);
        }
        
        printf("开始使用 %zu 个从数据库采样的向量重新拟合z值参数...\n", refit_nq);
        faiss::ABQSIMD* abq_index = dynamic_cast<faiss::ABQSIMD*>(index.get());
        if (abq_index) {
//            abq_index->refit_z_params(refit_nq, refit_queries.get());
            printf("z值参数拟合完成\n");
        }
    }

    // 使用暴力搜索计算ground truth
    printf("正在计算 ground truth...\n");
    std::unique_ptr<faiss::idx_t[]> ground_truth(new faiss::idx_t[nq * k]);
    std::unique_ptr<float[]> gt_distances(new float[nq * k]);
    {
        faiss::IndexFlatL2 flat_index(d);
        flat_index.add(nb, database.get());
        flat_index.search(nq, queries.get(), k, gt_distances.get(), ground_truth.get());
    }
    printf("ground truth 计算完成\n");

    // 休眠一段时间让系统稳定
    timespec delay = {10, 0};
    nanosleep(&delay, NULL);

    // 设置 nprobe = ncentroids
    int nprobe = ncentroids;
    printf("nprobe: %d (= ncentroids)\n", nprobe);

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
    float* q = queries.get();

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
                    q + ii * d,
                    k,
                    dis1,
                    nns1,
                    z,
                    &params);
        }
        double search_time = elapsed() - t1;

        double recall = calculate_recall(ground_truth.get(), nns, nq, k);
        double recall_1 = calculate_recall_r1(ground_truth.get(), nns, nq, k);
        
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
    printf("维度\t数据量(万)\tQPS\t召回率\tndis\n");
    printf("%zu\t%zu\t%.2f\t%.4f\t%.0f\n", d, n_wan, avg_qps[0], avg_recall[0], avg_ndis[0]);

    // 写入结果文件
    fprintf(result_file, "# 配置信息\n");
    fprintf(result_file, "dimension\t%zu\n", d);
    fprintf(result_file, "database_size\t%zu\n", nb);
    fprintf(result_file, "train_size\t%zu\n", nt);
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