//
// Created by root on 11/13/25.
//
// Experiments with different nprobe values
// ABQ，不同nprobe值的性能, 默认的nlist为 sqrt(n)
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
//#include <faiss/IndexIVFRaBitQ.h>
//#include <faiss/IndexIVFPQFastScan.h>
#include <thread>
#include <faiss/index_factory.h>
//#include <faiss/IndexIVFPQBlock.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/MetricType.h>
#include <faiss/index_io.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <fstream>
#include <map>     // 添加map头文件
#include <utility> // 添加pair头文件
// #include "faiss/IndexIVFPQ.h"
#include <H5Cpp.h>
#include <faiss/utils/utils.h>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#include "data_loader.h"
double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double calculate_recall(
        const std::vector<std::vector<int>>& ground_truth,
        const faiss::idx_t* query_results,
        int nq,
        int top_k) {
    double total_recall = 0.0;
    int num_queries = nq;
    int count = 0;
    for (size_t i = 0; i < num_queries; i++) {
        // 取模型返回的前 top_k 个结果
        std::vector<int> gt = ground_truth[i];
        for (int j = 0; j < top_k; j++) {
            size_t v1 = query_results[i * top_k + j];
            for (int k = 0; k < top_k; k++) {
                size_t v2 = gt[k];
                if (v1 == v2) {
                    count++;
                    break;
                }
            }
        }
    }

    double recall = (float) count / (float) (num_queries *  top_k); // 除以真值的 k
    printf("total_recall: %f" , count);
    return recall; // 平均召回率
}

// 计算平均召回率 (Recall@K)
double calculate_recall_r1(
        const std::vector<std::vector<int>>& ground_truth,
        const faiss::idx_t* query_results,
        int nq,
        int top_k) {
    double total_recall = 0.0;
    int num_queries = nq;
    int count = 0;
    for (size_t i = 0; i < num_queries; i++) {
        // 取模型返回的前 top_k 个结果
        std::vector<int> model_top_k(
                query_results + static_cast<long>(i * top_k),
                query_results + static_cast<long>((i + 1) * top_k));
        std::vector<int> gt = ground_truth[i];

        // 排序以便计算交集
        //        std::sort(model_top_k.begin(), model_top_k.end());
        //        std::sort(gt.begin(), gt.end());

        for (const auto& item : model_top_k) {
            if (item == gt[0]) {
                count++;
                break;
            }
        }
    }

    printf("total_recall: %d" , count);
    return (float) count / (float)num_queries; // 平均召回率
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


int main(int argc, char* argv[]) {
    printf("z: %d\n", argv[0]);
    printf("dataset: %s\n", argv[1]);
    printf("dataset: %s\n", argv[2]);
    char* dataset = argv[2];
    float z = std::stof(argv[3]);

    size_t d;
    size_t nb;
    size_t nt;
    size_t nq;
    std::unique_ptr<float[]> database;
    std::unique_ptr<float[]> train;
    std::unique_ptr<float[]> queries;
    std::vector<std::vector<int>> ground_truth;
    init_data_path_map();
    load_ivecs_from_h5(dataset, database, train,  queries, ground_truth, nb, nt, nq,d);

    printf("nb: %d\n", nb);
    printf("nt: %d\n", nt);
    printf("nq: %d\n", nq);
    printf("d: %d\n", d);
    int ncentroids = (int) std::sqrt(nb);
    std::vector<size_t> all_ncentroids;
    all_ncentroids.push_back(ncentroids);
//    all_ncentroids.push_back(2 * ncentroids);
//    all_ncentroids.push_back(3 * ncentroids);
//    all_ncentroids.push_back(4 * ncentroids);
//    all_ncentroids.push_back(5 * ncentroids);

    int test_round = 3;
    int length = 1;

    std::vector<std::vector<float>> all_query_latency(
            test_round, std::vector<float>(4));
    std::vector<std::vector<float>> all_recall(
            test_round, std::vector<float>(4));
    std::vector<std::vector<float>> all_ndis(
            test_round, std::vector<float>(4));

    std::string index_type = std::string(argv[5]);
    std::string result_path_str = std::string(argv[4]) + dataset + "_" + index_type + "_query_latency.txt";
    FILE* result_file = fopen(result_path_str.c_str(), "w");
    // 新增：读取索引类型参数
    printf("index_type: %s\n", index_type.c_str());
    bool block_distance = true;
    bool ordered_block = true;
    if (argc > 6) {
        block_distance = std::stoi(argv[6]);
    }
    if (argc > 7) {
        ordered_block = std::stoi(argv[7]);
    }
    for (int i = 0; i < length; i++) {
        faiss::IndexFlatL2 coarse_quantizer(d);
//        faiss::ABQ_SQ index(&coarse_quantizer, d, all_ncentroids[i]);
        // 根据index_type创建不同类型的索引
        std::unique_ptr<faiss::Index> index;
        std::string index_name = "IVF";
        if (index_type == "ABQ") {
            index = std::make_unique<faiss::ABQSIMD>(&coarse_quantizer, d, ncentroids, block_distance, ordered_block);
        } else if (index_type == "ABQ_SQ") {
            index = std::make_unique<faiss::ABQ_SQ>(&coarse_quantizer, d, ncentroids, block_distance, ordered_block);
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
            index_name = "unsupported";
            printf("unsupported index type: %s\n", index_type.c_str());
            exit(1);
        }

        faiss::IndexIVF* index_ivf = nullptr;
        if (index_type == "ABQ" || index_type == "ABQ_SQ") {
            index_ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
        } else {
            index = std::unique_ptr<faiss::Index>(faiss::index_factory(d, index_name.c_str()));
            // 处理OPQ索引类型
            if (index_type == "OPQ") {
                auto pre_transform = dynamic_cast<faiss::IndexPreTransform*>(index.get());
                if (pre_transform) {
                    index_ivf = dynamic_cast<faiss::IndexIVF*>(pre_transform->index);
                }
            } else {
                index_ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
            }
        }

        { // training
            index_ivf->verbose = true;
            //        index.train(nt, train.data());
            index->train(nt, train.get());
        }
        { // populating the database
            index->add(nb, database.get());
            // Initialize direct map for reconstructing vectors
//            index_ivf->make_direct_map(true);
        }

        timespec delay = {30, 0};
        nanosleep(&delay, NULL);
        double compute_time = 0.0f;
        double pruning_time = 0.0f;
        double q_time = 0.0f;
        double bound_time = 0.0f;
        double bound_time_ip = 0.0f;
        double t_sort = 0.0f;
        int n_scanned = 0;
        int c_scanned = 0;
        long ndis = 0;
        int k = 10;
        int nprobe = 10;
        faiss::SearchParametersIVF params;
        // params.inverted_list_context = &context;
        params.nprobe = ncentroids;
        //    all_nprobes.push_back(10);
        //        all_nprobes.push_back(1000);
        //    all_nprobes.push_back(ncentroids);

//        std::vector<size_t> all_nprobes;
//        all_nprobes.push_back((int)((float)ncentroids * (float)100.0f / 100.0));

        faiss::idx_t* nns = new faiss::idx_t[k * nq];
        float* dis = new float[k * nq];
        double search_time = 0;
        double search_time_2 = 0;
        std::vector<double> times(nq);
        std::vector<double> ndis_v(nq);
        //    omp_set_num_threads(20);
        float* q = queries.get();
        std::vector<double> all_times(nq);
        printf("\n-------query-------\n");
        for (int r = 0; r < test_round; r++) {
            double t1 = elapsed();
            double t1_1 = faiss::getmillisecs();
#pragma omp parallel for schedule(dynamic)
            for (int ii = 0; ii < nq; ii++) {
                faiss::idx_t* nns1 = nns + ii * k;
                float* dis1 = dis + ii * k;
                double t_start = faiss::getmillisecs();
                index_ivf->query(1, q + ii * d, k, dis1, nns1, z, &params);
                all_times[ii] = faiss::getmillisecs() - t_start;
            }

            search_time = elapsed() - t1;
            std::map<std::pair<int, faiss::idx_t>, std::vector<float>>
                    query_centroid_dists;

            double recall = calculate_recall(ground_truth, nns, nq, k);
            //            double recall_gt = calculate_recall_gt(gt, nns, nq, k);
            double recall_1 = calculate_recall_r1(ground_truth, nns, nq, k);
            printf("--------------\n");
            printf("--------------\n");
            //                printf("iter_factor: %lf\n", iter_factors[i]);
            printf("[recall]: %.3f\n", recall);
            printf("[recall_R1@10]: %.3f\n", recall_1);
            auto stats = index_ivf->get_ivf_stats();
            printf("centriod: %ld\n", index_ivf->nlist);
            printf("nprobe: %ld\n", params.nprobe);
            printf("quantization_time: %lf\n",
                   stats.quantization_time - q_time);
            printf("compute_time: %lf\n",
                   stats.compute_time - compute_time);
            printf("pruning_time: %lf\n",
                   stats.pruning_time - pruning_time);
            printf("bound_time: %lf\n", stats.bound_time - bound_time);
            printf("bound_time_ip: %lf\n",
                   stats.bound_time_ip - bound_time_ip);
            //            printf("t_sort: %lf\n", stats.t_sort - t_sort);
            printf("search_time_ivf: %lf\n",
                   stats.search_time - search_time_2);
            printf("n_scanned: %d\n", stats.nlist - n_scanned);
            printf("c_scanned: %d\n", stats.c_list - c_scanned);
            printf("search_time2: %lf\n", search_time);
            printf("ndis: %ld\n", stats.ndis - ndis);
            printf("qps : %f\n", nq / search_time);
            //            auto& stats = index.get_ivf_stats();


            //计算70%，80%，90%，99%的查询延迟
            std::sort(all_times.begin(), all_times.end());

            int per_index = 0;
            for (int per = 70; per <= 100; per += 10) {
                if (per == 100) {
                    per = 99;
                }
                int idx = all_times.size() * per / 100 - 1;
                all_query_latency[r][per_index++] = all_times[idx];
                printf("per\t%d\t%lf\n", per, all_times[idx]);
            }

            all_recall[r][i] = recall;
            all_ndis[r][i] = stats.ndis - ndis;
            ndis = stats.ndis;

            compute_time = stats.compute_time;
            pruning_time = stats.pruning_time;
            q_time = stats.quantization_time;
            search_time_2 = stats.search_time;
            bound_time = stats.bound_time;
            bound_time_ip = stats.bound_time_ip;
            //            t_sort = stats.t_sort;
            n_scanned = stats.nlist;
            c_scanned = stats.c_list;
            sort(times.begin(), times.end());
            sort(ndis_v.begin(), ndis_v.end());
            for (int per = 70; per <= 100; per += 5) {
                int idx = times.size() * per / 100 - 1;
                // printf("per\t%d\t%lf\tndis\t%lf\n", per, times[idx], ndis_v[idx]);
            }
            timespec delay = {5, 0};
            nanosleep(&delay, NULL);
        }
    }

    auto avg_time_latency = horizontal_avg(all_query_latency);
    auto avg_recall = horizontal_avg(all_recall);
    auto avg_ndis = horizontal_avg(all_ndis);

    printf("\n\nper\tquery_latency\n");

    for (size_t i = 0; i < 4; ++i) {
        printf("%zu\t%f\n",70 + i * 10, avg_time_latency[i]);
    }

    // 新增：将结果写入CSV文件
    // 第一行：训练数据量
    fprintf(result_file, "per");
    for (size_t i = 0; i < 4; i++) {
        int per = 70 + i * 10;
        if (per == 100) {
            per = 99;
        }
        fprintf(result_file, "\n");
        fprintf(result_file, "%d", per);
        fprintf(result_file, "\t%f", avg_time_latency[i]);
    }
    fclose(result_file);
    return 0;
}