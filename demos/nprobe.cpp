//
// Created by root on 11/13/25.
//
// Experiments with different nprobe values
// ABQ，不同nprobe值的性能, 默认的nlist为 sqrt(n)
#include <H5Cpp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include <omp.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
//#include <faiss/ABQSIMD.h>
//#include <faiss/ABQ_SQ.h>
//#include <faiss/ABQ_PQFastScan.h>
//#include <faiss/IndexIVFRaBitQ.h>
//#include <faiss/IndexIVFPQFastScan.h>
#include <thread>
//#include <faiss/IndexIVFPQBlock.h>
// #include <faiss/IndexIVFPQFastScan.h>
#include <faiss/MetricType.h>
#include <faiss/index_io.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexPreTransform.h>
#include <algorithm>
#include <fstream>
#include <map>     // 添加map头文件
#include <utility> // 添加pair头文件
#include <faiss/ABQSIMD.h>
#include <faiss/ABQ_SQ.h>
#include <H5Cpp.h>
#include <faiss/utils/utils.h>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#include "demos/data_loader.h"
#include <faiss/index_factory.h>
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
        const std::vector<int>& gt = ground_truth[i];
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
    printf("num_queries: %d, calculate_recall: total_recall: %d\n", num_queries, count);
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
        for (size_t j = 0; j < cols; ++j) {
            avg[j] += row[j];
        }
    }

    for (size_t j = 0; j < cols; ++j) {
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

    std::string index_type = std::string(argv[5]);

    std::string result_path_str = std::string(argv[4]) + dataset + "_" + index_type + "_nprobe.txt";
    FILE* result_file = fopen(result_path_str.c_str(), "w");
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
//    faiss::IndexFlatL2 coarse_quantizer(d);
////        faiss::IndexIVFPQFastScan index(&coarse_quantizer, d, ncentroids, 128, 4);
////        faiss::IndexIVFFlat index(&coarse_quantizer, d, ncentroids);
////        faiss::IndexIVFPQ index(&coarse_quantizer, d, ncentroids);
//    faiss::IndexIVFPQ index(&coarse_quantizer, d, ncentroids, d, 8);

    std::unique_ptr<faiss::Index> index_;
    faiss::IndexFlatL2 coarse_quantizer(d);
    bool block_distance = true;
    bool ordered_block = true;
    if (argc > 6) {
        block_distance = std::stoi(argv[6]);
    }
    if (argc > 7) {
        ordered_block = std::stoi(argv[7]);
    }
    std::string index_name = "IVF";
    if (index_type == "ABQ") {
        index_ = std::make_unique<faiss::ABQSIMD>(&coarse_quantizer, d, ncentroids, block_distance, ordered_block);
    } else if (index_type == "ABQ_SQ") {
        index_ = std::make_unique<faiss::ABQ_SQ>(&coarse_quantizer, d, ncentroids, block_distance, ordered_block);
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
    } else if (index_type == "HNSW") {
        // 对于HNSW，使用纯HNSW索引（默认Flat存储），不再附带IVF前缀
        index_name = "HNSW32";
    } else {
        index_name = "unsupported";
        printf("unsupported index type: %s\n", index_type.c_str());
        exit(1);
    }

    bool is_ivf = true;
    faiss::IndexIVF* index_ivf = nullptr;
    if (index_type == "ABQ" || index_type == "ABQ_SQ") {
        index_ivf = dynamic_cast<faiss::IndexIVF*>(index_.get());
    } else {
        index_ = std::unique_ptr<faiss::Index>(faiss::index_factory(d, index_name.c_str()));
        // 处理OPQ索引类型
        if (index_type == "OPQ") {
            auto pre_transform = dynamic_cast<faiss::IndexPreTransform*>(index_.get());
            if (pre_transform) {
                index_ivf = dynamic_cast<faiss::IndexIVF*>(pre_transform->index);
            }
        } else if (index_type != "HNSW"){
            index_ivf = dynamic_cast<faiss::IndexIVF*>(index_.get());
        } else {
            is_ivf = false;
        }
    }

    if (is_ivf && !index_ivf) {
        printf("Error: Failed to cast to IVF index!\\n");
        exit(1);
    }

    { // training
        if (is_ivf && index_ivf) {
            index_ivf->verbose = true;
        } else {
            index_->verbose = true;
        }
        //        index.train(nt, train.data());
        printf("-----index_->train(nt, train.get())\\n");
        index_->train(nt, train.get());
//        index_->train(nb, database.get());
    }
    { // populating the database
        printf("-----index_->add(nb, database.get())\\n");
        index_->add(nb, database.get());
        // Initialize direct map for reconstructing vectors
//        index->make_direct_map(true);
    }

    timespec delay = {30, 0};
    nanosleep(&delay, NULL);

    int k = 10;
    int nprobe = 10;
    faiss::SearchParametersIVF ivf_params;
    faiss::SearchParametersHNSW hnsw_params;
    // params.inverted_list_context = &context;
    ivf_params.nprobe = nprobe;

    // IVF 和 HNSW 使用不同的扫描参数列表：
    // - IVF: 使用 nprobe 列表（基于 ncentroids 的比例）
    // - HNSW: 使用独立的 efSearch 列表
    std::vector<size_t> probe_values;
    std::vector<size_t> all_nprobes;
    std::vector<int> all_efSearch;

    if (is_ivf) {
        all_nprobes.push_back(
                (size_t)((float)ncentroids * (float)5.0f / 1000.0f));

        for (int i = 1; i <= 5; i++) {
            all_nprobes.push_back(
                    (size_t)((float)ncentroids * (float)i / 100.0f));
        }

        for (int i = 1; i <= 10; i++) {
            all_nprobes.push_back(
                    (size_t)((float)ncentroids * (float)i / 10.0f));
        }
//        all_nprobes.push_back(ncentroids);
        probe_values = all_nprobes;
    } else {
        // HNSW: 手动指定一组递增的 efSearch，用于评估不同 efSearch 下的性能
        all_efSearch = {16, 32, 64, 96, 128, 256, 384, 512, 640, 1024};
        probe_values.assign(all_efSearch.begin(), all_efSearch.end());
    }

//    all_nprobes.push_back(10);
    //        all_nprobes.push_back(1000);
//    all_nprobes.push_back(ncentroids);
    int test_round = 2;
    int length = probe_values.size();
    std::vector<std::vector<float>> all_qps(
            test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(
            test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_ndis(
            test_round, std::vector<float>(length));

    std::vector<std::vector<float>> all_positive_c(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_pruning_time_rate(test_round, std::vector<float>(length));
    long ndis = 0;
    faiss::idx_t* nns = new faiss::idx_t[k * nq];
    float* dis = new float[k * nq];
    double search_time = 0;
    double search_time_2 = 0;
    std::vector<double> times(nq);
    std::vector<double> ndis_v(nq);


    printf("-----index_ivf->query(\n");
    //    omp_set_num_threads(20);
    float* q = queries.get();
    double compute_time = 0.0f;
    double pruning_time = 0.0f;
    double q_time = 0.0f;
    double bound_time = 0.0f;
    double bound_time_ip = 0.0f;
    double t_sort = 0.0f;
    int n_scanned = 0;
    int c_scanned = 0;
    for (int r = 0; r < test_round; r++) {
        for (int i = 0; i < length; i++) {
            size_t cur_probe = probe_values[i];
            if (is_ivf) {
                ivf_params.nprobe = cur_probe;
            } else {
                hnsw_params.efSearch = (int)cur_probe;
                // 每次实验前重置HNSW统计量，只统计当前nprobe/efSearch下的搜索代价
                faiss::hnsw_stats.reset();
            }
            double t1 = elapsed();
            double t1_1 = faiss::getmillisecs();
#pragma omp parallel for schedule(dynamic)
            for (int ii = 0; ii < nq; ii++) {
                faiss::idx_t* nns1 = nns + ii * k;
                float* dis1 = dis + ii * k;
                if (is_ivf) {
                    index_ivf->query(1, q + ii * d, k, dis1, nns1, z, &ivf_params);
                } else {
                    index_->search(1, q + ii * d, k, dis1, nns1, &hnsw_params);
                }
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

            if (is_ivf) {
                auto stats = index_ivf->get_ivf_stats();
                printf("centriod: %ld\n", index_ivf->nlist);
                printf("nprobe: %ld\n", ivf_params.nprobe);
                printf("quantization_time: %lf\n", stats.quantization_time - q_time);
                printf("compute_time: %lf\n", stats.compute_time - compute_time);
                printf("pruning_time: %lf\n", stats.pruning_time - pruning_time);
                printf("bound_time: %lf\n", stats.bound_time - bound_time);
                printf("bound_time_ip: %lf\n", stats.bound_time_ip - bound_time_ip);
                //            printf("t_sort: %lf\n", stats.t_sort - t_sort);
                printf("search_time_ivf: %lf\n", stats.search_time - search_time_2);
                printf("n_scanned: %d\n", stats.nlist - n_scanned);
                printf("c_scanned: %d\n", stats.c_list - c_scanned);
                printf("search_time2: %lf\n", search_time);
                printf("ndis: %ld\n", stats.ndis - ndis);
                printf("qps : %f\n", nq / search_time);

                all_ndis[r][i] = stats.ndis - ndis;
                all_positive_c[r][i] = stats.nlist - n_scanned;
                all_pruning_time_rate[r][i] = (stats.pruning_time - pruning_time) / (stats.search_time - search_time_2);

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
            } else {
                auto stats = faiss::hnsw_stats;
                printf("efSearch: %d\n", hnsw_params.efSearch);
                printf("search_time2: %lf\n", search_time);
                printf("ndis: %ld\n", (long)stats.ndis);
                printf("qps : %f\n", nq / search_time);

                all_ndis[r][i] = (float)stats.ndis;
                ndis = stats.ndis;
            }

            all_qps[r][i] = nq / search_time;
            all_recall[r][i] = recall;
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

    auto avg_qps = horizontal_avg(all_qps);
    auto avg_recall = horizontal_avg(all_recall);
    auto avg_ndis = horizontal_avg(all_ndis);
    auto avg_positive_c = horizontal_avg(all_positive_c);
    auto avg_pruning_time_rate = horizontal_avg(all_pruning_time_rate);
    if (is_ivf) {
        printf("\n\nnprobe\tQPS\trecall\n");
    } else {
        printf("\n\nefSearch\tQPS\trecall\n");
    }

    for (size_t i = 0; i < (size_t)length; ++i) {
        size_t cur = probe_values[i];
        float qps = avg_qps[i];
        float recall = avg_recall[i];
        float ndis_v = avg_ndis[i];
        printf("%zu\t%f\t%f\t%f\n", cur, qps, recall, ndis_v);
    }

    // 将结果写入文件
    if (is_ivf) {
        // 第一行：nprobe占比
        fprintf(result_file, "nprobe_percentage");
        for (size_t i = 0; i < (size_t)length; i++) {
            float percentage =
                    (static_cast<float>(all_nprobes[i]) / ncentroids) * 100.0f;
            fprintf(result_file, "\t%.2f%%", percentage);
        }
        fprintf(result_file, "\n");

        // 第二行：QPS
        fprintf(result_file, "qps");
        for (size_t i = 0; i < (size_t)length; i++) {
            fprintf(result_file, "\t%.2f", avg_qps[i]);
        }
        fprintf(result_file, "\n");

        // 第三行：召回率
        fprintf(result_file, "recall");
        for (size_t i = 0; i < (size_t)length; i++) {
            fprintf(result_file, "\t%.3f", avg_recall[i]);
        }
        fprintf(result_file, "\n");

        fprintf(result_file, "ndis");
        for (size_t i = 0; i < (size_t)length; i++) {
            fprintf(result_file, "\t%ld", (long)avg_ndis[i]);
        }
    } else {
        // HNSW: 第一行使用 efSearch
        fprintf(result_file, "efSearch");
        for (size_t i = 0; i < (size_t)length; i++) {
            fprintf(result_file, "\t%zu", probe_values[i]);
        }
        fprintf(result_file, "\n");

        // 第二行：QPS
        fprintf(result_file, "qps");
        for (size_t i = 0; i < (size_t)length; i++) {
            fprintf(result_file, "\t%.2f", avg_qps[i]);
        }
        fprintf(result_file, "\n");

        // 第三行：召回率
        fprintf(result_file, "recall");
        for (size_t i = 0; i < (size_t)length; i++) {
            fprintf(result_file, "\t%.3f", avg_recall[i]);
        }
        fprintf(result_file, "\n");

        fprintf(result_file, "ndis");
        for (size_t i = 0; i < (size_t)length; i++) {
            fprintf(result_file, "\t%ld", (long)avg_ndis[i]);
        }

        fprintf(result_file, "Positive clusters");
        for (size_t i = 0; i < length; i++) {
            fprintf(result_file, "\t%ld", (long) avg_positive_c[i]);
        }
        fprintf(result_file, "\n");

        fprintf(result_file, "pruning_time_rate");
        for (size_t i = 0; i < length; i++) {
            fprintf(result_file, "\t%.2f", avg_pruning_time_rate[i]);
        }
    }

    fclose(result_file);
    return 0;
}