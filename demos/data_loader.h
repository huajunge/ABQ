//
// Created by root on 11/13/25.
//

#ifndef FAISS_DATA_LOADER_H
#define FAISS_DATA_LOADER_H
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include <omp.h>
#include <thread>
#include <algorithm>
#include <fstream>
#include <map>     // 添加map头文件
#include <utility> // 添加pair头文件
#include <H5Cpp.h>
#include <faiss/utils/utils.h>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#include <string>  // 确保包含string头文件

std::unique_ptr<float[]> fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    auto x = std::make_unique<float[]>(n * (d + 1)); // 使用智能指针

    size_t nr __attribute__((unused)) = fread(x.get(), sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    float* x_ptr = x.get();
    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x_ptr + i * d, x_ptr + 1 + i * (d + 1), d * sizeof(*x_ptr));

    fclose(f);
    return x;
}

void load_ivecs(
        const std::string& file_path,
        std::vector<std::vector<int>>& results) {
    std::ifstream file(file_path, std::ios::binary);
    int32_t k;
    file.read(reinterpret_cast<char*>(&k), sizeof(int32_t)); // 读取首个k值
    printf("Loading ivecs file: %s, k=%d\n", file_path.c_str(), k);
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t num_queries = file_size / (sizeof(int32_t) * (k + 1)); // 计算查询数

    results.resize(num_queries);
    file.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_queries; ++i) {
        file.seekg(sizeof(int32_t), std::ios::cur); // 跳过当前k值
        results[i].resize(k);
        file.read(
                reinterpret_cast<char*>(results[i].data()),
                k * sizeof(int32_t));
    }
}
struct DatasetPaths {
    size_t d;
    std::string databasePath;    // 数据库文件路径
    std::string trainPath;       // 训练数据路径
    std::string queriesPath;     // 查询数据路径
    std::string groundTruthPath; // 真实结果路径
    size_t n_learn = 0;
    int type = 0;
    DatasetPaths() = default;
    DatasetPaths(int type, size_t d, std::string databasePath, std::string trainPath, std::string queriesPath, std::string groundTruthPath)
            : type(type), d(d), databasePath(databasePath), trainPath(trainPath), queriesPath(queriesPath), groundTruthPath(groundTruthPath) {}
    DatasetPaths(int type, size_t d, std::string databasePath): type(type), d(d), databasePath(databasePath) {}
    DatasetPaths(int type, size_t d, std::string databasePath, size_t n_learn): type(type), d(d), databasePath(databasePath), n_learn(n_learn) {}
};

std::map<std::string, DatasetPaths> dataPathMap;

struct VectorDataset {
    size_t dimension;
    std::string distance_metric;
    std::string point_type;

    std::vector<float> train_vectors;     // 基向量集 (n_corpus x dim)
    std::vector<float> test_vectors;      // 查询向量 (n_test x dim)
    std::vector<size_t> ground_truth_ids;    // 最近邻ID (n_test x 100)
    std::vector<float> ground_truth_dists; // 最近邻距离 (n_test x 100)
    std::vector<float> avg_distances;     // 平均距离 (n_test)

    // OOD数据集专用
    std::vector<float> learn_vectors;
    std::vector<long> learn_neighbors;
};

// 读取HDF5数据集到内存
VectorDataset read_vector_dataset(const std::string& file_path, DatasetPaths& datasetPaths) {
    VectorDataset dataset;
    try {
        // 使用RAII管理HDF5文件资源
        H5::H5File file(file_path, H5F_ACC_RDONLY);
        printf("Reading vector dataset from: %s\n", file_path.c_str());
        // 安全读取属性函数
        auto safe_read_attribute = [&](const std::string& name) -> std::string {
            if (!file.attrExists(name)) {
                printf("Missing required attribute: %s\n", name.c_str());
                throw std::runtime_error("Missing required attribute: " + name);
            }

            H5::Attribute attr = file.openAttribute(name);
            H5::DataType dtype = attr.getDataType();
            H5::StrType str_type = attr.getStrType();

            // 获取属性大小
            size_t attr_size = str_type.getSize();
            if (attr_size == 0) {
                return "";
            }

            // 分配缓冲区并读取
            std::vector<char> buffer(attr_size + 1, '\0');
            attr.read(str_type, buffer.data());
            return std::string(buffer.data());
        };
        // 读取数值属性
        auto read_numeric_attribute = [&](const std::string& name) {
            if (!file.attrExists(name)) {
                printf("Missing required attribute: %s\n", name.c_str());
                throw std::runtime_error("Missing required attribute: " + name);
            }

            H5::Attribute attr = file.openAttribute(name);
            H5::DataType dtype = attr.getDataType();

            if (dtype.getClass() == H5T_INTEGER) {
                uint32_t value;
                attr.read(dtype, &value);
                return value;
            }
            throw std::runtime_error(
                    "Unsupported numeric attribute type: " + name);
        };
        // 读取属性值
        dataset.distance_metric = safe_read_attribute("distance");
        printf("distance_metric: %s\n", dataset.distance_metric.c_str());

        dataset.dimension = read_numeric_attribute("dimension");
        printf("dimension: %d\n", dataset.dimension);

        dataset.point_type = safe_read_attribute("point_type");
        printf("point_type: %s\n", dataset.point_type.c_str());

        // 读取数据集
        auto read_dataset = [&](const std::string& name,
                                auto& container,
                                H5::PredType data_type) {
            if (!file.exists(name)) {
                throw std::runtime_error("Missing required dataset: " + name);
            }

            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            printf("获取维度信息 vector dataset from: %s\n", file_path.c_str());

            // 获取维度信息
            const int ndims = space.getSimpleExtentNdims();
            std::vector<hsize_t> dims(ndims);
            space.getSimpleExtentDims(dims.data());

            // 计算元素总数
            size_t total_elements = 1;
            for (auto dim : dims) {
                total_elements *= dim;
            }

            // 调整容器大小并读取数据
            container.resize(total_elements);
            ds.read(container.data(), data_type);
        };

        // 读取必须的数据集
        read_dataset(
                "train", dataset.train_vectors, H5::PredType::NATIVE_FLOAT);
        read_dataset("test", dataset.test_vectors, H5::PredType::NATIVE_FLOAT);
        read_dataset(
                "neighbors", dataset.ground_truth_ids, H5::PredType::STD_I64LE);

        // 读取可选的OOD数据集
        auto read_optional_dataset = [&](const std::string& name,
                                         auto& container,
                                         H5::PredType data_type) {
            bool learn_data_from_train_vectors = true;
            if (file.exists(name)) {
                H5::DataSet ds = file.openDataSet(name);
                H5::DataSpace space = ds.getSpace();

                const int ndims = space.getSimpleExtentNdims();
                std::vector<hsize_t> dims(ndims);
                space.getSimpleExtentDims(dims.data());

                size_t total_elements = 1;
                for (auto dim : dims) {
                    total_elements *= dim;
                }

//                container.resize(total_elements);
//                ds.read(container.data(), data_type);
//                learn_data_from_train_vectors = false;
                if (total_elements < 100000 * datasetPaths.d) {
                    container.resize(total_elements);
                    ds.read(container.data(), data_type);
                    learn_data_from_train_vectors = false;
                }
            }

            if (learn_data_from_train_vectors) {
                std::cout << "Optional dataset '" << name << "' not found." << std::endl;
                // 当learn数据集不存在时，从train_vectors中选取10万条向量
                if (name == "learn") {
                    size_t total_train_vectors = dataset.train_vectors.size() / dataset.dimension;
                    size_t train_size = datasetPaths.n_learn;
//                    if (total_train_vectors < 1000000) {
//                        train_size = 50000;
//                    }

                    size_t vectors_to_copy = std::min(total_train_vectors, train_size);

                    if (vectors_to_copy > 0) {
                        size_t elements_to_copy = vectors_to_copy * dataset.dimension;
                        container.resize(elements_to_copy);
//                        std::copy(
//                                dataset.train_vectors.begin(),
//                                dataset.train_vectors.begin() + elements_to_copy,
//                                container.begin()
//                        );

                        int stride = total_train_vectors / train_size;
                        int count = 0;
                        for (int i = 0; i < total_train_vectors && count < train_size; i += stride) {
//                            for (int j = 0; j < dataset.dimension; j++) {
//                                container.push_back(dataset.train_vectors[i * dataset.dimension + j]);
//                            }
                            std::copy(dataset.train_vectors.begin() + i * dataset.dimension,
                                      dataset.train_vectors.begin() + (i + 1) * dataset.dimension,
                                      container.begin() + count * dataset.dimension);
                            count++;
                        }

                        std::cout << "Populated learn_vectors from train_vectors (first "
                                  << vectors_to_copy << " vectors)." << std::endl;
                    } else {
                        std::cout << "Warning: train_vectors is empty, cannot populate learn_vectors." << std::endl;
                    }
                }
            }
        };
        read_optional_dataset("learn", dataset.learn_vectors, H5::PredType::NATIVE_FLOAT);
//        read_optional_dataset(
//                "learn_neighbors",
//                dataset.learn_neighbors,
//                H5::PredType::STD_I64LE);

        // 显式关闭文件并释放资源
        file.close();
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 error: " << e.getDetailMsg() << std::endl;
        throw std::runtime_error(
                "Failed to read vector dataset from: " + file_path);
    } catch (const std::exception& e) {
        std::cerr << "Error reading dataset: " << e.what() << std::endl;
        throw;
    }

    return dataset;
}

void init_data_path_map() {
    // 添加数据集路径
    std::string base_path = "/data/Projects/data/vectors/";
    DatasetPaths siftPaths;
    siftPaths.d = 128;
    siftPaths.databasePath = base_path + "sift/sift_base.fvecs";
    siftPaths.trainPath = base_path + "sift/sift_learn.fvecs";
    siftPaths.queriesPath = base_path + "sift/sift_query.fvecs";
    siftPaths.groundTruthPath = base_path + "sift/sift_groundtruth.ivecs";
    dataPathMap["SIFT1M"] = siftPaths;

    DatasetPaths imagenetPaths(1,512, base_path + "imagenet-clip-512-normalized/imagenet-clip-512-normalized.hdf5", 50000);
    dataPathMap["ImageNet"] = imagenetPaths;

    DatasetPaths agnewsPaths(1, 1024, base_path + "agnews-mxbai-1024-euclidean/agnews-mxbai-1024-euclidean.hdf5",50000);
    dataPathMap["AgNews"] = agnewsPaths;

    DatasetPaths laionPaths(1, 512, base_path + "laion/laion-clip-512-normalized.hdf5", 200000);
    dataPathMap["Laion"] = laionPaths;

    DatasetPaths gooap(1, 768, base_path + "gooaq/gooaq-distilroberta-768-normalized.hdf5", 100000);
    dataPathMap["GooAQ"] = gooap;

    DatasetPaths gist(0, 960, base_path + "glove/",100000);
    gist.databasePath = base_path + "GIST/gist/gist_base.fvecs";
    gist.trainPath = base_path + "GIST/gist/gist_learn.fvecs";
    gist.queriesPath = base_path + "GIST/gist/gist_query.fvecs";
    gist.groundTruthPath = base_path + "GIST/gist/gist_groundtruth.ivecs";
    dataPathMap["Gist"] = gist;


    DatasetPaths glove(0, 300, base_path + "glove/",100000);
    glove.databasePath = base_path + "glove/glove_base.fvecs";
    glove.trainPath = base_path + "glove/glove_learn.fvecs";
    glove.queriesPath = base_path + "glove/glove_query.fvecs";
    glove.groundTruthPath = base_path + "glove/glove_groundtruth.ivecs";
    dataPathMap["Glove"] = glove;


    DatasetPaths glove120(0, 300, base_path + "glove120",100000);
    glove120.databasePath = base_path + "glove120/glove_base.fvecs";
    glove120.trainPath = base_path + "glove120/glove_learn.fvecs";
    glove120.queriesPath = base_path + "glove120/glove_query.fvecs";
    glove120.groundTruthPath = base_path + "glove120/glove_groundtruth.ivecs";
    dataPathMap["Glove120"] = glove120;
}


void load_ivecs_from_h5(std::string data_name, std::unique_ptr<float[]>& database, std::unique_ptr<float[]>& train, std::unique_ptr<float[]>& queries, std::vector<std::vector<int>>& ground_truth,
                        size_t& nb, size_t& nt, size_t& nq, size_t& d) {
    auto it = dataPathMap.find(data_name);
    if (it == dataPathMap.end()) {
        throw std::runtime_error("Dataset not found: " + data_name);
    }

    DatasetPaths datasetPaths = it->second;
    d = datasetPaths.d;
    switch (datasetPaths.type) {
        case 0:
            database = fvecs_read(datasetPaths.databasePath.c_str(), &d, &nb);
            train = fvecs_read(datasetPaths.trainPath.c_str(), &d, &nt);
            queries = fvecs_read(datasetPaths.queriesPath.c_str(), &d, &nq);
            load_ivecs(datasetPaths.groundTruthPath.c_str(), ground_truth);
            break;
        case 1:
            // 复制数据到新分配的内存
            VectorDataset dataset =
                    read_vector_dataset(datasetPaths.databasePath, datasetPaths);
            size_t database_size = dataset.train_vectors.size();
            database = std::make_unique<float[]>(database_size);
            std::copy(
                    dataset.train_vectors.begin(),
                    dataset.train_vectors.end(),
                    database.get());

            size_t train_size = dataset.learn_vectors.size();
            train = std::make_unique<float[]>(train_size);
            std::copy(
                    dataset.learn_vectors.begin(),
                    dataset.learn_vectors.end(),
                    train.get());

            size_t queries_size = dataset.test_vectors.size();
            queries = std::make_unique<float[]>(queries_size);
            std::copy(
                    dataset.test_vectors.begin(),
                    dataset.test_vectors.end(),
                    queries.get());

            // 重新设置ground_truth大小
            nb = database_size / d;
            nt = train_size / d;
            nq = queries_size / d;

            // 按每100个元素分组
            ground_truth.resize(nq);
            const size_t k = 100;  // 每组100个元素
            for (size_t i = 0; i < nq; i++) {
                ground_truth[i].resize(k);
                std::copy(
                        dataset.ground_truth_ids.begin() + i * k,
                        dataset.ground_truth_ids.begin() + (i + 1) * k,
                        ground_truth[i].begin()
                );
            }
            break;
    }
}

#endif // FAISS_DATA_LOADER_H
