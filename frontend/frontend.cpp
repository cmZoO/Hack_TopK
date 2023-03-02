#include <iostream>
#include <string>
#include "Math/math-functions.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <thread>
#include <iomanip>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

using namespace std;

using namespace sci;
using namespace std;
using namespace Eigen;

int party, port = 32000;
string address = "127.0.0.1";

//int dim = 1ULL << 16;
int dim = 1000000;
int topk = 10;
int bw_x = 24;

uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));

sci::NetIO* ioArr[1];
sci::OTPack<sci::NetIO>* otpackArr[1];
MathFunctions* math;

bool compare(int32_t a, int32_t b) {
    return a > b;
}


void init(uint64_t* a_share, uint64_t* b_share, uint64_t block_cnt, uint64_t block_a_len, uint64_t block_b_len) {
    uint64_t* sub_share = new uint64_t[block_a_len * block_cnt];
    for (int i = 0; i < block_cnt; i++) {
        elemWiseSub(block_a_len, a_share + i * block_a_len, b_share + i * block_b_len + block_b_len - block_a_len, sub_share + i * block_a_len);
    }
    math->ReLU(block_a_len * block_cnt, sub_share, sub_share, bw_x);
    uint64_t a, b;
    for (int i = 0; i < block_cnt; i++) {
        for (int j = 0; j < block_a_len; j++) {
            a = a_share[i * block_a_len + j];
            b = b_share[i * block_b_len + block_b_len - block_a_len + j];
            a_share[i * block_a_len + j] = sub_share[i * block_a_len + j] + b;
            b_share[i * block_b_len + block_b_len - block_a_len + j] = a - sub_share[i * block_a_len + j];
        }
    }
    delete[] sub_share;
}

void cal_block_topk(uint64_t* a_share, uint64_t* b_share, uint64_t block_cnt, uint64_t block_a_len, uint64_t block_b_len, uint64_t* topk_share) {
    uint64_t* a_tmp = new uint64_t[block_cnt * block_a_len];
    uint64_t* b_tmp = new uint64_t[block_cnt * block_a_len];
    uint64_t* ab_tmp = new uint64_t[2 * block_cnt];
    uint64_t a, b;
    uint32_t block_a_start = 0, block_b_start = 0;
    for (uint64_t rank = 0; rank < topk; rank++) {
        int a_len = block_a_len - block_a_start;
        int b_len = block_b_len - block_b_start;
        int b_start_flag = b_len - a_len;
        while (a_len > 1) {
            int half_a_len = a_len / 2;
            int block_a_mid = block_a_start + half_a_len + a_len % 2;
            int block_b_mid = block_b_start + b_start_flag + half_a_len + a_len % 2;
            for (int i = 0; i < block_cnt; i++) {
                int block_ai_index = i * block_a_len;
                int block_bi_index = i * block_b_len;
                elemWiseSub(half_a_len, a_share + block_ai_index + block_a_start, a_share + block_ai_index + block_a_mid, a_tmp + i * half_a_len);
                elemWiseSub(half_a_len, b_share + block_bi_index + block_b_start + b_start_flag, b_share + block_bi_index + block_b_mid, b_tmp + i * half_a_len);
            }
            math->ReLU(half_a_len * block_cnt, a_tmp, a_tmp, b_tmp, b_tmp, bw_x);
            for (int i = 0; i < block_cnt; i++) {
                int block_ai_index = i * block_a_len;
                int block_bi_index = i * block_b_len;
                for (int j = 0; j < half_a_len; j++) {
                    a = a_share[block_ai_index + block_a_start + j];
                    b = a_share[block_ai_index + block_a_mid + j];
                    a_share[block_ai_index + block_a_start + j] = a_tmp[i * half_a_len + j] + b;
                    a_share[block_ai_index + block_a_mid + j] = a - a_tmp[i * half_a_len + j];

                    a = b_share[block_bi_index + block_b_start + b_start_flag + j];
                    b = b_share[block_bi_index + block_b_mid + j];
                    b_share[block_bi_index + block_b_start + b_start_flag + j] = b_tmp[i * half_a_len + j] + b;
                    b_share[block_bi_index + block_b_mid + j] = a - b_tmp[i * half_a_len + j];
                }
            }
            a_len = half_a_len + a_len % 2;
        }
        if (b_start_flag) {
            if (b_len > 1) {
                for (int i = 0; i < block_cnt; i++) {
                    int block_ai_index = i * block_a_len;
                    int block_bi_index = i * block_b_len;
                    ab_tmp[2 * i] = b_share[block_bi_index + block_b_start] - b_share[block_bi_index + block_b_start + 1];
                    ab_tmp[2 * i + 1] = b_share[block_bi_index + block_b_start] - a_share[block_ai_index + block_a_start];
                }
                math->ReLU(2 * block_cnt, ab_tmp, ab_tmp, bw_x);
                for (int i = 0; i < block_cnt; i++) {
                    int block_ai_index = i * block_a_len;
                    int block_bi_index = i * block_b_len;

                    a = b_share[block_bi_index + block_b_start];
                    b = b_share[block_bi_index + block_b_start + 1];
                    b_share[block_bi_index + block_b_start] = ab_tmp[i * 2] + b;
                    b_share[block_bi_index + block_b_start + 1] = a - ab_tmp[i * 2];

                    a = b_share[block_bi_index + block_b_start];
                    b = a_share[block_ai_index + block_a_start];
                    b_share[block_bi_index + block_b_start] = ab_tmp[2 * i + 1] + b;
                    a_share[block_ai_index + block_a_start] = a - ab_tmp[2 * i + 1];
                }
            }
            for (int i = 0; i < block_cnt; i++) {
                topk_share[i * topk + rank] = b_share[i * block_b_len + block_b_start];
            }
            block_b_start++;
        }
        else {
            for (int i = 0; i < block_cnt; i++) {
                topk_share[i * topk + rank] = a_share[i * block_a_len + block_a_start];
            }
            block_a_start++;
        }
    }
    delete[] a_tmp;
    delete[] b_tmp;
    delete[] ab_tmp;
}


void cal_app_topk(uint64_t* x, uint64_t block_cnt, uint64_t* topk_res_share, int dim) {
    uint64_t block_size = dim / block_cnt;
    if (block_size < topk) {
        block_size = topk;
        block_cnt = dim / block_size;
    }
    cout << log2(block_size) << " " << log2(dim) << endl;
    uint32_t block_a_len = block_size / 2, block_b_len = block_size / 2 + block_size % 2;
    uint64_t* a_share = new uint64_t[block_a_len * block_cnt];
    uint64_t* b_share = new uint64_t[block_b_len * block_cnt];
    uint64_t* block_topk_share = new uint64_t[block_cnt * topk];
    uint64_t* topk_share = new uint64_t[dim % block_size + block_cnt * topk];
    //uint64_t* a_share = new uint64_t[dim];
    //uint64_t* b_share = new uint64_t[dim];
    //uint64_t* block_topk_share = new uint64_t[dim];
    //uint64_t* topk_share = new uint64_t[dim];

    // 初始化变成a全部大于b的情况
    memcpy(a_share, x, block_a_len * block_cnt * sizeof(uint64_t));
    memcpy(b_share, x + block_a_len * block_cnt, block_b_len * block_cnt * sizeof(uint64_t));
    init(a_share, b_share, block_cnt, block_a_len, block_b_len);
    // 计算每一个block里面的top k，并且放到对应的topk_share里面
    int ogtopk = topk;
    topk = 1;
    cal_block_topk(a_share, b_share, block_cnt, block_a_len, block_b_len, block_topk_share);
    // 将每个block的topk与之前block取余出来的数据放到一起
    memcpy(topk_share, x + block_cnt * block_size, (dim % block_size) * sizeof(uint64_t));
    memcpy(topk_share + (dim % block_size), block_topk_share, block_cnt * topk * sizeof(uint64_t));
    // 最后重排并获取结果
    block_a_len = (dim % block_size + block_cnt * topk) / 2;
    block_b_len = block_a_len + (dim % block_size + block_cnt * topk) % 2;
    block_cnt = 1;
    cout << (block_a_len + block_b_len) << " " << log2(block_a_len + block_b_len) << endl;
    memcpy(a_share, topk_share, block_a_len * sizeof(uint64_t));
    memcpy(b_share, topk_share + block_a_len, block_b_len * sizeof(uint64_t));
    topk = ogtopk;
    init(a_share, b_share, block_cnt, block_a_len, block_b_len);
    cal_block_topk(a_share, b_share, block_cnt, block_a_len, block_b_len, block_topk_share);
    // 构造结果返回
    memcpy(topk_res_share, block_topk_share, topk * sizeof(uint64_t));
}

bool topk2(int argc, char** argv) {
    assert(topk <= dim);
    /************* Argument Parsing  ************/
    /********************************************/
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("N", dim, "Number of ReLU operations");
    amap.arg("ip", address, "IP Address of server (ALICE)");

    amap.parse(argc, argv);

    /********** Setup IO and Base OTs ***********/
    /********************************************/
    ioArr[0] = new sci::NetIO(party == 1 ? nullptr : address.c_str(), port);
    otpackArr[0] = new OTPack<sci::NetIO>(ioArr[0], party);
    std::cout << "All Base OTs Done" << std::endl;

    /************ Generate Test Data ************/
    /********************************************/
    PRG128 prg;

    uint64_t* x = new uint64_t[dim];
    uint64_t* x_other = new uint64_t[dim];

    int32_t magnitude_bound = 1ULL << (bw_x - 2);
    prg.random_data(x, dim * sizeof(uint64_t));
    prg.random_data(x_other, dim * sizeof(uint64_t));
    if (party == sci::ALICE) {
        for (int i = 0; i < dim; i++) {
            x[i] = (int32_t)x[i] % magnitude_bound;
        }
        if (dim <= 16) {
            cout << "生成数组:";
            for (int i = 0; i < dim; i++) {
                cout << (int32_t)x[i] << " ";
            }
        }

        for (int i = 0; i < dim; i++) {
            x[i] = uint64_t(x[i] - x_other[i]);
        }
        ioArr[0]->send_data(x_other, dim * sizeof(uint64_t));
    }
    else {
        ioArr[0]->recv_data(x, dim * sizeof(uint64_t));
    }
    cout << endl;

    /************** Calculate Topk ****************/
       /********************************************/
    std::cout << "Calculate Topk" << std::endl;
    uint64_t thread_comm[1];
    thread_comm[0] = ioArr[0]->counter;
    math = new MathFunctions(party, ioArr[0], otpackArr[0]);
    uint64_t* topk_res_share = new uint64_t[topk];
    auto start = high_resolution_clock::now();

    cal_app_topk(x, topk / 0.01, topk_res_share, dim);

    long long t = std::chrono::duration_cast<std::chrono::microseconds>(high_resolution_clock::now() - start).count();

    thread_comm[0] = ioArr[0]->counter - thread_comm[0];

    /************** Verification ****************/
    /********************************************/
    if (party == sci::ALICE) {
        uint64_t* topk_value_share_other = new uint64_t[topk];
        ioArr[0]->recv_data(topk_value_share_other, topk * sizeof(uint64_t));
        ioArr[0]->recv_data(x_other, dim * sizeof(uint64_t));
        int32_t* signed_x = new int32_t[dim];
        for (int i = 0; i < dim; i++) {
            x[i] += x_other[i];
            signed_x[i] = signed_val(x[i], bw_x);
        }
        if (dim <= 16) {
            cout << "复原数组:";
            for (int i = 0; i < dim; i++) {
                cout << signed_x[i] << " ";
            }
            cout << endl;
        }
        // 排序原数组
        sort(signed_x, signed_x + dim, compare);
        if (dim <= 16) {
            cout << "排序数组:";
            for (int i = 0; i < dim; i++) {
                cout << signed_x[i] << " ";
            }
            cout << endl;
        }
        cout << "Topk数组:";
        for (int i = 0; i < topk; i++) {
            int64_t argmax_value = signed_val(topk_res_share[i] + topk_value_share_other[i], bw_x);
            cout << argmax_value << " ";
            //assert(argmax_value == signed_x[i]);
            if (argmax_value != signed_x[i]) {
                return false;
            }
        }
        cout << endl;

        cout << "Topk Tests Passed" << endl;

        delete[] topk_value_share_other;
        delete[] signed_x;
    }
    else { // party == BOB
        ioArr[0]->send_data(topk_res_share, topk * sizeof(uint64_t));
        ioArr[0]->send_data(x, dim * sizeof(uint64_t));
    }

    /**** Process & Write Benchmarking Data *****/
    /********************************************/
    cout << "Number of Topk/s:\t" << (double(topk) / t) * 1e6 << std::endl;
    cout << "Topk Time\t" << t / (1000.0) << " ms" << endl;
    cout << "Topk Bytes Sent\t" << thread_comm[0] << " bytes" << endl;

    /******************* Cleanup ****************/
    /********************************************/
    delete[] x;
    delete[] x_other;
    delete[] topk_res_share;
    delete ioArr[0];
    delete otpackArr[0];
    delete math;

    return true;
}


void topk3(int argc, char** argv) {
    assert(topk <= dim);
    /************* Argument Parsing  ************/
    /********************************************/
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("N", dim, "Number of ReLU operations");
    amap.arg("ip", address, "IP Address of server (ALICE)");

    amap.parse(argc, argv);

    /********** Setup IO and Base OTs ***********/
    /********************************************/
    ioArr[0] = new sci::NetIO(party == 1 ? nullptr : address.c_str(), port);
    otpackArr[0] = new OTPack<sci::NetIO>(ioArr[0], party);
    std::cout << "All Base OTs Done" << std::endl;

    /************ Generate Test Data ************/
    /********************************************/
    PRG128 prg;

    uint64_t* x = new uint64_t[dim];
    uint64_t* x_other = new uint64_t[dim];

    int32_t magnitude_bound = 1ULL << (bw_x - 2);
    prg.random_data(x, dim * sizeof(uint64_t));
    prg.random_data(x_other, dim * sizeof(uint64_t));
    if (party == sci::ALICE) {
        for (int i = 0; i < dim; i++) {
            x[i] = (int32_t)x[i] % magnitude_bound;
        }
        if (dim <= 16) {
            cout << "生成数组:";
            for (int i = 0; i < dim; i++) {
                cout << (int32_t)x[i] << " ";
            }
        }

        for (int i = 0; i < dim; i++) {
            x[i] = uint64_t(x[i] - x_other[i]);
        }
        ioArr[0]->send_data(x_other, dim * sizeof(uint64_t));
    }
    else {
        ioArr[0]->recv_data(x, dim * sizeof(uint64_t));
    }
    cout << endl;

    /************** Calculate Topk ****************/
       /********************************************/
    std::cout << "Calculate Topk" << std::endl;
    uint64_t thread_comm[1];
    thread_comm[0] = ioArr[0]->counter;
    auto start = high_resolution_clock::now();
    math = new MathFunctions(party, ioArr[0], otpackArr[0]);
    uint32_t b_len = dim / 2, s_len = dim / 2 + dim % 2;
    uint64_t* b_share = new uint64_t[b_len], * b_tmp = new uint64_t[b_len];
    uint64_t* s_share = new uint64_t[s_len], * s_tmp = new uint64_t[s_len];;
    uint64_t* topk_value_share = new uint64_t[dim];
    memcpy(b_share, x, b_len * sizeof(uint64_t));
    memcpy(s_share, x + b_len, s_len * sizeof(uint64_t));
    {
        int len = b_len;
        uint64_t* tmp_share = new uint64_t[b_len];
        for (int i = 0; i < len; i++) {
            tmp_share[i] = uint64_t(b_share[i] - s_share[i + dim % 2]);
        }
        math->ReLU(len, tmp_share, tmp_share, bw_x);
        for (int i = 0; i < len; i++) {
            uint64_t a = b_share[i];
            uint64_t b = s_share[i + dim % 2];
            b_share[i] = uint64_t(tmp_share[i] + b);
            s_share[i + dim % 2] = uint64_t(a - tmp_share[i]);
        }
    }
    uint32_t b_left_index = 0, s_left_index = 0;
    uint64_t ab_tmp[2];
    for (uint32_t rank = 0; rank < topk; rank++) {
        int len = b_len - b_left_index;
        int s_left_flag = s_len - s_left_index - len;
        while (len > 1) {
            int half_len = len / 2;
            int b_right_index = b_left_index + half_len + len % 2;
            int s_right_index = s_left_index + s_left_flag + half_len + len % 2;
            for (int i = 0; i < half_len; i++) {
                b_tmp[i] = uint64_t(b_share[b_left_index + i] - b_share[b_right_index + i]);
                s_tmp[i] = uint64_t(s_share[s_left_index + s_left_flag + i] - s_share[s_right_index + i]);
            }
            math->ReLU(half_len, b_tmp, b_tmp, s_tmp, s_tmp, bw_x);
            for (int i = 0; i < half_len; i++) {
                uint64_t a = b_share[b_left_index + i];
                uint64_t b = b_share[b_right_index + i];
                b_share[b_left_index + i] = uint64_t(b_tmp[i] + b);
                b_share[b_right_index + i] = uint64_t(a - b_tmp[i]);

                a = s_share[s_left_index + s_left_flag + i];
                b = s_share[s_right_index + i];
                s_share[s_left_index + s_left_flag + i] = uint64_t(s_tmp[i] + b);
                s_share[s_right_index + i] = uint64_t(a - s_tmp[i]);
            }
            len = half_len + len % 2;
        }
        if (s_left_flag) {
            if (s_len - s_left_index > 1) {
                ab_tmp[0] = uint64_t(s_share[s_left_index] - s_share[s_left_index + 1]);
                ab_tmp[1] = uint64_t(s_share[s_left_index] - b_share[b_left_index]);
                math->ReLU(2, ab_tmp, ab_tmp, bw_x);

                uint64_t a = s_share[s_left_index];
                uint64_t b = s_share[s_left_index + 1];
                s_share[s_left_index] = uint64_t(ab_tmp[0] + b);
                s_share[s_left_index + 1] = uint64_t(a - ab_tmp[0]);

                a = s_share[s_left_index];
                b = b_share[b_left_index];
                s_share[s_left_index] = uint64_t(ab_tmp[1] + b);
                b_share[b_left_index] = uint64_t(a - ab_tmp[1]);
            }
            topk_value_share[rank] = s_share[s_left_index++];
        }
        else {
            topk_value_share[rank] = b_share[b_left_index++];
        }
    }
    long long t = std::chrono::duration_cast<std::chrono::microseconds>(high_resolution_clock::now() - start).count();
    
    thread_comm[0] = ioArr[0]->counter - thread_comm[0];

    /************** Verification ****************/
    /********************************************/
    if (party == sci::ALICE) {
        uint64_t* topk_value_share_other = new uint64_t[topk];
        ioArr[0]->recv_data(topk_value_share_other, topk * sizeof(uint64_t));
        ioArr[0]->recv_data(x_other, dim * sizeof(uint64_t));
        int32_t* signed_x = new int32_t[dim];
        for (int i = 0; i < dim; i++) {
            x[i] += x_other[i];
            signed_x[i] = signed_val(x[i], bw_x);
        }
        if (dim <= 16) {
            cout << "复原数组:";
            for (int i = 0; i < dim; i++) {
                cout << signed_x[i] << " ";
            }
            cout << endl;
        }
        // 排序原数组
        sort(signed_x, signed_x + dim, compare);
        if (dim <= 16) {
            cout << "排序数组:";
            for (int i = 0; i < dim; i++) {
                cout << signed_x[i] << " ";
            }
            cout << endl;
        }
        cout << "Topk数组:";
        for (int i = 0; i < topk; i++) {
            int64_t argmax_value = signed_val(topk_value_share[i] + topk_value_share_other[i], bw_x);
            cout << argmax_value << " ";
            assert(argmax_value == signed_x[i]);
        }
        cout << endl;

        cout << "Topk Tests Passed" << endl;

        delete[] topk_value_share_other;
        delete[] signed_x;
    }
    else { // party == BOB
        ioArr[0]->send_data(topk_value_share, topk * sizeof(uint64_t));
        ioArr[0]->send_data(x, dim * sizeof(uint64_t));
    }

    /**** Process & Write Benchmarking Data *****/
    /********************************************/
    cout << "Number of Topk/s:\t" << (double(topk) / t) * 1e6 << std::endl;
    cout << "Topk Time\t" << t / (1000.0) << " ms" << endl;
    cout << "Topk Bytes Sent\t" << thread_comm[0] << " bytes" << endl;

    /******************* Cleanup ****************/
    /********************************************/
    delete[] x;
    delete[] x_other;
    delete[] b_share;
    delete[] topk_value_share;
    delete[] s_share;
    delete ioArr[0];
    delete otpackArr[0];
    delete math;
}

int main(int argc, char** argv) {

    topk3(argc, argv);
    //topk2(argc, argv);

	return 0;
}
