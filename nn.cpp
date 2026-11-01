#include "nn.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

NN::NN() {
  data.train_input.resize(50000);
  data.train_output.resize(50000);
  data.validation_input.resize(10000);
  data.validation_output.resize(10000);
  data.test_input.resize(10000);
  data.test_output.resize(10000);
};

void NN::load() {
  auto read_file = [](const std::string &path, const int size,
                      const int magic_number) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
      std::cerr << "File not open" << std::endl;
      std::exit(1);
    }

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Size check
    if (file_size != size) {
      std::cerr << "Invalid file size" << std::endl;
      std::exit(2);
    };
    std::vector<unsigned char> buffer(size);
    file.read(reinterpret_cast<char *>(buffer.data()), size);
    file.close();
    // Magic number check
    if ((uint32_t(0) | buffer[3] | buffer[2] << 8 | buffer[1] << 16 |
         buffer[0] << 24) != magic_number) {
      std::cerr << "Wrong magic number" << std::endl;
      std::exit(3);
    };
    return buffer;
  };

  auto load_data = [](int idx, const std::vector<unsigned char> &origin_image,
                      const std::vector<unsigned char> &origin_label,
                      std::array<double, 784> &dest_input,
                      uint8_t &dest_output) {
    for (int j = 0; j < 784; ++j) {
      int pixel_idx = idx * 784 + j + 16;
      dest_input[j] = origin_image[pixel_idx] / 255.0;
    }
    dest_output = origin_label[idx + 8];
  };

  auto train_images = read_file(
      std::string(DATA_DIR) + "/train-images.idx3-ubyte", 47040016, 2051);
  auto train_labels = read_file(
      std::string(DATA_DIR) + "/train-labels.idx1-ubyte", 60008, 2049);
  auto test_images = read_file(
      std::string(DATA_DIR) + "/t10k-images.idx3-ubyte", 7840016, 2051);
  auto test_labels =
      read_file(std::string(DATA_DIR) + "/t10k-labels.idx1-ubyte", 10008, 2049);

  for (int i = 0; i < 60000; ++i) {
    if (i >= 50000) {
      load_data(i, train_images, train_labels, data.validation_input[i - 50000],
                data.validation_output[i - 50000]);
    } else {
      load_data(i, train_images, train_labels, data.train_input[i],
                data.train_output[i]);
    }
  }
  for (int i = 0; i < 10000; ++i) {
    load_data(i, test_images, test_labels, data.test_input[i],
              data.test_output[i]);
  }
  training_order.resize(50000);
  std::iota(training_order.begin(), training_order.end(), 0);
}

void NN::train() {
  load();
  load_weights();
  std::random_device rd;
  std::default_random_engine re(rd());
  Cache c;
  c.e_i.resize(51);
  c.e_j.resize(51);
  c.e_k.resize(10);
  c.s_i.resize(51);
  c.s_j.resize(51);
  c.s_k.resize(10);
  c.s_i[0] = 1;
  c.s_j[0] = 1;
  reset();
  for (int epoch = 0; epoch < 10; ++epoch) {
    std::shuffle(training_order.begin(), training_order.end(), re);
    double best_mse = 1.0;
    int train_count = 0;
    double mse_training = 0.0;
    while (train_count < 5000) {
      for (int i = 0; i < batch_size; ++i) {
        int idx = training_order[train_count * batch_size + i];
        forward_pass(data.train_input[idx], c);
        back_propagation(data.train_input[idx], c, data.train_output[idx]);
        mse_training += se(c.s_k, data.train_output[idx]);
      }
      update_weights();
      ++train_count;
    }

    int validation_count = 0;
    double mse_validation = 0.0;
    int correct_count = 0;
    while (validation_count < 1000) {
      for (int i = 0; i < batch_size; ++i) {
        int idx = validation_count * batch_size + i;
        forward_pass(data.validation_input[idx], c);
        mse_validation += se(c.s_k, data.validation_output[idx]);
        if (std::max_element(c.s_k.begin(), c.s_k.end()) - c.s_k.begin() ==
            data.validation_output[idx])
          ++correct_count;
      }
      ++validation_count;
    }

    mse_training /= 50000;
    mse_validation /= 10000;
    if (mse_validation < best_mse) {
      best_mse = mse_validation;
      write();
    }
    std::cout << "Epoch: " + std::to_string(epoch) << std::endl;
    std::cout << "MSE Training: " + std::to_string(mse_training) << std::endl;
    std::cout << "MSE Validation: " + std::to_string(mse_validation)
              << std::endl;
    std::cout << "Accuracy " + std::to_string(correct_count / 10000.0)
              << std::endl;
  }
};

void NN::load_weights() {
  std::ifstream file(std::string(WEIGHTS_DIR) + "/weights", std::ios::binary);
  if (file) {

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size == weights_storage.size) {
      file.read(reinterpret_cast<char *>(&weights), sizeof(weights));
      return;
    }
    std::cerr << "Weights not loaded" << std::endl;
  }
  std::uniform_real_distribution<double> unif(-0.1, 0.1);
  std::random_device rd;
  std::default_random_engine re(rd());
  auto init_weights = [&](auto &weights) {
    for (int i = 0; i < weights.size(); ++i) {
      for (int j = 0; j < weights[0].size(); ++j) {
        weights[i][j] = unif(re);
      }
    }
  };
  init_weights(weights.v);
  init_weights(weights.w);
  init_weights(weights.u);
};

void NN::forward_pass(const std::array<double, 784> &input, Cache &cache) {
  for (int i = 1; i < 51; ++i) {
    double a_i = weights.v[0][i];
    for (int h = 1; h < 785; ++h) {
      a_i += input[h - 1] * weights.v[h][i];
    }
    cache.s_i[i] = activation_function(a_i);
  }
  for (int j = 1; j < 51; ++j) {
    double a_j = 0;
    for (int i = 0; i < 51; ++i) {
      a_j += cache.s_i[i] * weights.w[i][j];
    }
    cache.s_j[j] = activation_function(a_j);
  }

  for (int k = 0; k < 10; ++k) {
    double a_k = 0;
    for (int j = 0; j < 51; ++j) {
      a_k += cache.s_j[j] * weights.u[j][k];
    }
    cache.s_k[k] = activation_function(a_k);
  }
};

void NN::back_propagation(const std::array<double, 784> &input, Cache &cache,
                          uint8_t desired_output) {
  for (int k = 0; k < 10; ++k) {
    double d_k = desired_output == k ? 1 : 0;
    cache.e_k[k] = (cache.s_k[k] - d_k) * derivative(cache.s_k[k]);
    for (int j = 0; j < 51; ++j) {
      derivatives.du[j][k] += cache.s_j[j] * cache.e_k[k];
    }
  }
  for (int j = 1; j < 51; ++j) {
    cache.e_j[j] = 0;
    for (int k = 0; k < 10; ++k) {
      cache.e_j[j] += weights.u[j][k] * cache.e_k[k];
    }
    cache.e_j[j] *= derivative(cache.s_j[j]);
    for (int i = 0; i < 51; ++i) {
      derivatives.dw[i][j] += cache.s_i[i] * cache.e_j[j];
    }
  }

  for (int i = 1; i < 51; ++i) {
    cache.e_i[i] = 0;
    for (int j = 1; j < 51; ++j) {
      cache.e_i[i] += weights.w[i][j] * cache.e_j[j];
    }
    cache.e_i[i] *= derivative(cache.s_i[i]);
    derivatives.dv[0][i] += cache.e_i[i];
    for (int h = 1; h < 785; ++h) {
      derivatives.dv[h][i] += input[h - 1] * cache.e_i[i];
    }
  }
};

void NN::update_weights() {
  for (int h = 0; h < 785; ++h) {
    for (int i = 1; i < 51; ++i) {
      weights.v[h][i] -= learning_rate * derivatives.dv[h][i] / batch_size;
    }
  }

  for (int i = 0; i < 51; ++i) {
    for (int j = 1; j < 51; ++j) {
      weights.w[i][j] -= learning_rate * derivatives.dw[i][j] / batch_size;
    }
  }

  for (int j = 0; j < 51; ++j) {
    for (int k = 0; k < 10; ++k) {
      weights.u[j][k] -= learning_rate * derivatives.du[j][k] / batch_size;
    }
  }
  reset();
};

double NN::derivative(double argument) { return argument * (1 - argument); };
double NN::activation_function(double argument) {
  return (1.0 / (1.0 + std::exp(-argument)));
};

void NN::reset() {
  for (auto &e : derivatives.du) {
    e.fill(0.0);
  }
  for (auto &e : derivatives.dw) {
    e.fill(0.0);
  }
  for (auto &e : derivatives.dv) {
    e.fill(0.0);
  }
};

double NN::se(std::vector<double> output, uint8_t expected_output) {
  double error = 0.0;
  int i = 0;
  for (auto &y : output) {
    double d = expected_output == i ? 1 : 0;
    error += (y - d) * (y - d);
    ++i;
  }
  return error / 2.0;
};

int NN::get_digit(const std::array<double, 784> &input) {
  Cache c;
  c.e_i.resize(51);
  c.e_j.resize(51);
  c.e_k.resize(10);
  c.s_i.resize(51);
  c.s_j.resize(51);
  c.s_k.resize(10);
  c.s_i[0] = 1;
  c.s_j[0] = 1;
  forward_pass(input, c);
  double best = 0;
  int number = 0;
  for (int i = 0; i < 10; i++) {
    if (c.s_k[i] > best) {
      best = c.s_k[i];
      number = i;
    }
  }
  return number;
};

void NN::write() {
  std::ofstream file(std::string(WEIGHTS_DIR) + "/weights", std::ios::binary);
  if (!file) {
    std::cerr << "Weights File not found" << std::endl;
    std::exit(4);
  }
  file.write(reinterpret_cast<const char *>(&weights), sizeof(weights));
};
