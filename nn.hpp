#pragma once
#include <array>
#include <cstdint>
#include <vector>

struct Cache {
  std::vector<double> s_i;
  std::vector<double> s_j;
  std::vector<double> s_k;
  std::vector<double> e_i;
  std::vector<double> e_j;
  std::vector<double> e_k;
};

class NN {
public:
  int get_digit(const std::array<double, 784> &input);
  void load();
  void train();
  void write();
  void load_weights();
  NN();

private:
  void forward_pass(const std::array<double, 784> &input, Cache &cache);
  void back_propagation(const std::array<double, 784> &input, Cache &cache,
                        uint8_t desired_output);
  void update_weights();
  double activation_function(double argument);
  double derivative(double argument);
  double se(std::vector<double> output, uint8_t expected_output);
  void reset();
  double learning_rate = 1.0;
  int batch_size = 10;
  std::vector<int> training_order;

  struct WeightsStorage {
    int size = 345168;
    int v = 40035;
    int w = 2601;
    int u = 510;
  } weights_storage;

  struct Data {
    std::vector<std::array<double, 784>> train_input;
    std::vector<std::array<double, 784>> validation_input;
    std::vector<std::array<double, 784>> test_input;
    std::vector<uint8_t> train_output;
    std::vector<uint8_t> validation_output;
    std::vector<uint8_t> test_output;
  } data;

  struct Weights {
    std::array<std::array<double, 51>, 785> v;
    std::array<std::array<double, 51>, 51> w;
    std::array<std::array<double, 10>, 51> u;
  } weights;

  struct Derivatives {
    std::array<std::array<double, 51>, 785> dv;
    std::array<std::array<double, 51>, 51> dw;
    std::array<std::array<double, 10>, 51> du;
  } derivatives;
};
