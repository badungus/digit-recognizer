#include "nn.hpp"
#include <iostream>
#include <memory>

int main() {
  std::cout << "Hello world!" << std::endl;
  auto nn = std::make_unique<NN>();
  nn->train();
  return 0;
};
