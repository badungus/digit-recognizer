#include "nn.hpp"
#include "visualizer.hpp"
#include <iostream>
#include <memory>

int main() {
  std::cout << "Hello world!" << std::endl;
  auto nn = std::make_unique<NN>();
  nn->load_weights();
  GUI gui(nn);
  gui.run();
  return 0;
};
