#pragma once
#include "nn.hpp"
#include <array>
#include <memory>
#include <raylib.h>
#include <string>

struct Input {
  Vector2 mouse_position;
  bool mouse_down;
};

class GUI {
public:
  void run();
  GUI(std::unique_ptr<NN> &nn);

private:
  bool board_empty();
  std::unique_ptr<NN> &nn_;
  void handle_input();
  void input_board(Input &input);
  void draw_board();
  void draw_button(Color color, std::string label);
  void draw_output();
  void board_reset();
  Vector2 board_position;
  std::array<std::array<uint8_t, 280>, 280> board;
  const float board_size = 280.0;
  const int brush_radius = 5;
  const int screenWidth = 800;
  const int screenHeight = 450;
};
