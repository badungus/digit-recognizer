#include "visualizer.hpp"
#include "nn.hpp"
#include "raylib.h"
#include "raymath.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

GUI::GUI(std::unique_ptr<NN> &nn) : nn_(nn) {};
void GUI::run() {

  InitWindow(screenWidth, screenHeight, "digit-recognizer");

  SetTargetFPS(60); // Set our game to run at 60 frames-per-second
  board_position = {screenWidth / 2.0f - board_size / 2, screenHeight / 4.0f};

  board_reset();
  // Main game loop
  while (!WindowShouldClose()) // Detect window close button or ESC key
  {
    handle_input();
    BeginDrawing();
    ClearBackground(RAYWHITE);
    draw_board();
    if (!board_empty())
      draw_output();
    EndDrawing();
  }
};

void GUI::draw_board() {
  DrawRectangle(board_position.x - 5, board_position.y - 5, board_size + 10,
                board_size + 10, BLACK);
  DrawRectangle(board_position.x, board_position.y, board_size, board_size,
                RAYWHITE);
  for (int row = 0; row < board_size; ++row) {
    for (int col = 0; col < board_size; ++col) {
      if (!board[col][row])
        continue;
      DrawPixel(board_position.x + col, board_position.y + row, BLACK);
    }
  }
};

void GUI::draw_output() {
  struct Bounds {
    int south = 0;
    int north = INT32_MAX;
    int east = 0;
    int west = INT32_MAX;
    int width;
    int height;
    int center_x;
    int center_y;
    int size;
  } bounds;
  for (int row = 0; row < board_size; ++row) {
    for (int col = 0; col < board_size; ++col) {
      if (!board[col][row])
        continue;

      bounds.south = std::max(bounds.south, row);
      bounds.north = std::min(bounds.north, row);
      bounds.east = std::max(bounds.east, col);
      bounds.west = std::min(bounds.west, col);
    }
  }

  bounds.width = bounds.east - bounds.west;
  bounds.height = bounds.south - bounds.north;
  bounds.center_x = bounds.west + bounds.width / 2;
  bounds.center_y = bounds.north + bounds.height / 2;

  double scale;
  struct Loop {
    int start_x;
    int start_y;
  } loop;

  if (bounds.width > bounds.height) {
    scale = bounds.width / 20.0;
    bounds.size = bounds.width;
    loop.start_x = bounds.west;
    loop.start_y = bounds.center_y - bounds.size / 2;
  } else {

    scale = bounds.height / 20.0;
    bounds.size = bounds.height;
    loop.start_x = bounds.center_x - bounds.size / 2;
    loop.start_y = bounds.north;
  }
  /*
    int sum_row = 0;
    int sum_col = 0;
    for (int row = bounds.north; row < bounds.south; ++row) {
      for (int col = bounds.west; col < bounds.east; ++col) {
        sum_col += board[col][row];
        sum_row += board[col][row];
      }
    }
  */

  std::array<double, 784> image;
  image.fill(0);
  for (int row = 0; row < 20; ++row) {

    for (int col = 0; col < 20; ++col) {

      double average_brightness = 0;
      for (int x = loop.start_x + col * scale;

           x < loop.start_x + (col + 1) * scale; ++x) {

        for (int y = loop.start_y + row * scale;
             y < loop.start_y + (row + 1) * scale; ++y) {
          if (x < 0 || x >= board_size || y < 0 || y >= board_size)
            continue;
          average_brightness += board[x][y];
        }
      }
      image[(row + 4) * 28 + (col + 4)] =
          average_brightness / (255.0 * scale * scale);
    }
  }
  /*
    for (int row = 0; row < 28; ++row) {
      for (int col = 0; col < 28; ++col) {
        if (!image[28 * row + col])
          continue;
        DrawPixel(board_position.x + 280 + col, board_position.y + 280 + row,
                  BLACK);
      }
    }
    */

  DrawText(std::to_string(nn_->get_digit(image)).c_str(), screenWidth / 2, 20,
           50, BLACK);
};

void GUI::input_board(Input &input) {
  if (!input.mouse_down ||
      !CheckCollisionPointRec(
          input.mouse_position,
          {board_position.x, board_position.y, board_size, board_size}))
    return;
  Vector2 mouse = Vector2Subtract(input.mouse_position, board_position);
  for (int row = mouse.y - brush_radius; row < mouse.y + brush_radius + 1;
       ++row) {
    if (row < 0 || row >= board_size)
      continue;

    for (int col = mouse.x - brush_radius; col < mouse.x + brush_radius + 1;
         ++col) {

      if (!((row - mouse.y) * (row - mouse.y) +
                (col - mouse.x) * (col - mouse.x) <
            brush_radius * brush_radius) ||
          col < 0 || col >= board_size)
        continue;
      board[col][row] = 255;
    }
  }
};

void GUI::handle_input() {
  Input input;
  input.mouse_position = GetMousePosition();
  input.mouse_down = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
  input_board(input);
  if (IsKeyPressed(KEY_R))
    board_reset();

  if (IsKeyPressed(KEY_T))
    nn_->train();
};

void GUI::draw_button(Color color, std::string label) {};

void GUI::board_reset() {
  for (int i = 0; i < board_size; ++i) {
    board[i].fill(0);
  }
};

bool GUI::board_empty() {
  int total = 0;
  for (int row = 0; row < board_size; ++row) {
    for (int col = 0; col < board_size; ++col) {
      total += board[col][row];
    }
  }
  return bool(!total);
};
