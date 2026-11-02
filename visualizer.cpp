#include "visualizer.hpp"
#include "nn.hpp"
#include "raylib.h"
#include <algorithm>
#include <memory>
#include <string>

GUI::GUI(std::unique_ptr<NN> &nn) : nn_(nn) {};
void GUI::run() {

  InitWindow(screenWidth, screenHeight, "digit-recognizer");

  SetTargetFPS(60); // Set our game to run at 60 frames-per-second
  board_position = {screenWidth / 2.0f - 14 * pixel_size, screenHeight / 4.0f};

  board_reset();
  // Main game loop
  while (!WindowShouldClose()) // Detect window close button or ESC key
  {
    handle_input();
    BeginDrawing();
    ClearBackground(RAYWHITE);
    draw_board();
    draw_output();
    EndDrawing();
  }
};

void GUI::draw_board() {
  DrawRectangle(board_position.x - 5, board_position.y - 5,
                28 * pixel_size + 10, 28 * pixel_size + 10, BLACK);
  DrawRectangle(board_position.x, board_position.y, 28 * pixel_size,
                28 * pixel_size, RAYWHITE);
  for (int i = 0; i < 784; ++i) {
    Color c = board[i] > 0.5 ? BLACK : RAYWHITE;
    int col = i % 28;
    int row = i / 28;
    int x = board_position.x + (col * pixel_size);
    int y = board_position.y + (row * pixel_size);
    DrawRectangle(x, y, pixel_size, pixel_size, c);
  }
};

void GUI::draw_output() {
  DrawText(std::to_string(nn_->get_digit(board)).c_str(), screenWidth / 2, 20,
           20, BLACK);
};

void GUI::input_board(Input &input) {
  if (!input.mouse_down ||
      !CheckCollisionPointRec(input.mouse_position,
                              {board_position.x, board_position.y,
                               28 * pixel_size, 28 * pixel_size}))
    return;
  double col = static_cast<int>((input.mouse_position.x - board_position.x) /
                                pixel_size);
  double row = static_cast<int>((input.mouse_position.y - board_position.y) /

                                pixel_size);

  int idx = int(row * 28 + col);
  board[idx] = 1.0;
  if (row >= 1 && row < 28 && col >= 1 && col < 28) {
    board[idx + 1] = std::clamp(board[idx + 1] + 0.5, 0.0, 1.0);
    board[idx - 1] = std::clamp(board[idx - 1] + 0.5, 0.0, 1.0);
    board[idx + 28] = std::clamp(board[idx + 28] + 0.5, 0.0, 1.0);
    board[idx - 28] = std::clamp(board[idx - 28] + 0.5, 0.0, 1.0);
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

void GUI::board_reset() { board.fill(0.0); };
