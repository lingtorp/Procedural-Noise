#include <iostream>
#include "noise.hpp"
#include <SDL2/SDL.h>
#include <future>

/** Zooms into details of the noise */
const int DIVISOR = 64;
/** */
const int SEED = 1;
/** */
const double TIME_STEP = 0.01;

struct Region {
  const size_t nx, ny;
  size_t x0, x1;
  size_t y0, y1;
};

/// Semaphore
struct Semaphore {
private:
  size_t value = 0;
  std::mutex mut;

public:
  explicit Semaphore(size_t val): value(val) {}
  
  void post(size_t val = 1) {
    std::unique_lock<std::mutex> lk(mut);
    value += val;
  }
  
  size_t get_value() {
    std::unique_lock<std::mutex> lk(mut);
    return value;
  }
  
  bool peek() {
    std::unique_lock<std::mutex> lk(mut);
    return value != 0;
  }
  
  bool try_wait() {
    std::unique_lock<std::mutex> lk(mut);
    if (value == 0) {
      return false;
    } else {
      value--;
      return true;
    }
  }
};

void draw(Semaphore* qsem, Semaphore* sem, Region reg, const double* time, uint32_t* pixels, const Noise& noise_gen) {
  while (qsem->peek()) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(13));
    while (sem->try_wait()) {
      for (size_t x = reg.x0; x <= reg.x1; x++) {
        for (size_t y = reg.y0; y <= reg.y1; y++) {
          // std::vector<float> amplitudes = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
          // double noise = noise_gen.octaves(x, y, time, amplitudes);
          // double noise = noise_gen.domain_wrapping(x, y, time, DIVISOR);
          // double noise = noise_gen.turbulence_ridged(x, y, time, DIVISOR);
          // double noise = noise_gen.get_value(x, y);
          // double noise = noise_gen.turbulence(x, y, DIVISOR);
          double noise = noise_gen.fbm(x, y, *time, DIVISOR);
          /// Trick to clamp values since they are out of bounds some times
          if (noise < -1.0 || noise > 1.0) {
            noise = noise < -1.0 ? -1.0 : 1.0;
          }
          double color = 0.5 + noise * 0.5;
          color = std::sqrt(color); // Gamma-2 correction
          auto ir = uint32_t(color * 255);
          auto ig = uint32_t(color * 255);
          auto ib = uint32_t(color * 255);
          auto ia = uint32_t(1);
          uint32_t pixel = 0;
          pixel += (ia << (8 * 3));
          pixel += (ir << (8 * 2));
          pixel += (ig << (8 * 1));
          pixel += (ib << (8 * 0));
          pixels[((reg.ny - y) * reg.nx) + x] = pixel;
        }
      }
    }
  }
}

int main() {
  SDL_Init(SDL_INIT_EVERYTHING);
  
  const size_t nx = 400;
  const size_t ny = 400;
  SDL_Window* window = SDL_CreateWindow("Noise", 0, 0, nx, ny, 0);
  SDL_Surface* scr = SDL_GetWindowSurface(window);
  uint32_t* pixels = (uint32_t*) scr->pixels;
  
  Perlin::Improved<> noise(SEED);
  double time = 0.0;
  
  const size_t num_threads = std::thread::hardware_concurrency() == 0 ? 4 : std::thread::hardware_concurrency();
  Semaphore sem{0};  // Signals the number thread workloads left for the current iteration
  Semaphore qsem{1}; // Signals whether or not all work is done and the threads should terminate
  std::vector<std::thread> threads{};
  for (size_t i = 0; i < num_threads; i++) {
    const size_t num_rows = ny / num_threads;
    Region reg{nx, ny, 0, nx, i * num_rows, (i + 1) * num_rows};
    threads.emplace_back(std::thread{draw, &qsem, &sem, reg, &time, pixels, noise});
  }
  
  SDL_Event event;
  bool quit = false;
  while (!quit) {
    while (SDL_PollEvent(&event)) {
        if (event.key.keysym.sym == SDLK_ESCAPE || event.type == SDL_QUIT) {
            quit = true;
            break;
        }
    }
    time += TIME_STEP;
    auto start = std::chrono::high_resolution_clock::now();
    sem.post(num_threads); // Post work for the threads
    while (!sem.peek()) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << diff << " ns/frame" << std::endl;
    SDL_UpdateWindowSurface(window);
  }
  // FIXME: Super slow closing time
  while (qsem.try_wait()) {} // Try to signal quit to all threads
  for (auto& thread : threads) {
    thread.join();
  }
  
  SDL_DestroyWindow(window);
  SDL_Quit();
  return EXIT_SUCCESS;
}