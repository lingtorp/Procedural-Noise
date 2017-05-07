#include <iostream>
#include "noise.hpp"
#include <SDL2/SDL.h>
#include <chrono>

const int WIDTH        = 512;
const int HEIGHT       = 512;
const int DIVISOR      = 64;
const int SEED         = 1;
const double TIME_STEP = 0.1;

/// Used to prevent the removal of the assignment when using optimizations enabled
volatile static double *mem_dump;

void draw(double time, SDL_Renderer *renderer, Noise &noise_gen) {
    for (size_t x = 0; x < WIDTH; x++) {
        for (size_t y = 0; y < HEIGHT; y++) {
            auto noise = noise_gen.turbulence(x, y, time, DIVISOR);
            auto color = 125.5f + noise * 125.5f;
            SDL_SetRenderDrawColor(renderer, color, color, color, 1.0f);
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
    SDL_RenderPresent(renderer);
}

inline void draw(double time, Noise &noise_gen) {
    for (size_t x = 0; x < WIDTH; x++) {
        for (size_t y = 0; y < HEIGHT; y++) {
            double noise = noise_gen.turbulence(x, y, time, DIVISOR);
            double color = 125.5f + noise * 125.5f;
            *mem_dump = color;
        }
    }
}

int main() {
    mem_dump = (double *) malloc(sizeof(double)); // Should prevent removal by optimization

    Perlin noise(SEED);
    double time = 0.0;
    int frames = 0;
    int total_frames = 10; // Total frames to render
    std::vector<float> execution_times(total_frames);

    while (frames < total_frames) {
        time += TIME_STEP;
        std::chrono::high_resolution_clock::time_point start  = std::chrono::high_resolution_clock::now();
        draw(time, noise);
        std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
        execution_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / 1000.0f);
        frames++;
    }

    std::cout << "Avg. execution time: " << std::accumulate(execution_times.begin(), execution_times.end(), 0) / total_frames << " ms" << std::endl;

    std::cout << mem_dump + execution_times.size() << std::endl; // Should prevent removal by optimization

    return EXIT_SUCCESS;
}