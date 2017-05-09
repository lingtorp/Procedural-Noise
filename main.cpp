#include <iostream>
#include "noise.hpp"
#include <SDL2/SDL.h>

const int WIDTH = 512;
const int HEIGHT = 512;
const int ZOOM = 1;
const int DIVISOR = 64; // Zooms into details of the noise
const int SEED = 1;
const double TIME_STEP = 0.5;

void draw(double time, SDL_Renderer *renderer, Noise &noise_gen) {
    for (size_t x = 0; x < WIDTH; x += ZOOM) {
        for (size_t y = 0; y < HEIGHT; y += ZOOM) {
            // std::vector<float> amplitudes = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
            // auto noise = noise_gen.octaves(x, y, time, amplitudes);
            // auto noise = std::abs(noise_gen.octaves(x, y, time, DIVISOR, 0.9));
            // auto noise = noise_gen.get_value(x / DIVISOR, y / DIVISOR);
            // auto noise = noise_gen.get_value(x / DIVISOR, y / DIVISOR, time / DIVISOR);
            // auto noise = noise_gen.turbulence_ridged(x, y, time, DIVISOR);
            // auto noise = noise_gen.domain_wrapping(x, y, time, DIVISOR);
            // auto noise = noise_gen.turbulence_ridged(x, y, time, DIVISOR);
            // auto noise = noise_gen.get_value(x, y);
            auto noise = noise_gen.turbulence(x, y, DIVISOR);
            auto color = 125.5f + noise * 125.5f;
            // SDL_Log("Noise: %f, Color: %f \n", noise, color);
            SDL_SetRenderDrawColor(renderer, color, color, color, 1.0f);
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
    SDL_RenderPresent(renderer);
}

int main() {
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, 0, &window, &renderer);
    Simplex_Patent noise(SEED);
    bool quit = false;
    double time = 0.0;
    auto last_tick = SDL_GetTicks();
    SDL_Event event;
    while (!quit) {
        auto new_tick = SDL_GetTicks();
        float delta = new_tick - last_tick;
        last_tick = new_tick;
        if (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                quit = true;
        }
        time += TIME_STEP;
        draw(time, renderer, noise);
        std::cout << std::floor(1000 / delta) << std::endl; // Prints fps
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}