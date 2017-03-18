#include <iostream>
#include "noise.hpp"
#include <SDL2/SDL.h>

int main() {
    int WIDTH = 1000;
    int HEIGHT = 1000;
    int ZOOM = 1;
    int DIVISOR = 2;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, 0, &window, &renderer);
    auto noise_gen = Perlin(1);
    bool quit;
    SDL_Event event;
    for (size_t x = 0; x < WIDTH; x += ZOOM) {
        for (size_t y = 0; y < HEIGHT; y += ZOOM) {
            // auto noise = noise_gen.octaves(x / DIVISOR, y / DIVISOR, 4, 0.5);
            auto noise = noise_gen.get_value(x / DIVISOR, y / DIVISOR, 2.123);
            auto color = 125.5f + noise * 125.5f;
            // SDL_Log("Noise: %f, Color: %f \n", noise, color);
            SDL_SetRenderDrawColor(renderer, color, color, color, 1.0f);
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
    SDL_RenderPresent(renderer);

    while (!quit) {
        if (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                break;
        }
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return EXIT_SUCCESS;
}