#include <iostream>
#include "noise.hpp"
#include <SDL2/SDL.h>

int main() {
    int WIDTH = 1000;
    int HEIGHT = 1000;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, 0, &window, &renderer);
    auto perlin_noise = Noise(1);
    bool quit;
    SDL_Event event;
    for (int x = 0; x < WIDTH; x++) {
        for (int y = 0; y < HEIGHT; y++) {
            auto noise = perlin_noise.perlin2d(x, y);
            auto color = noise * 255;
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