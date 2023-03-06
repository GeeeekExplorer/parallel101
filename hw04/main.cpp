#include <cstdio>
#include <cstdlib>
#include <array>
#include <chrono>
#include <cmath>

float frand() {
    return (float)rand() / RAND_MAX * 2 - 1;
}

const size_t N = 48;
struct Star {
    float px[N], py[N], pz[N];
    float vx[N], vy[N], vz[N];
    float mass[N];
};

Star stars;

void init() {
    for (size_t i = 0; i < N; i++) {
        stars.px[i] = frand();
        stars.py[i] = frand();
        stars.pz[i] = frand();
        stars.vx[i] = frand();
        stars.vy[i] = frand();
        stars.vz[i] = frand();
        stars.mass[i] = frand() + 1;
    }
}

const float G = 0.001;
const float eps = 0.001;
const float dt = 0.01;
const float Gdt = G * dt;
const float eps2 = eps * eps;

void step() {
    for (size_t i = 0; i < N; i++) {
        float px = stars.px[i], py = stars.py[i], pz = stars.pz[i];
        float vx = stars.vx[i], vy = stars.vy[i], vz = stars.vz[i];
        #pragma omp simd
        for (size_t j = 0; j < N; j++) {
            float dx = stars.px[j] - px;
            float dy = stars.py[j] - py;
            float dz = stars.pz[j] - pz;
            float d2 = dx * dx + dy * dy + dz * dz + eps2;
            d2 *= std::sqrt(d2);
            float t = stars.mass[j] / d2 * Gdt;
            vx += dx * t;
            vy += dy * t;
            vz += dz * t;
        }
        stars.vx[i] = vx, stars.vy[i] = vy, stars.vz[i] = vz;
    }
    #pragma omp simd
    for (size_t i = 0; i < N; i++) {
        stars.px[i] += stars.vx[i] * dt;
        stars.py[i] += stars.vy[i] * dt;
        stars.pz[i] += stars.vz[i] * dt;
    }
}

float calc() {
    float energy = 0;
    for (size_t i = 0; i < N; i++) {
        float px = stars.px[i], py = stars.py[i], pz = stars.pz[i];
        float v2 = stars.vx[i] * stars.vx[i] + stars.vy[i] * stars.vy[i] + stars.vz[i] * stars.vz[i];
        energy += stars.mass[i] * v2 / 2;
        float t = stars.mass[i] * G / 2;
        #pragma omp simd
        for (size_t j = 0; j < N; j++) {
            float dx = stars.px[j] - px;
            float dy = stars.py[j] - py;
            float dz = stars.pz[j] - pz;
            float d2 = dx * dx + dy * dy + dz * dz + eps2;
            energy -= stars.mass[j] / std::sqrt(d2) * t;
        }
    }
    return energy;
}

template <class Func>
long benchmark(Func const &func) {
    auto t0 = std::chrono::steady_clock::now();
    func();
    auto t1 = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    return dt.count();
}

int main() {
    init();
    printf("Initial energy: %f\n", calc());
    auto dt = benchmark([&] {
        for (int i = 0; i < 100000; i++)
            step();
    });
    printf("Final energy: %f\n", calc());
    printf("Time elapsed: %ld ms\n", dt);
    return 0;
}
