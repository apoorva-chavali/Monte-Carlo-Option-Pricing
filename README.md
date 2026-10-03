# GPU-Accelerated Monte Carlo Option Pricing

A CUDA implementation of Monte Carlo pricing for a European call option under the Black-Scholes model, with a single-threaded CPU baseline for comparison of price and runtime.

## Model

The stock price at expiry follows geometric Brownian motion:

$$S_T = S_0 \exp\left((r - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}\,Z\right), \quad Z \sim \mathcal{N}(0,1)$$

The option price is the discounted expected payoff, estimated by averaging over simulated paths:

$$C = e^{-rT}\,\mathbb{E}\left[\max(S_T - K,\ 0)\right]$$

| Parameter | Value |
|---|---|
| Spot price `S0` | 100 |
| Strike `K` | 100 |
| Risk-free rate `r` | 5% |
| Volatility `sigma` | 20% |
| Maturity `T` | 1 year |
| Paths | 10,000,000 |

For these parameters the Black-Scholes closed-form price is approximately **10.4506**.

## Implementation

### GPU (`monte_carlo_kernel`)

- **Launch configuration:** 1024 blocks × 256 threads = 262,144 threads.
- **Random numbers:** each thread initializes its own cuRAND state with a shared seed and its thread ID as the subsequence, giving non-overlapping random streams.
- **Work distribution:** a grid-stride loop assigns paths to threads, about 38 paths per thread for 10M paths.
- **Reduction:** each thread accumulates payoffs in a register and writes a single partial sum to global memory. There is no inter-thread communication, so no atomics or synchronization are needed in the kernel.
- **Final sum:** partial sums are copied to the host, summed in double precision, and discounted.

### CPU baseline (`cpu_monte_carlo`)

A serial loop over all paths using `std::mt19937` (seed 42) and `std::normal_distribution`, computing the same payoff and discounting.

### Timing

- **GPU kernel time** is measured with CUDA events around the kernel launch. It includes cuRAND state initialization.
- **GPU total time** is measured with `std::chrono` and includes device allocation, the kernel, the device-to-host copy, and the host-side reduction.
- **CPU time** is measured with `std::chrono` around the CPU function.
- **Speedup** is reported as CPU time divided by GPU total time.

## Build and Run

Requires the CUDA toolkit with cuRAND.

```bash
nvcc -O3 -std=c++17 main.cu -o montecarlo -lcurand
./montecarlo
```

Profiling:

```bash
nvprof ./montecarlo
```

## Output

The program prints:

```
GPU kernel time: <ms>

CPU price: <price>
CPU time: <ms>

GPU price: <price>
GPU total time: <ms>

Speedup: <x>
```

## Tech Stack

CUDA, C++17, cuRAND, nvprof
