#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int M;
    long long B;
    vector<long long> cost(0), w(0), d(0);

    auto ensureAffordable = [&]() {
        // Guarantee at least one port is affordable
        int cheapest = 0;
        for (int i = 1; i < M; i++) if (cost[i] < cost[cheapest]) cheapest = i;
        if (cost[cheapest] > B) cost[cheapest] = B;
    };

    auto printTest = [&]() {
        cout << M << " " << B << "\n";
        for (int i = 0; i < M; i++) cout << cost[i] << (i == M-1 ? "\n" : " ");
        for (int i = 0; i < M; i++) cout << w[i]    << (i == M-1 ? "\n" : " ");
        for (int i = 0; i < M; i++) cout << d[i]    << (i == M-1 ? "\n" : " ");
    };

    if (testId == 1) {
        // Minimal M=2, tight budget picks only port 0
        M = 2; B = 1;
        cost = {1, 2};
        w    = {5, 3};
        d    = {1, 1};

    } else if (testId == 2) {
        // Exact example from problem statement
        M = 6; B = 5;
        cost = {3, 2, 2, 4, 1, 3};
        w    = {5, 1, 4, 2, 3, 6};
        d    = {1, 2, 1, 1, 2, 1};

    } else if (testId == 3) {
        // All costs = 1, huge budget -> can pick everyone; tests saturation interaction
        M = 20; B = 20;
        cost.resize(M); w.resize(M); d.resize(M);
        for (int i = 0; i < M; i++) {
            cost[i] = 1;
            w[i]    = rnd.next(1, 1000000);
            d[i]    = rnd.next(1, M);
        }

    } else if (testId == 4) {
        // Very tight budget: only one port can be afforded
        M = 30; B = 3;
        cost.resize(M); w.resize(M); d.resize(M);
        for (int i = 0; i < M; i++) {
            cost[i] = rnd.next(4, 1000);
            w[i]    = rnd.next(1, 1000000);
            d[i]    = rnd.next(1, M);
        }
        // Force exactly one affordable port
        int lucky = rnd.next(0, M-1);
        cost[lucky] = 3;

    } else if (testId == 5) {
        // All saturation limits = 1: each residue can contribute at most w[r]
        M = 100; B = 200;
        cost.resize(M); w.resize(M); d.resize(M);
        for (int i = 0; i < M; i++) {
            cost[i] = rnd.next(1, 10);
            w[i]    = rnd.next(1, 1000000);
            d[i]    = 1;
        }

    } else if (testId == 6) {
        // All saturation limits = M (no effective saturation), uniform high weights
        M = 200; B = 500;
        cost.resize(M); w.resize(M); d.resize(M);
        for (int i = 0; i < M; i++) {
            cost[i] = rnd.next(1, 20);
            w[i]    = 1000000;
            d[i]    = M;
        }

    } else if (testId == 7) {
        // Medium M, wildly varying costs and weights
        M = 2000; B = 50000LL;
        cost.resize(M); w.resize(M); d.resize(M);
        for (int i = 0; i < M; i++) {
            cost[i] = rnd.next(1LL, 1000000000LL);
            w[i]    = rnd.next(1, 1000000);
            d[i]    = rnd.next(1, M);
        }
        ensureAffordable();

    } else if (testId == 8) {
        // Large M, costs clustered near budget boundary
        M = 10000; B = 1000000LL;
        cost.resize(M); w.resize(M); d.resize(M);
        for (int i = 0; i < M; i++) {
            cost[i] = rnd.next(1LL, 200000LL);
            w[i]    = rnd.next(1, 1000000);
            d[i]    = rnd.next(1, M);
        }

    } else if (testId == 9) {
        // Large M = 100000, dense affordable ports
        M = 100000; B = (long long)2e14;
        cost.resize(M); w.resize(M); d.resize(M);
        for (int i = 0; i < M; i++) {
            cost[i] = rnd.next(1LL, 1000000LL);
            w[i]    = rnd.next(1, 1000000);
            d[i]    = rnd.next(1, M);
        }

    } else {
        // Maximum M = 200000, maximum budget, stress test
        M = 200000; B = (long long)1e18;
        cost.resize(M); w.resize(M); d.resize(M);
        for (int i = 0; i < M; i++) {
            cost[i] = rnd.next(1LL, 1000000000LL);
            w[i]    = rnd.next(1, 1000000);
            d[i]    = rnd.next(1, M);
        }
    }

    // Validate constraints before printing
    ensuref(M >= 2 && M <= 200000, "M out of range");
    ensuref(B >= 1, "B must be positive");
    ensuref((int)cost.size() == M, "cost size mismatch");
    ensuref((int)w.size() == M, "w size mismatch");
    ensuref((int)d.size() == M, "d size mismatch");
    bool anyAffordable = false;
    for (int i = 0; i < M; i++) {
        ensuref(cost[i] >= 1 && cost[i] <= 1000000000LL, "cost[i] out of range");
        ensuref(w[i] >= 1 && w[i] <= 1000000, "w[i] out of range");
        ensuref(d[i] >= 1 && d[i] <= M, "d[i] out of range");
        if (cost[i] <= B) anyAffordable = true;
    }
    ensuref(anyAffordable, "No affordable port");

    printTest();
    return 0;
}