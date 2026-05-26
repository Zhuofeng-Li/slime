#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generate exactly r distinct edges among n nodes; returns list of (u,v) 0-indexed
vector<pair<int,int>> genEdges(int n, int r) {
    set<pair<int,int>> es;
    // First build a random spanning forest to seed
    vector<int> perm(n);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end());
    for (int i = 1; i < n && (int)es.size() < r; i++) {
        int u = perm[rnd.next(0, i - 1)];
        int v = perm[i];
        if (u > v) swap(u, v);
        es.insert({u, v});
    }
    int tries = 0;
    while ((int)es.size() < r && tries < 2000000) {
        tries++;
        int u = rnd.next(0, n - 1);
        int v = rnd.next(0, n - 2);
        if (v >= u) v++;
        if (u > v) swap(u, v);
        es.insert({u, v});
    }
    return vector<pair<int,int>>(es.begin(), es.end());
}

void printCase(int n, long long m, int r,
               vector<pair<long long,long long>>& pq,
               vector<tuple<int,int,int>>& edges) {
    cout << n << " " << m << " " << r << "\n";
    for (auto& [p, q] : pq)
        cout << p << " " << q << "\n";
    for (auto& [u, v, w] : edges)
        cout << u << " " << v << " " << w << "\n";
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int t = 20;
    cout << t << "\n";

    for (int tc = 0; tc < t; tc++) {
        int n, r;
        long long m;

        if (testId == 1) {
            // Tiny: n=2..5, m=n..n+10, sparse
            n = rnd.next(2, 5);
            m = (long long)n + rnd.next(0, 10);
            int maxr = min(n * (n - 1) / 2, 6);
            r = rnd.next(1, maxr);
        } else if (testId == 2) {
            // Small: n=5..15, m=n (tightest packing)
            n = rnd.next(5, 15);
            m = (long long)n;
            int maxr = min(n * (n - 1) / 2, 30);
            r = rnd.next(1, maxr);
        } else if (testId == 3) {
            // Small n, large m (lots of placement freedom)
            n = rnd.next(5, 20);
            m = rnd.next((long long)1000000, (long long)1000000000);
            int maxr = min(n * (n - 1) / 2, 40);
            r = rnd.next(1, maxr);
        } else if (testId == 4) {
            // Medium n, m close to n
            n = rnd.next(30, 80);
            m = (long long)n + rnd.next(0LL, (long long)n);
            int maxr = min(n * (n - 1) / 2, 300);
            r = rnd.next(n - 1, maxr);
        } else if (testId == 5) {
            // Medium n, large m, moderate density
            n = rnd.next(30, 80);
            m = rnd.next((long long)1000, (long long)1000000000);
            int maxr = min(n * (n - 1) / 2, 500);
            r = rnd.next(n, maxr);
        } else if (testId == 6) {
            // All oscillators share the same p/q ratio
            n = rnd.next(5, 25);
            m = rnd.next((long long)n, (long long)10000);
            int maxr = min(n * (n - 1) / 2, 60);
            r = rnd.next(1, maxr);
        } else if (testId == 7) {
            // Small q values → periodic sines at small distances
            n = rnd.next(5, 25);
            m = rnd.next((long long)n, (long long)1000);
            int maxr = min(n * (n - 1) / 2, 60);
            r = rnd.next(1, maxr);
        } else if (testId == 8) {
            // Star topology
            n = rnd.next(10, 60);
            m = rnd.next((long long)n, (long long)1000000000);
            r = n - 1;
        } else if (testId == 9) {
            // Path/chain topology
            n = rnd.next(10, 100);
            m = rnd.next((long long)n, (long long)1000000000);
            r = n - 1;
        } else {
            // Large n, large m, denser graph
            n = rnd.next(80, 190);
            m = rnd.next((long long)n, (long long)1000000000);
            int maxr = min((long long)n * (n - 1) / 2, (long long)min(30000, 3000));
            r = rnd.next(n, maxr);
        }

        // Generate oscillator ratios
        vector<pair<long long, long long>> pq(n);
        long long fixedP = rnd.next(1, 10);
        long long fixedQ = rnd.next(2, 20);
        for (int i = 0; i < n; i++) {
            if (testId == 6) {
                pq[i] = {fixedP, fixedQ};
            } else if (testId == 7) {
                long long p = rnd.next(1LL, 5LL);
                long long q = rnd.next(1LL, 10LL);
                pq[i] = {p, q};
            } else {
                long long p = rnd.next(1LL, 1000000000LL);
                long long q = rnd.next(1LL, 1000000000LL);
                pq[i] = {p, q};
            }
        }

        // Generate edges
        vector<tuple<int,int,int>> edges;
        if (testId == 8) {
            // Star: node 1 is center
            int center = 1;
            set<int> used;
            used.insert(0);
            for (int i = 1; i < n; i++) {
                int w = rnd.next(1, 1000000);
                edges.push_back({center, i + 1, w});
            }
        } else if (testId == 9) {
            // Random path permutation
            vector<int> perm(n);
            iota(perm.begin(), perm.end(), 0);
            shuffle(perm.begin(), perm.end());
            for (int i = 0; i < n - 1; i++) {
                int u = perm[i], v = perm[i + 1];
                if (u > v) swap(u, v);
                int w = rnd.next(1, 1000000);
                edges.push_back({u + 1, v + 1, w});
            }
        } else {
            auto ep = genEdges(n, r);
            for (auto& [u, v] : ep) {
                int w = rnd.next(1, 1000000);
                edges.push_back({u + 1, v + 1, w});
            }
        }

        // Ensure r matches actual edge count
        r = (int)edges.size();
        printCase(n, m, r, pq, edges);
    }

    return 0;
}