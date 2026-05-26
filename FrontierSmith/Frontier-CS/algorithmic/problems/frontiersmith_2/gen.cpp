#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Build a random connected graph with V nodes and exactly E edges (E >= V-1)
// Returns list of (u, v, w)
vector<tuple<int,int,int>> genConnectedGraph(int V, int E, int wMin, int wMax) {
    vector<tuple<int,int,int>> edges;
    set<pair<int,int>> existing;

    // Spanning tree via random Prüfer-like attachment
    vector<int> perm(V);
    iota(perm.begin(), perm.end(), 1);
    shuffle(perm.begin(), perm.end());

    for (int i = 1; i < V; i++) {
        int u = perm[rnd.next(0, i - 1)];
        int v = perm[i];
        int w = rnd.next(wMin, wMax);
        if (u > v) swap(u, v);
        existing.insert({u, v});
        edges.push_back({u, v, w});
    }

    // Extra edges
    int attempts = 0;
    while ((int)edges.size() < E && attempts < 2000000) {
        attempts++;
        int u = rnd.next(1, V);
        int v = rnd.next(1, V);
        if (u == v) continue;
        if (u > v) swap(u, v);
        if (existing.count({u, v})) continue;
        existing.insert({u, v});
        edges.push_back({u, v, rnd.next(wMin, wMax)});
    }

    shuffle(edges.begin(), edges.end());
    return edges;
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    if (testId == 1) {
        // Tiny example: 2 nodes, 1 edge, 1 pad, 1 shipment, budget just enough
        int V = 2, E = 1, M = 1, N = 1;
        long long C = 5;
        cout << V << " " << E << " " << M << " " << N << " " << C << "\n";
        cout << "1 2 10\n";
        cout << "1 5\n";
        cout << "1 2\n";
    }
    else if (testId == 2) {
        // Linear path: pads at ends help a lot; budget tight (can buy one)
        int V = 10, E = 9, M = 4, N = 6;
        long long C = 7;
        cout << V << " " << E << " " << M << " " << N << " " << C << "\n";
        for (int i = 1; i < V; i++)
            cout << i << " " << (i + 1) << " 10\n";
        // pads: at 1,3,7,10
        cout << "1 5\n";
        cout << "3 4\n";
        cout << "7 3\n";
        cout << "10 6\n";
        // shipments
        cout << "1 10\n2 9\n3 8\n4 7\n5 6\n2 8\n";
    }
    else if (testId == 3) {
        // Star graph: center (node 1) as pad is extremely beneficial
        int V = 12, E = 11, M = 3, N = 10;
        long long C = 20;
        cout << V << " " << E << " " << M << " " << N << " " << C << "\n";
        for (int i = 2; i <= V; i++)
            cout << "1 " << i << " 100\n";
        // Pad candidates: center + two leaves
        cout << "1 8\n";
        cout << "2 10\n";
        cout << "6 10\n";
        // Shipments between leaves
        int nship = 0;
        for (int a = 2; a <= V && nship < N; a++)
            for (int b = a + 1; b <= V && nship < N; b++, nship++)
                cout << a << " " << b << "\n";
    }
    else if (testId == 4) {
        // Budget is zero (smaller than cheapest pad): must build nothing
        int V = 5, E = 6, M = 3, N = 4;
        long long C = 2;
        cout << V << " " << E << " " << M << " " << N << " " << C << "\n";
        cout << "1 2 3\n1 3 5\n2 3 2\n3 4 4\n4 5 1\n2 5 8\n";
        // all pads cost more than C
        cout << "2 5\n3 5\n4 5\n";
        cout << "1 5\n2 4\n3 5\n1 4\n";
    }
    else if (testId == 5) {
        // Large budget: can afford all pads
        int V = 15, E = 20, M = 5, N = 15;
        long long C = 1000000000LL;
        auto edges = genConnectedGraph(V, E, 1, 50);
        cout << V << " " << (int)edges.size() << " " << M << " " << N << " " << C << "\n";
        for (auto& [u, v, w] : edges)
            cout << u << " " << v << " " << w << "\n";
        // M pads at random distinct nodes
        set<int> usedPads;
        for (int j = 1; j <= M; j++) {
            int node;
            do { node = rnd.next(1, V); } while (usedPads.count(node));
            usedPads.insert(node);
            long long cost = rnd.next(1LL, 100LL);
            cout << node << " " << cost << "\n";
        }
        // N random shipments
        for (int i = 0; i < N; i++) {
            int s = rnd.next(1, V), t;
            do { t = rnd.next(1, V); } while (t == s);
            cout << s << " " << t << "\n";
        }
    }
    else if (testId == 6) {
        // Grid-like graph (V=25, E~40), many shipments, moderate budget
        int side = 5;
        int V = side * side;
        vector<tuple<int,int,int>> edges;
        set<pair<int,int>> seen;
        auto idx = [&](int r, int c) { return r * side + c + 1; };
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                if (c + 1 < side) {
                    int u = idx(r, c), v = idx(r, c + 1), w = rnd.next(1, 20);
                    edges.push_back({u, v, w});
                    seen.insert({u, v});
                }
                if (r + 1 < side) {
                    int u = idx(r, c), v = idx(r + 1, c), w = rnd.next(1, 20);
                    edges.push_back({u, v, w});
                    seen.insert({u, v});
                }
            }
        }
        int E = (int)edges.size();
        int M = 8, N = 30;
        long long C = 50;
        cout << V << " " << E << " " << M << " " << N << " " << C << "\n";
        for (auto& [u, v, w] : edges)
            cout << u << " " << v << " " << w << "\n";
        set<int> usedPads;
        for (int j = 1; j <= M; j++) {
            int node;
            do { node = rnd.next(1, V); } while (usedPads.count(node));
            usedPads.insert(node);
            long long cost = rnd.next(5LL, 20LL);
            cout << node << " " << cost << "\n";
        }
        for (int i = 0; i < N; i++) {
            int s = rnd.next(1, V), t;
            do { t = rnd.next(1, V); } while (t == s);
            cout << s << " " << t << "\n";
        }
    }
    else if (testId == 7) {
        // Single pad candidate, very high benefit, affordable
        int V = 20, E = 30, M = 1, N = 20;
        long long C = 100;
        auto edges = genConnectedGraph(V, E, 1, 10);
        cout << V << " " << (int)edges.size() << " " << M << " " << N << " " << C << "\n";
        for (auto& [u, v, w] : edges)
            cout << u << " " << v << " " << w << "\n";
        // Pad at node 1 (well-connected) at cheap cost
        cout << "1 10\n";
        for (int i = 0; i < N; i++) {
            int s = rnd.next(1, V), t;
            do { t = rnd.next(1, V); } while (t == s);
            cout << s << " " << t << "\n";
        }
    }
    else if (testId == 8) {
        // Many shipments (N=100000), moderate graph, stress test
        int V = 100, E = 300, M = 20, N = 100000;
        long long C = 500;
        auto edges = genConnectedGraph(V, E, 1, 100);
        cout << V << " " << (int)edges.size() << " " << M << " " << N << " " << C << "\n";
        for (auto& [u, v, w] : edges)
            cout << u << " " << v << " " << w << "\n";
        set<int> usedPads;
        for (int j = 1; j <= M; j++) {
            int node;
            do { node = rnd.next(1, V); } while (usedPads.count(node));
            usedPads.insert(node);
            long long cost = rnd.next(10LL, 50LL);
            cout << node << " " << cost << "\n";
        }
        for (int i = 0; i < N; i++) {
            int s = rnd.next(1, V), t;
            do { t = rnd.next(1, V); } while (t == s);
            cout << s << " " << t << "\n";
        }
    }
    else if (testId == 9) {
        // Near-maximum constraints: large V, E, M, N
        int V = 2000, E = 10000, M = 500, N = 50000;
        long long C = (long long)rnd.next(1000000000, 2000000000);
        auto edges = genConnectedGraph(V, E, 1, 1000000000);
        cout << V << " " << (int)edges.size() << " " << M << " " << N << " " << C << "\n";
        for (auto& [u, v, w] : edges)
            cout << u << " " << v << " " << w << "\n";
        set<int> usedPads;
        for (int j = 1; j <= M; j++) {
            int node;
            do { node = rnd.next(1, V); } while (usedPads.count(node));
            usedPads.insert(node);
            long long cost = rnd.next(1LL, 1000000000LL);
            cout << node << " " << cost << "\n";
        }
        for (int i = 0; i < N; i++) {
            int s = rnd.next(1, V), t;
            do { t = rnd.next(1, V); } while (t == s);
            cout << s << " " << t << "\n";
        }
    }
    else {
        // testId == 10: Max constraints
        int V = 4000, E = 20000, M = 1000, N = 80000;
        long long C = 500000000000LL;
        auto edges = genConnectedGraph(V, E, 1, 1000000000);
        cout << V << " " << (int)edges.size() << " " << M << " " << N << " " << C << "\n";
        for (auto& [u, v, w] : edges)
            cout << u << " " << v << " " << w << "\n";
        set<int> usedPads;
        for (int j = 1; j <= M; j++) {
            int node;
            do { node = rnd.next(1, V); } while (usedPads.count(node));
            usedPads.insert(node);
            long long cost = rnd.next(1LL, 1000000000LL);
            cout << node << " " << cost << "\n";
        }
        for (int i = 0; i < N; i++) {
            int s = rnd.next(1, V), t;
            do { t = rnd.next(1, V); } while (t == s);
            cout << s << " " << t << "\n";
        }
    }

    return 0;
}