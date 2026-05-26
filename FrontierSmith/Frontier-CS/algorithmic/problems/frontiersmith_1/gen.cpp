#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generate a connected graph with n nodes and at least n-1 edges
// Returns list of edges (u,v) 1-indexed
vector<pair<int,int>> genConnectedGraph(int n, int extraEdges) {
    vector<pair<int,int>> edges;
    // Random spanning tree via Prüfer-like: shuffle [2..n], connect each to a random earlier node
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i + 1;
    shuffle(perm.begin(), perm.end());
    // perm[0] is root; perm[i] connects to perm[rnd.next(0, i-1)]
    for (int i = 1; i < n; i++) {
        int u = perm[i];
        int v = perm[rnd.next(0, i - 1)];
        edges.push_back({u, v});
    }
    // Add extra edges (no self-loops, may duplicate but we'll filter)
    set<pair<int,int>> edgeSet;
    for (auto& e : edges) {
        int u = min(e.first, e.second);
        int v = max(e.first, e.second);
        edgeSet.insert({u, v});
    }
    int attempts = 0;
    int added = 0;
    while (added < extraEdges && attempts < 10000) {
        attempts++;
        int u = rnd.next(1, n);
        int v = rnd.next(1, n);
        if (u == v) continue;
        if (u > v) swap(u, v);
        if (edgeSet.count({u, v})) continue;
        edgeSet.insert({u, v});
        edges.push_back({u, v});
        added++;
    }
    return edges;
}

void printGraph(int n, vector<pair<int,int>>& edges,
                vector<int>& a, vector<int>& b, vector<int>& c) {
    int e = edges.size();
    cout << n << " " << e << "\n";
    for (int i = 0; i < e; i++) {
        cout << edges[i].first << " " << edges[i].second << " "
             << a[i] << " " << b[i] << " " << c[i] << "\n";
    }
}

void printDays(int n, int m, int maxL, int maxR, int maxV, int maxC) {
    cout << m << "\n";
    for (int d = 0; d < m; d++) {
        // nodes 2..n are candidates
        int availNodes = n - 1;
        int r = rnd.next(1, min(maxR, availNodes));
        int L = rnd.next(0, maxL);
        // pick r distinct nodes from [2..n]
        vector<int> nodes;
        for (int i = 2; i <= n; i++) nodes.push_back(i);
        shuffle(nodes.begin(), nodes.end());
        nodes.resize(r);
        cout << r << " " << L;
        for (int i = 0; i < r; i++) {
            int val = rnd.next(1, maxV);
            cout << " " << nodes[i] << " " << val;
        }
        cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    if (testId == 1) {
        // Absolute minimum: n=2, e=1, m=1, 1 depot
        int n = 2;
        vector<pair<int,int>> edges = {{1, 2}};
        vector<int> a = {rnd.next(1, 10)};
        vector<int> b = {rnd.next(1, 10)};
        vector<int> c = {rnd.next(1, 5)};
        printGraph(n, edges, a, b, c);
        cout << 1 << "\n";
        int val = rnd.next(1, 100);
        cout << 1 << " " << 1 << " " << 2 << " " << val << "\n";

    } else if (testId == 2) {
        // Chain graph n=5, tree only, m=5
        int n = 5;
        vector<pair<int,int>> edges = {{1,2},{2,3},{3,4},{4,5}};
        int e = edges.size();
        vector<int> a(e), b(e), c(e);
        for (int i = 0; i < e; i++) {
            a[i] = rnd.next(1, 20);
            b[i] = rnd.next(1, 10);
            c[i] = rnd.next(1, 5);
        }
        printGraph(n, edges, a, b, c);
        printDays(n, 5, 2, 3, 50, 5);

    } else if (testId == 3) {
        // Tree with n=15, high depot values, moderate L
        int n = 15;
        int extra = 0;
        vector<pair<int,int>> edges = genConnectedGraph(n, extra);
        int e = edges.size();
        vector<int> a(e), b(e), c(e);
        for (int i = 0; i < e; i++) {
            a[i] = rnd.next(1, 50);
            b[i] = rnd.next(1, 30);
            c[i] = rnd.next(1, 10);
        }
        printGraph(n, edges, a, b, c);
        printDays(n, 8, 5, 5, 1000, 10);

    } else if (testId == 4) {
        // Graph with cycles, n=20, moderate settings
        int n = 20;
        int extra = rnd.next(3, 8);
        vector<pair<int,int>> edges = genConnectedGraph(n, extra);
        int e = edges.size();
        vector<int> a(e), b(e), c(e);
        for (int i = 0; i < e; i++) {
            a[i] = rnd.next(1, 100);
            b[i] = rnd.next(1, 50);
            c[i] = rnd.next(1, 10);
        }
        printGraph(n, edges, a, b, c);
        printDays(n, 10, 5, 7, 500, 10);

    } else if (testId == 5) {
        // Star graph from node 1, n=10, many days
        int n = 10;
        vector<pair<int,int>> edges;
        for (int i = 2; i <= n; i++) edges.push_back({1, i});
        // add a few extra
        for (int i = 0; i < 3; i++) {
            int u = rnd.next(2, n);
            int v = rnd.next(2, n);
            if (u != v) edges.push_back({u, v});
        }
        int e = edges.size();
        vector<int> a(e), b(e), c(e);
        for (int i = 0; i < e; i++) {
            a[i] = rnd.next(1, 30);
            b[i] = rnd.next(1, 15);
            c[i] = rnd.next(1, 8);
        }
        printGraph(n, edges, a, b, c);
        printDays(n, 15, 8, 5, 200, 15);

    } else if (testId == 6) {
        // Large L values (can detonate many bridges per day), n=30
        int n = 30;
        int extra = rnd.next(5, 15);
        vector<pair<int,int>> edges = genConnectedGraph(n, extra);
        int e = edges.size();
        vector<int> a(e), b(e), c(e);
        for (int i = 0; i < e; i++) {
            a[i] = rnd.next(1, 200);
            b[i] = rnd.next(1, 100);
            c[i] = rnd.next(1, 20);
        }
        printGraph(n, edges, a, b, c);
        printDays(n, 12, 20, 10, 1000, 12);

    } else if (testId == 7) {
        // c_j = 1 for all bridges (can only detonate once), n=25, many days
        int n = 25;
        int extra = rnd.next(3, 10);
        vector<pair<int,int>> edges = genConnectedGraph(n, extra);
        int e = edges.size();
        vector<int> a(e), b(e), c(e);
        for (int i = 0; i < e; i++) {
            a[i] = rnd.next(1, 80);
            b[i] = rnd.next(1, 40);
            c[i] = 1;  // can only detonate once
        }
        printGraph(n, edges, a, b, c);
        printDays(n, 20, 5, 8, 500, 20);

    } else if (testId == 8) {
        // High installation costs, low detonation costs, n=20
        int n = 20;
        int extra = rnd.next(2, 7);
        vector<pair<int,int>> edges = genConnectedGraph(n, extra);
        int e = edges.size();
        vector<int> a(e), b(e), c(e);
        for (int i = 0; i < e; i++) {
            a[i] = rnd.next(500, 1000);  // high install
            b[i] = rnd.next(1, 10);      // low blast
            c[i] = rnd.next(3, 15);
        }
        printGraph(n, edges, a, b, c);
        printDays(n, 10, 6, 8, 1000, 10);

    } else if (testId == 9) {
        // Low installation costs, high detonation costs, n=20
        int n = 20;
        int extra = rnd.next(2, 7);
        vector<pair<int,int>> edges = genConnectedGraph(n, extra);
        int e = edges.size();
        vector<int> a(e), b(e), c(e);
        for (int i = 0; i < e; i++) {
            a[i] = rnd.next(1, 10);       // low install
            b[i] = rnd.next(500, 1000);   // high blast
            c[i] = rnd.next(1, 10);
        }
        printGraph(n, edges, a, b, c);
        printDays(n, 10, 6, 8, 1000, 10);

    } else {
        // testId == 10: Larger stress with dense connections, n=50, m=20
        int n = 50;
        int extra = rnd.next(15, 30);
        vector<pair<int,int>> edges = genConnectedGraph(n, extra);
        int e = edges.size();
        vector<int> a(e), b(e), c(e);
        for (int i = 0; i < e; i++) {
            a[i] = rnd.next(1, 500);
            b[i] = rnd.next(1, 200);
            c[i] = rnd.next(1, 20);
        }
        printGraph(n, edges, a, b, c);
        printDays(n, 20, 10, 15, 1000, 20);
    }

    return 0;
}