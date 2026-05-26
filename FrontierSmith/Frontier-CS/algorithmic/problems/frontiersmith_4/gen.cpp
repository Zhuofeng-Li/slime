#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Print a test case
void printTest(int n, int m, int k, vector<tuple<int,int,int>>& edges) {
    cout << n << " " << m << " " << k << "\n";
    for (auto& [x, y, c] : edges) {
        cout << x << " " << y << " " << c << "\n";
    }
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // All cases: single test output
    if (testId == 1) {
        // Exact sample case
        int n = 4, m = 4, k = 2;
        vector<tuple<int,int,int>> edges = {
            {1,2,3},{2,3,2},{3,1,4},{1,4,5}
        };
        printTest(n, m, k, edges);

    } else if (testId == 2) {
        // Single node, only self-loops, many teams
        int n = 1, m = 8, k = 3;
        vector<tuple<int,int,int>> edges;
        for (int i = 0; i < m; i++) {
            edges.push_back({1, 1, rnd.next(1, 100)});
        }
        printTest(n, m, k, edges);

    } else if (testId == 3) {
        // Star graph centered at node 1
        int n = 15, k = 5;
        vector<tuple<int,int,int>> edges;
        for (int v = 2; v <= n; v++) {
            edges.push_back({1, v, rnd.next(1, 1000000000)});
        }
        // Add some self-loops
        edges.push_back({1, 1, 5});
        edges.push_back({3, 3, 7});
        int m = (int)edges.size();
        printTest(n, m, k, edges);

    } else if (testId == 4) {
        // Linear path (chain), k=1
        int n = 20, k = 1;
        vector<tuple<int,int,int>> edges;
        for (int v = 1; v < n; v++) {
            edges.push_back({v, v+1, rnd.next(1, 1000)});
        }
        int m = (int)edges.size();
        printTest(n, m, k, edges);

    } else if (testId == 5) {
        // Linear path (chain), k=40 (many teams, few edges)
        int n = 10, k = 40;
        vector<tuple<int,int,int>> edges;
        for (int v = 1; v < n; v++) {
            edges.push_back({v, v+1, rnd.next(1, 100)});
        }
        int m = (int)edges.size();
        printTest(n, m, k, edges);

    } else if (testId == 6) {
        // Multiple parallel edges between same pairs + self-loops
        int n = 5, k = 4;
        vector<tuple<int,int,int>> edges;
        // Multiple edges 1-2
        for (int i = 0; i < 5; i++)
            edges.push_back({1, 2, rnd.next(1, 50)});
        // Multiple edges 2-3
        for (int i = 0; i < 4; i++)
            edges.push_back({2, 3, rnd.next(1, 50)});
        // Self-loops
        edges.push_back({2, 2, 10});
        edges.push_back({3, 3, 20});
        // Connect back
        edges.push_back({3, 1, rnd.next(1, 50)});
        edges.push_back({1, 4, rnd.next(1, 50)});
        edges.push_back({4, 5, rnd.next(1, 50)});
        edges.push_back({5, 1, rnd.next(1, 50)});
        int m = (int)edges.size();
        printTest(n, m, k, edges);

    } else if (testId == 7) {
        // Graph where some nodes have odd degree (requires pairing), k=3
        int n = 8, k = 3;
        vector<tuple<int,int,int>> edges;
        // Spanning tree from 1
        for (int v = 2; v <= n; v++) {
            int par = rnd.next(1, v-1);
            edges.push_back({par, v, rnd.next(1, 200)});
        }
        // Extra edges to create odd-degree vertices
        for (int i = 0; i < 10; i++) {
            int u = rnd.next(1, n), v = rnd.next(1, n);
            edges.push_back({u, v, rnd.next(1, 200)});
        }
        int m = (int)edges.size();
        printTest(n, m, k, edges);

    } else if (testId == 8) {
        // Medium-large graph, k=20, random structure
        int n = 100, k = 20;
        vector<tuple<int,int,int>> edges;
        // Spanning tree
        for (int v = 2; v <= n; v++) {
            int par = rnd.next(1, v-1);
            edges.push_back({par, v, rnd.next(1, 1000000000)});
        }
        // Extra random edges
        int extra = 200;
        for (int i = 0; i < extra; i++) {
            int u = rnd.next(1, n), v = rnd.next(1, n);
            edges.push_back({u, v, rnd.next(1, 1000000000)});
        }
        int m = (int)edges.size();
        printTest(n, m, k, edges);

    } else if (testId == 9) {
        // Large graph near max constraints, k=40
        int n = 200000, k = 40;
        vector<tuple<int,int,int>> edges;
        // Spanning tree (linear chain to ensure connectivity)
        for (int v = 2; v <= n; v++) {
            edges.push_back({v-1, v, rnd.next(1, 1000000000)});
        }
        // Extra random edges up to m=200000
        int remaining = 200000 - (n-1);
        for (int i = 0; i < remaining; i++) {
            int u = rnd.next(1, n), v = rnd.next(1, n);
            edges.push_back({u, v, rnd.next(1, 1000000000)});
        }
        int m = (int)edges.size();
        printTest(n, m, k, edges);

    } else if (testId == 10) {
        // Max constraints: n and m both 200000, k=40, random spanning tree + extra
        int n = 200000, m_target = 200000, k = 40;
        vector<tuple<int,int,int>> edges;
        // Random spanning tree
        for (int v = 2; v <= n; v++) {
            int par = rnd.next(1, v-1);
            edges.push_back({par, v, rnd.next(1, 1000000000)});
        }
        // Fill remaining with random edges (including self-loops)
        int remaining = m_target - (n-1);
        for (int i = 0; i < remaining; i++) {
            int u = rnd.next(1, n), v = rnd.next(1, n);
            edges.push_back({u, v, rnd.next(1, 1000000000)});
        }
        int m = (int)edges.size();
        printTest(n, m, k, edges);
    }

    return 0;
}