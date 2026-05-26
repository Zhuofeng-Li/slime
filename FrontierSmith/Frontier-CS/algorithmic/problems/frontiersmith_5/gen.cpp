#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);

    int testId = atoi(argv[1]);

    int n, m, t, k;

    // Helper lambda to build and print a random tree
    // Returns parent array (1-indexed, par[1]=0)
    auto printTree = [&](int N, vector<pair<int,int>>& edges) {
        // edges already built, just print them
        for (auto& e : edges) {
            cout << e.first << " " << e.second << "\n";
        }
    };

    auto makePathTree = [&](int N) -> vector<pair<int,int>> {
        vector<pair<int,int>> edges;
        for (int i = 2; i <= N; i++) edges.push_back({i-1, i});
        return edges;
    };

    auto makeStarTree = [&](int N) -> vector<pair<int,int>> {
        vector<pair<int,int>> edges;
        for (int i = 2; i <= N; i++) edges.push_back({1, i});
        return edges;
    };

    auto makeRandomTree = [&](int N) -> vector<pair<int,int>> {
        vector<pair<int,int>> edges;
        for (int i = 2; i <= N; i++) {
            int p = rnd.next(1, i-1);
            edges.push_back({p, i});
        }
        return edges;
    };

    auto makeBinaryTree = [&](int N) -> vector<pair<int,int>> {
        vector<pair<int,int>> edges;
        for (int i = 2; i <= N; i++) {
            int p = i / 2;
            edges.push_back({p, i});
        }
        return edges;
    };

    if (testId == 1) {
        // Minimal: small n, small m, few observatories
        n = 5; m = 10; t = 3; k = 3;
        cout << n << " " << m << " " << t << " " << k << "\n";
        // a_i
        cout << "0 1 4 6 9\n";
        // c_i
        cout << "5 2 2 1 1\n";
        cout << "1 2\n1 3\n3 4\n3 5\n";
        cout << "1 4\n3 3\n4 2\n";

    } else if (testId == 2) {
        // Path tree, medium size
        n = 20; m = 7; t = 10; k = 5;
        cout << n << " " << m << " " << t << " " << k << "\n";
        for (int i = 1; i <= n; i++) cout << rnd.next(0, 1000000000) << " \n"[i==n];
        for (int i = 1; i <= n; i++) cout << rnd.next(1, 1000000000) << " \n"[i==n];
        auto edges = makePathTree(n);
        for (auto& e : edges) cout << e.first << " " << e.second << "\n";
        for (int j = 0; j < t; j++) {
            int s = rnd.next(1, n);
            int w = rnd.next(1, 1000000000);
            cout << s << " " << w << "\n";
        }

    } else if (testId == 3) {
        // Star tree, many observatories at leaves, k=1
        n = 50; m = 13; t = 30; k = 1;
        cout << n << " " << m << " " << t << " " << k << "\n";
        for (int i = 1; i <= n; i++) cout << rnd.next(0, 100) << " \n"[i==n];
        for (int i = 1; i <= n; i++) cout << rnd.next(1, 10) << " \n"[i==n];
        auto edges = makeStarTree(n);
        for (auto& e : edges) cout << e.first << " " << e.second << "\n";
        for (int j = 0; j < t; j++) {
            int s = rnd.next(1, n);
            int w = rnd.next(1, 1000000000);
            cout << s << " " << w << "\n";
        }

    } else if (testId == 4) {
        // All a_i = 0, small m, binary tree
        n = 100; m = 5; t = 50; k = 10;
        cout << n << " " << m << " " << t << " " << k << "\n";
        for (int i = 1; i <= n; i++) cout << 0 << " \n"[i==n];
        for (int i = 1; i <= n; i++) cout << rnd.next(1, 100) << " \n"[i==n];
        auto edges = makeBinaryTree(n);
        for (auto& e : edges) cout << e.first << " " << e.second << "\n";
        for (int j = 0; j < t; j++) {
            int s = rnd.next(1, n);
            int w = rnd.next(100, 1000000000);
            cout << s << " " << w << "\n";
        }

    } else if (testId == 5) {
        // k=0: no retuners allowed (trivial output)
        n = 200; m = 11; t = 100; k = 0;
        // Wait, k>=1 per constraints? No: "1 <= k <= n" per constraints.
        // Actually re-reading: "1 <= k <= n" so k>=1. Use k=1.
        k = 1;
        cout << n << " " << m << " " << t << " " << k << "\n";
        for (int i = 1; i <= n; i++) cout << rnd.next(0, 1000000000) << " \n"[i==n];
        // High costs to discourage retuner installation
        for (int i = 1; i <= n; i++) cout << 1000000000 << " \n"[i==n];
        auto edges = makeRandomTree(n);
        for (auto& e : edges) cout << e.first << " " << e.second << "\n";
        for (int j = 0; j < t; j++) {
            int s = rnd.next(1, n);
            int w = rnd.next(1, 100);
            cout << s << " " << w << "\n";
        }

    } else if (testId == 6) {
        // Large n, small m=3 (only prime is 2)
        n = 1000; m = 3; t = 500; k = 50;
        cout << n << " " << m << " " << t << " " << k << "\n";
        for (int i = 1; i <= n; i++) cout << rnd.next(0, 1000000000) << " \n"[i==n];
        for (int i = 1; i <= n; i++) cout << rnd.next(1, 1000) << " \n"[i==n];
        auto edges = makeRandomTree(n);
        for (auto& e : edges) cout << e.first << " " << e.second << "\n";
        for (int j = 0; j < t; j++) {
            // Observatories mostly at root to stress test
            int s = (rnd.next(0, 3) == 0) ? 1 : rnd.next(1, n);
            int w = rnd.next(1, 1000000000);
            cout << s << " " << w << "\n";
        }

    } else if (testId == 7) {
        // Large m=997 (prime), many primes in P
        n = 500; m = 997; t = 200; k = 20;
        cout << n << " " << m << " " << t << " " << k << "\n";
        for (int i = 1; i <= n; i++) cout << rnd.next(0, 1000000000) << " \n"[i==n];
        for (int i = 1; i <= n; i++) cout << rnd.next(1, 1000000000) << " \n"[i==n];
        auto edges = makeBinaryTree(n);
        for (auto& e : edges) cout << e.first << " " << e.second << "\n";
        for (int j = 0; j < t; j++) {
            int s = rnd.next(1, n);
            int w = rnd.next(1, 1000000000);
            cout << s << " " << w << "\n";
        }

    } else if (testId == 8) {
        // Many observatories all at root, deep path
        n = 1000; m = 10; t = 1000; k = 100;
        cout << n << " " << m << " " << t << " " << k << "\n";
        for (int i = 1; i <= n; i++) cout << rnd.next(0, 9) << " \n"[i==n];
        for (int i = 1; i <= n; i++) cout << rnd.next(1, 1000000) << " \n"[i==n];
        auto edges = makePathTree(n);
        for (auto& e : edges) cout << e.first << " " << e.second << "\n";
        for (int j = 0; j < t; j++) {
            // All observatories at root
            cout << 1 << " " << rnd.next(1, 1000000000) << "\n";
        }

    } else if (testId == 9) {
        // Max n, balanced random tree, medium m
        n = 100000; m = 100; t = 100000; k = 1000;
        cout << n << " " << m << " " << t << " " << k << "\n";
        for (int i = 1; i <= n; i++) cout << rnd.next(0, 1000000000) << " \n"[i==n];
        for (int i = 1; i <= n; i++) cout << rnd.next(1, 1000000000) << " \n"[i==n];
        auto edges = makeRandomTree(n);
        for (auto& e : edges) cout << e.first << " " << e.second << "\n";
        for (int j = 0; j < t; j++) {
            int s = rnd.next(1, n);
            int w = rnd.next(1, 1000000000);
            cout << s << " " << w << "\n";
        }

    } else if (testId == 10) {
        // Max n, path tree, max m, max t, max k - stress test
        n = 100000; m = 1000; t = 100000; k = 100000;
        cout << n << " " << m << " " << t << " " << k << "\n";
        for (int i = 1; i <= n; i++) cout << rnd.next(0, 1000000000) << " \n"[i==n];
        for (int i = 1; i <= n; i++) cout << rnd.next(1, 1000000000) << " \n"[i==n];
        // Mix of path and random
        vector<pair<int,int>> edges;
        for (int i = 2; i <= n; i++) {
            // Mostly sequential but occasionally jump to shallower node
            int lo = max(1, i - 10);
            int p = rnd.next(lo, i-1);
            edges.push_back({p, i});
        }
        for (auto& e : edges) cout << e.first << " " << e.second << "\n";
        for (int j = 0; j < t; j++) {
            int s = rnd.next(1, n);
            int w = rnd.next(1, 1000000000);
            cout << s << " " << w << "\n";
        }
    }

    return 0;
}