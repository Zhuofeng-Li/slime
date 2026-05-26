#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);

    int testId = atoi(argv[1]);

    int L, N, M;
    // mode: 0=all A, 1=all J, 2=alternating, 3=random, 4=blocks
    int sMode;

    if (testId == 1) {
        // Absolute minimal
        L = 1; N = 1; M = 1; sMode = 0;
    } else if (testId == 2) {
        // Small, all A routes, residents with A coupons (free shuttles)
        L = 6; N = 3; M = 3; sMode = 0;
    } else if (testId == 3) {
        // Small, all J routes, residents with A coupons (expensive shuttles)
        L = 6; N = 3; M = 3; sMode = 1;
    } else if (testId == 4) {
        // Small alternating routes
        L = 10; N = 5; M = 5; sMode = 2;
    } else if (testId == 5) {
        // Medium mixed random routes
        L = 200; N = 100; M = 500; sMode = 3;
    } else if (testId == 6) {
        // Very large L, few residents and demands
        L = 5000; N = 20; M = 50; sMode = 3;
    } else if (testId == 7) {
        // Many residents, small L (lots of shuttle candidates)
        L = 50; N = 5000; M = 200; sMode = 2;
    } else if (testId == 8) {
        // Many demands, moderate L
        L = 500; N = 50; M = 20000; sMode = 3;
    } else if (testId == 9) {
        // Maximum scale
        L = 5000; N = 5000; M = 20000; sMode = 3;
    } else {
        // Large L with block-pattern routes, large D residents
        L = 2000; N = 500; M = 5000; sMode = 4;
    }

    // Generate route string S
    string S = "";
    if (sMode == 0) {
        S = string(L, 'A');
    } else if (sMode == 1) {
        S = string(L, 'J');
    } else if (sMode == 2) {
        for (int k = 0; k < L; k++) S += (k % 2 == 0 ? 'A' : 'J');
    } else if (sMode == 3) {
        for (int k = 0; k < L; k++) S += (rnd.next(0, 1) == 0 ? 'A' : 'J');
    } else {
        // blocks of size ~50
        int blockSize = max(1, L / 20);
        char cur = 'A';
        int cnt = 0;
        for (int k = 0; k < L; k++) {
            S += cur;
            cnt++;
            if (cnt >= blockSize) {
                cnt = 0;
                cur = (cur == 'A') ? 'J' : 'A';
            }
        }
    }

    cout << L << " " << N << " " << M << "\n";
    cout << S << "\n";

    // Generate residents
    for (int i = 0; i < N; i++) {
        int x = rnd.next(0, L);
        char c = (rnd.next(0, 1) == 0) ? 'A' : 'J';
        int h;
        if (testId == 2) {
            // Cheap residents with matching coupon
            h = rnd.next(0, 5);
            c = 'A';
        } else if (testId == 3) {
            // Expensive residents with wrong coupon
            h = rnd.next(100, 1000000000);
            c = 'A';
        } else if (testId == 10) {
            // Residents with large D (can span far)
            h = rnd.next(0, 100);
            c = (rnd.next(0, 1) == 0) ? 'A' : 'J';
        } else {
            h = rnd.next(0, 1000000000);
        }
        int d;
        if (testId == 10) {
            d = rnd.next(L / 2, L);
        } else {
            d = rnd.next(1, L);
        }
        cout << x << " " << c << " " << h << " " << d << "\n";
    }

    // Generate demands
    // For testId == 7 and 8, bias demands toward long distances
    for (int j = 0; j < M; j++) {
        int a, b;
        if (L == 1) {
            a = 0; b = 1;
        } else if (testId == 7 || testId == 8 || testId == 9) {
            // Prefer long-distance demands
            a = rnd.next(0, L);
            b = rnd.next(0, L);
            int tries = 0;
            while (b == a && tries < 100) { b = rnd.next(0, L); tries++; }
            if (b == a) b = (a == 0) ? 1 : a - 1;
        } else {
            a = rnd.next(0, L);
            b = rnd.next(0, L);
            int tries = 0;
            while (b == a && tries < 100) { b = rnd.next(0, L); tries++; }
            if (b == a) b = (a == 0) ? 1 : a - 1;
        }
        int w = rnd.next(1, 1000000);
        cout << a << " " << b << " " << w << "\n";
    }

    return 0;
}