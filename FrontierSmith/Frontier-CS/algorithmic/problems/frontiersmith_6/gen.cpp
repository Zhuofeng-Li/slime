#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    ensuref(testId >= 1 && testId <= 10, "testId must be between 1 and 10");

    int N, K;
    double P, A, B;
    vector<tuple<double,double,double>> sensors;

    if (testId == 1) {
        // Absolute minimum: 1 sensor, 1 hub
        N = 1; K = 1;
        P = 10.0; A = 1.0; B = 0.5;
        sensors.push_back({0.0, 0.0, 5.0});
    }
    else if (testId == 2) {
        // Two sensors, forced single hub
        N = 2; K = 1;
        P = 100.0; A = 1.0; B = 1.0;
        sensors.push_back({-100.0, 0.0, 3.0});
        sensors.push_back({ 100.0, 0.0, 3.0});
    }
    else if (testId == 3) {
        // Two sensors, two hubs allowed — optimal to split
        N = 2; K = 2;
        P = 1.0; A = 1000.0; B = 1000.0;
        sensors.push_back({-500.0, 0.0, 10.0});
        sensors.push_back({ 500.0, 0.0, 10.0});
    }
    else if (testId == 4) {
        // All sensors at the same point — radius = 0 no matter what
        N = 15; K = 4;
        P = 10.0; A = 1.0; B = 0.5;
        for (int i = 0; i < N; i++) {
            sensors.push_back({42.0, -17.0, (double)(i+1)});
        }
    }
    else if (testId == 5) {
        // Two clearly separated tight clusters
        N = 12; K = 2;
        P = 5.0; A = 2.0; B = 1.0;
        // Cluster A near (0,0)
        for (int i = 0; i < 6; i++) {
            double x = rnd.next(-5.0, 5.0);
            double y = rnd.next(-5.0, 5.0);
            double d = rnd.next(1.0, 10.0);
            sensors.push_back({x, y, d});
        }
        // Cluster B near (1000,1000)
        for (int i = 0; i < 6; i++) {
            double x = 1000.0 + rnd.next(-5.0, 5.0);
            double y = 1000.0 + rnd.next(-5.0, 5.0);
            double d = rnd.next(1.0, 10.0);
            sensors.push_back({x, y, d});
        }
    }
    else if (testId == 6) {
        // Sensors uniformly on x-axis, K groups
        N = 30; K = 5;
        P = 10.0; A = 1.0; B = 2.0;
        for (int i = 0; i < N; i++) {
            double x = (double)i * 10.0;
            double d = rnd.next(1.0, 20.0);
            sensors.push_back({x, 0.0, d});
        }
    }
    else if (testId == 7) {
        // K == N: every sensor can have its own hub
        N = 20; K = 20;
        P = 1.0; A = 100.0; B = 100.0;
        for (int i = 0; i < N; i++) {
            double x = rnd.next(-1000.0, 1000.0);
            double y = rnd.next(-1000.0, 1000.0);
            double d = rnd.next(1.0, 100.0);
            sensors.push_back({x, y, d});
        }
    }
    else if (testId == 8) {
        // K=1 forces all sensors into one hub — load cost dominates
        N = 50; K = 1;
        P = 1.0; A = 1.0; B = 1000.0;
        for (int i = 0; i < N; i++) {
            double x = rnd.next(-500.0, 500.0);
            double y = rnd.next(-500.0, 500.0);
            double d = rnd.next(1.0, 50.0);
            sensors.push_back({x, y, d});
        }
    }
    else if (testId == 9) {
        // Large P: penalizes using many hubs; spread-out sensors
        N = 40; K = 10;
        P = 1000000.0; A = 1.0; B = 0.001;
        for (int i = 0; i < N; i++) {
            double x = rnd.next(-1000000.0, 1000000.0);
            double y = rnd.next(-1000000.0, 1000000.0);
            double d = rnd.next(1.0, 10.0);
            sensors.push_back({x, y, d});
        }
    }
    else {
        // testId == 10
        // Large A: radius cost dominates; highly varied load values
        N = 35; K = 7;
        P = 5.0; A = 1000000.0; B = 0.001;
        // 7 clusters, each slightly spread
        for (int g = 0; g < 7; g++) {
            double cx = (double)g * 300.0;
            double cy = (double)g * 100.0;
            int cnt = N / 7 + (g < N % 7 ? 1 : 0);
            for (int i = 0; i < cnt; i++) {
                double x = cx + rnd.next(-10.0, 10.0);
                double y = cy + rnd.next(-10.0, 10.0);
                double d = rnd.next(1.0, 1000000.0);
                sensors.push_back({x, y, d});
            }
        }
        N = (int)sensors.size();
    }

    cout << N << " " << K << "\n";
    cout << fixed << setprecision(6);
    cout << P << " " << A << " " << B << "\n";
    for (auto& [x, y, d] : sensors) {
        cout << x << " " << y << " " << d << "\n";
    }
    return 0;
}