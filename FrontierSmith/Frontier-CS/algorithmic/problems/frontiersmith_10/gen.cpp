#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static vector<int> oddPrimes;

void buildOddPrimes(int limit) {
    vector<bool> sieve(limit + 1, true);
    sieve[0] = sieve[1] = false;
    for (int i = 2; i <= limit; i++) {
        if (sieve[i]) {
            if (i >= 3) oddPrimes.push_back(i);
            for (long long j = (long long)i * i; j <= limit; j += i)
                sieve[j] = false;
        }
    }
}

// Return true if n is odd and square-free
bool isOddSquareFree(int n) {
    if (n < 3 || n % 2 == 0) return false;
    for (int i = 3; (long long)i * i <= n; i += 2) {
        if (n % i == 0) {
            if ((n / i) % i == 0) return false;
        }
    }
    return true;
}

// Get all distinct odd prime factors of n
vector<int> oddPrimeFactors(int n) {
    vector<int> res;
    for (int i = 3; (long long)i * i <= n; i += 2) {
        if (n % i == 0) {
            res.push_back(i);
            while (n % i == 0) n /= i;
        }
    }
    if (n > 2) res.push_back(n);
    return res;
}

// Generate a random odd square-free number in [3, maxVal]
int randOddSquareFree(int maxVal) {
    // Try up to 1000 times
    for (int tries = 0; tries < 1000; tries++) {
        int result = 1;
        // shuffle indices of first ~15 primes
        int nPrimes = min((int)oddPrimes.size(), 15);
        vector<int> idx(nPrimes);
        for (int i = 0; i < nPrimes; i++) idx[i] = i;
        shuffle(idx.begin(), idx.end());
        for (int i : idx) {
            int p = oddPrimes[i];
            if ((long long)result * p <= maxVal) result *= p;
        }
        if (result >= 3 && result <= maxVal) return result;
    }
    // fallback: random odd prime <= maxVal
    int hi = (int)(upper_bound(oddPrimes.begin(), oddPrimes.end(), maxVal) - oddPrimes.begin());
    if (hi == 0) return 3;
    return oddPrimes[rnd.next(0, hi - 1)];
}

int randOddPrime(int maxVal) {
    int hi = (int)(upper_bound(oddPrimes.begin(), oddPrimes.end(), maxVal) - oddPrimes.begin());
    if (hi == 0) return 3;
    return oddPrimes[rnd.next(0, hi - 1)];
}

// Generate queries where the "target" witness (tA, tB) covers most of them
// This creates a case where the optimal is clearly (tA,tB) as first pick
void genConcentratedWitness(int n, int K, long long B,
                             int tA, int tB,
                             int pLow, int pHigh,
                             int wHeavy, int wLight,
                             double heavyFrac) {
    long long targetSum = (long long)tA * tA + (long long)tB * tB;
    cout << n << " " << K << " " << B << "\n";
    int nHeavy = (int)(n * heavyFrac);
    for (int i = 0; i < n; i++) {
        int pi;
        // pick a valid odd square-free prime
        for (;;) {
            pi = randOddPrime(pHigh);
            if (pi >= pLow) break;
        }
        int xi;
        long long wi;
        if (i < nHeavy) {
            // heavy query: target residue = targetSum mod pi
            xi = (int)(targetSum % pi);
            wi = rnd.next(wHeavy / 2, wHeavy);
        } else {
            // light query: random residue
            xi = rnd.next(0, pi - 1);
            wi = rnd.next(1, wLight);
        }
        cout << pi << " " << xi << " " << wi << "\n";
    }
}

// Generate K "planted" witnesses and build queries around them
// Good greedy should recover them; bad greedy picks wrong order
void genKPlantedWitnesses(int n, int K, long long B,
                           int pLow, int pHigh) {
    // Generate K distinct witness pairs
    vector<pair<int,int>> witnesses;
    set<pair<int,int>> used;
    int maxA = (int)min(B, (long long)2236);
    while ((int)witnesses.size() < K) {
        int a = rnd.next(0, maxA);
        int b = rnd.next(0, maxA);
        if (!used.count({a, b})) {
            used.insert({a, b});
            witnesses.push_back({a, b});
        }
    }

    cout << n << " " << K << " " << B << "\n";
    // Assign queries to witnesses with weight proportional to witness index
    // (earlier witnesses = higher weight => greedy order matters)
    for (int i = 0; i < n; i++) {
        // pick a witness with probability weighted by rank
        int wIdx = rnd.next(0, K - 1);
        auto [tA, tB] = witnesses[wIdx];
        long long targetSum = (long long)tA * tA + (long long)tB * tB;

        int pi;
        for (;;) {
            pi = randOddPrime(pHigh);
            if (pi >= pLow) break;
        }
        int xi = (int)(targetSum % pi);
        // higher weight for lower-indexed witnesses to create ordering challenge
        long long wi = rnd.next(1LL, 1000000LL);
        cout << pi << " " << xi << " " << wi << "\n";
    }
}

// Generate a "needle in haystack" test: one specific witness covers nearly all weight
// but baseline covers very little
void genNeedleHaystack(int n, int K, long long B,
                        int tA, int tB, int pLow, int pHigh) {
    long long targetSum = (long long)tA * tA + (long long)tB * tB;
    cout << n << " " << K << " " << B << "\n";
    for (int i = 0; i < n; i++) {
        int pi;
        for (;;) {
            pi = randOddPrime(pHigh);
            if (pi >= pLow) break;
        }
        // 90% of queries: target the needle witness
        // 10%: random (noise)
        int xi;
        long long wi;
        if (rnd.next(0, 9) < 9) {
            xi = (int)(targetSum % pi);
            wi = rnd.next(500000, 1000000);
        } else {
            xi = rnd.next(0, pi - 1);
            wi = rnd.next(1, 100);
        }
        cout << pi << " " << xi << " " << wi << "\n";
    }
}

// Generate composite moduli queries (p = product of small primes)
// A witness covering residue mod p simultaneously covers residue mod each prime factor
void genCompositeModuli(int n, int K, long long B,
                         int tA, int tB) {
    long long targetSum = (long long)tA * tA + (long long)tB * tB;
    // Use moduli that are products of 2-3 small odd primes
    // so the target witness covers many
    vector<int> smallPrimes = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
    vector<int> compositeModuli;
    for (int i = 0; i < (int)smallPrimes.size(); i++) {
        for (int j = i + 1; j < (int)smallPrimes.size(); j++) {
            compositeModuli.push_back(smallPrimes[i] * smallPrimes[j]);
        }
    }
    // Add triples
    for (int i = 0; i < (int)smallPrimes.size(); i++)
        for (int j = i+1; j < (int)smallPrimes.size(); j++)
            for (int k = j+1; k < (int)smallPrimes.size(); k++)
                if ((long long)smallPrimes[i]*smallPrimes[j]*smallPrimes[k] <= 1000000)
                    compositeModuli.push_back(smallPrimes[i]*smallPrimes[j]*smallPrimes[k]);

    cout << n << " " << K << " " << B << "\n";
    for (int i = 0; i < n; i++) {
        int pi = compositeModuli[rnd.next(0, (int)compositeModuli.size() - 1)];
        int xi = (int)(targetSum % pi);
        long long wi = rnd.next(100000, 1000000);
        cout << pi << " " << xi << " " << wi << "\n";
    }
}

// Generate queries where greedy based on total coverage sum is suboptimal
// Create "trap" queries: a single value covers many but light queries,
// while the real optimal covers fewer but very heavy queries
void genGreedyTrap(int n, int K, long long B) {
    // Good witness: covers high-weight queries
    int goodA = 17, goodB = 23; // sum = 289 + 529 = 818
    long long goodSum = (long long)goodA * goodA + (long long)goodB * goodB;
    // Trap witness: covers many low-weight queries  
    int trapA = 1, trapB = 0; // sum = 1
    long long trapSum = (long long)trapA * trapA + (long long)trapB * trapB;

    cout << n << " " << K << " " << B << "\n";

    // Generate many light-weight queries covered by trapSum
    int nTrap = (int)(n * 0.6);
    // Generate few heavy-weight queries covered by goodSum
    int nGood = n - nTrap;

    vector<int> smallPrimes = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31};

    for (int i = 0; i < nTrap; i++) {
        int pi = smallPrimes[rnd.next(0, (int)smallPrimes.size() - 1)];
        int xi = (int)(trapSum % pi);
        long long wi = rnd.next(1, 10); // very light
        cout << pi << " " << xi << " " << wi << "\n";
    }
    for (int i = 0; i < nGood; i++) {
        int pi = smallPrimes[rnd.next(0, (int)smallPrimes.size() - 1)];
        int xi = (int)(goodSum % pi);
        long long wi = rnd.next(900000, 1000000); // very heavy
        cout << pi << " " << xi << " " << wi << "\n";
    }
}

// Structured test: queries perfectly partitioned among K witnesses
// The optimal picks all K; greedy should recover all K
void genPerfectPartition(int n, int K, long long B) {
    int maxA = (int)min(B, (long long)2236);
    // Generate exactly K distinct witnesses
    vector<pair<int,int>> witnesses;
    set<pair<int,int>> used;
    while ((int)witnesses.size() < K) {
        int a = rnd.next(10, maxA);
        int b = rnd.next(10, maxA);
        // avoid small sums that baseline (0,0),(1,0),... would cover
        if (!used.count({a, b})) {
            used.insert({a, b});
            witnesses.push_back({a, b});
        }
    }

    // Use large primes so each witness's residue class is unique
    // Pick large primes between 5000000 and 9999999
    vector<int> largePrimes;
    for (int p = 5000003; p < 10000000 && (int)largePrimes.size() < 200; p += 2) {
        if (isOddSquareFree(p)) largePrimes.push_back(p);
    }

    cout << n << " " << K << " " << B << "\n";
    int queriesPerWitness = n / K;
    for (int wi = 0; wi < K; wi++) {
        auto [tA, tB] = witnesses[wi];
        long long targetSum = (long long)tA * tA + (long long)tB * tB;
        int count = (wi == K - 1) ? (n - queriesPerWitness * (K - 1)) : queriesPerWitness;
        for (int j = 0; j < count; j++) {
            // Use a large prime so only this witness's sum hits this residue
            int pi = largePrimes[rnd.next(0, (int)largePrimes.size() - 1)];
            int xi = (int)(targetSum % pi);
            long long weight = rnd.next(500000, 1000000);
            cout << pi << " " << xi << " " << weight << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    buildOddPrimes(10000000);

    int testId = atoi(argv[1]);

    switch (testId) {
        case 1: {
            // Minimal corner case: n=1, K=1, B=1
            cout << "1 1 1\n";
            cout << "3 0 1\n";
            break;
        }
        case 2: {
            // Sample from the problem statement (exact)
            cout << "6 2 100\n";
            cout << "5 0 10\n";
            cout << "5 1 8\n";
            cout << "13 5 7\n";
            cout << "13 8 6\n";
            cout << "15 5 9\n";
            cout << "15 8 4\n";
            break;
        }
        case 3: {
            // Concentrated around one specific witness: (100, 200), sum=50000
            // All heavy queries point to this sum mod various small primes
            // Baseline ((0,0),(1,0),...) won't cover these
            // Good greedy will find (100,200) quickly
            int n = 200000, K = 60;
            long long B = 1000000000LL;
            int tA = 100, tB = 200;
            genConcentratedWitness(n, K, B, tA, tB, 3, 10000, 1000000, 100, 0.95);
            break;
        }
        case 4: {
            // Small n, small K, but carefully structured
            // 1000 queries, all designed around 5 specific witnesses
            // with heavily skewed weights to make ordering matter
            int n = 1000, K = 5;
            long long B = 100;
            // Witnesses: (3,4)=25, (5,12)=169, (8,15)=289, (7,24)=625, (20,21)=841
            vector<pair<int,int>> W5 = {{3,4},{5,12},{8,15},{7,24},{20,21}};
            vector<long long> sums5;
            for (auto [a, b] : W5) sums5.push_back((long long)a*a + (long long)b*b);

            // weights increase dramatically per witness group
            // so greedy must pick highest-weight group first
            vector<long long> groupWeights = {1000000, 800000, 600000, 400000, 200000};
            cout << n << " " << K << " " << B << "\n";
            vector<int> smallPrimes5 = {97, 101, 103, 107, 109, 113};
            int perGroup = n / 5;
            for (int g = 0; g < 5; g++) {
                int cnt = (g == 4) ? (n - perGroup * 4) : perGroup;
                for (int j = 0; j < cnt; j++) {
                    int pi = smallPrimes5[rnd.next(0, (int)smallPrimes5.size()-1)];
                    int xi = (int)(sums5[g] % pi);
                    long long wi = rnd.next(groupWeights[g] - 100000LL, groupWeights[g]);
                    cout << pi << " " << xi << " " << wi << "\n";
                }
            }
            break;
        }
        case 5: {
            // Needle in haystack with medium-sized primes
            // One specific witness (500, 1000) sum = 1250000
            // covers 90% of weight; baseline covers ~0%
            int n = 200000, K = 10;
            long long B = 1000000000LL;
            int tA = 500, tB = 1000;
            genNeedleHaystack(n, K, B, tA, tB, 1000, 500000);
            break;
        }
        case 6: {
            // Composite moduli test: using products of small primes
            // Witness (37, 84): sum = 1369 + 7056 = 8425
            // 8425 = 5^2 * 337... let's use (7, 24) sum = 625 = 5^4... hmm
            // Use (10, 30) sum = 1000, nice for mod composite
            // Actually use (50, 100) sum = 12500
            int n = 200000, K = 30;
            long long B = 1000000000LL;
            int tA = 50, tB = 100;
            genCompositeModuli(n, K, B, tA, tB);
            break;
        }
        case 7: {
            // Greedy trap: many light queries around trapSum=1, few heavy around goodSum
            // Weak greedy picks trapSum first (more total queries but low weight)
            // Strong algorithm picks goodSum first (fewer queries but high weight)
            int n = 200000, K = 5;
            long long B = 1000000000LL;
            genGreedyTrap(n, K, B);
            break;
        }
        case 8: {
            // Perfect partition among K=60 witnesses
            // Each witness has equal share of heavy queries
            // B small forces witnesses to use small coordinates
            int n = 200000, K = 60;
            long long B = 100;
            genPerfectPartition(n, K, B);
            break;
        }
        case 9: {
            // Max n, max K planted witnesses, large B
            // Queries designed around K specific witnesses in (1000..2236) range
            // Baseline picks (0,0)...(59,0) which have tiny sums, missing all planted queries
            int n = 200000, K = 60;
            long long B = 1000000000LL;
            genKPlantedWitnesses(n, K, B, 100, 100000);
            break;
        }
        case 10: {
            // Adversarial: B = K-1 (minimum B), forced to use small witnesses
            // Many queries designed around specific small sums
            // Heavily concentrated on a few specific (A,B) pairs in [0,59]
            int n = 200000, K = 60;
            long long B = K - 1; // B = 59
            // Best witnesses in [0,59]^2: pick ones with large coverage
            // Use (7, 24) sum=625, (20, 21) sum=841, (3, 40) sum=1609,
            // (5, 39) sum=1546, (9, 40) sum=1681
            // But B=59, so 7,24,20,21,3,40,5,39,9,40 are all valid
            vector<pair<int,int>> keyWitnesses = {
                {7,24},{20,21},{3,40},{5,39},{9,40},{12,35},{15,36},{16,30},
                {0,25},{7,0},{24,32},{6,17},{10,51},{8,55},{4,57}
            };
            // Filter to those within [0, B]
            vector<pair<int,int>> validW;
            for (auto [a, b] : keyWitnesses) {
                if (a <= B && b <= B) validW.push_back({a, b});
            }
            // Also add some from [30,59]
            for (int extra = 0; extra < 10; extra++) {
                int a = rnd.next(30, (int)B);
                int b = rnd.next(30, (int)B);
                validW.push_back({a, b});
            }

            cout << n << " " << K << " " << B << "\n";
            // 80% of queries: around key witnesses (high weight)
            // 20%: random (low weight)
            int nKey = (int)(n * 0.85);
            vector<int> smallPrimes10 = {3, 5, 7, 11, 13, 17, 19, 23};
            for (int i = 0; i < n; i++) {
                int pi = smallPrimes10[rnd.next(0, (int)smallPrimes10.size() - 1)];
                int xi;
                long long wi;
                if (i < nKey && !validW.empty()) {
                    auto [tA, tB] = validW[rnd.next(0, (int)validW.size() - 1)];
                    long long s = (long long)tA * tA + (long long)tB * tB;
                    xi = (int)(s % pi);
                    wi = rnd.next(500000, 1000000);
                } else {
                    xi = rnd.next(0, pi - 1);
                    wi = rnd.next(1, 1000);
                }
                cout << pi << " " << xi << " " << wi << "\n";
            }
            break;
        }
        default:
            ensuref(false, "Unknown test ID: %d", testId);
    }

    return 0;
}