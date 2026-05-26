#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

string genString(int len, const string& alphabet) {
    string res;
    res.reserve(len);
    for (int i = 0; i < len; i++) {
        res += alphabet[rnd.next(0, (int)alphabet.size() - 1)];
    }
    return res;
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n, q;
    vector<string> names;
    vector<tuple<int,int,int>> queries;

    auto addQueries = [&](int cnt) {
        for (int j = 0; j < cnt; j++) {
            int l = rnd.next(1, n - 1);
            int r = rnd.next(l + 1, n);
            int k = rnd.next(1, n);
            queries.push_back({l, r, k});
        }
    };

    if (testId == 1) {
        // Minimal: n=2, q=1
        n = 2; q = 1;
        names = {"a", "b"};
        queries.push_back({1, 2, 1});

    } else if (testId == 2) {
        // All single-char same name 'a', small n/q
        n = 10; q = 50;
        for (int i = 0; i < n; i++) names.push_back("a");
        addQueries(q);

    } else if (testId == 3) {
        // Hand-crafted cross-boundary matching potential
        n = 6; q = 30;
        names = {"ab", "ba", "aba", "bab", "a", "b"};
        addQueries(q);

    } else if (testId == 4) {
        // Large n and q, small alphabet (2 chars), single-char names
        n = 2000; q = 200000;
        for (int i = 0; i < n; i++)
            names.push_back(string(1, (char)('a' + rnd.next(0, 1))));
        addQueries(q);

    } else if (testId == 5) {
        // Large n, long names (up to 40), binary alphabet -> many cross-boundary hits
        n = 2000; q = 200000;
        // 2000 * 40 = 80000 <= 100000 sum constraint
        for (int i = 0; i < n; i++) {
            int len = rnd.next(1, 40);
            names.push_back(genString(len, "ab"));
        }
        addQueries(q);

    } else if (testId == 6) {
        // Highly overlapping patterns: "aa*" family
        n = 50; q = 5000;
        string pool[] = {"a", "aa", "aaa", "aaaa", "aaaaa"};
        for (int i = 0; i < n; i++)
            names.push_back(pool[rnd.next(0, 4)]);
        addQueries(q);

    } else if (testId == 7) {
        // All same long string, large n
        n = 500; q = 100000;
        string pat = genString(40, "ab");
        for (int i = 0; i < n; i++) names.push_back(pat);
        addQueries(q);

    } else if (testId == 8) {
        // Small n, large q: stress on query accumulation
        n = 5; q = 200000;
        for (int i = 0; i < n; i++) {
            int len = rnd.next(1, 10);
            names.push_back(genString(len, "abc"));
        }
        addQueries(q);

    } else if (testId == 9) {
        // Moderate n, diverse lengths, 5-letter alphabet
        n = 500; q = 50000;
        for (int i = 0; i < n; i++) {
            int len = rnd.next(1, 40);
            names.push_back(genString(len, "abcde"));
        }
        addQueries(q);

    } else {
        // testId == 10
        // Max n, max q, full alphabet, mixed lengths
        n = 2000; q = 200000;
        int totalLen = 0;
        for (int i = 0; i < n; i++) {
            int remaining = n - i;
            int maxLen = min(40, (100000 - totalLen) - (remaining - 1));
            if (maxLen < 1) maxLen = 1;
            int len = rnd.next(1, maxLen);
            totalLen += len;
            names.push_back(genString(len, "abcdefghijklmnopqrstuvwxyz"));
        }
        addQueries(q);
    }

    // Output
    cout << n << " " << q << "\n";
    for (int i = 0; i < n; i++) cout << names[i] << "\n";
    for (int j = 0; j < (int)queries.size(); j++) {
        int l = get<0>(queries[j]);
        int r = get<1>(queries[j]);
        int k = get<2>(queries[j]);
        cout << l << " " << r << " " << k << "\n";
    }

    return 0;
}