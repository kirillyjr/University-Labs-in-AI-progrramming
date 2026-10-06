#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <queue>

#define sp " "
#define elif else if
#define rep(s, n, h) for(long long i = s; i < n; i+=h)

using namespace std;
using ll = long long;
using vll = vector<long long>;
using str = string;
using ch = char;



int main(){
    ll t;
    cin >> t;
    while(t--){
        ll n, k;
        cin >> n >> k;
        string l;
        cin >> l;

        ll cur = 0, ans = 0;

        rep(0, k, 1){
        if (l[i] == 'W') cur++;
        }
            
        ans = cur;

        rep(k, n, 1){
            if (l[i] == 'W') cur++;
            if (l[i-k] == 'W') cur--;
            ans = min(ans, cur);
        }

        cout << ans << endl;
        
        

    }
}