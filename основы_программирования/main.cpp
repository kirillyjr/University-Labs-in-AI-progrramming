
//Лабораторная работа 1
//Задание 1.1 Вычисление математического выражения. Вариант 4.

#include <iostream>
#include <cmath>
#include <iomanip>
#define _USE_MATH_DEFINES
using namespace std;
int main() {
    double a, b, t;
    cout << "a = "; cin >> a;//Вводим данные(вещественные числа)
    cout << "b = "; cin >> b;
    cout << "t = "; cin >> t;
    double y = exp(-b*t)  * sin(a*t+b) - sqrt(abs(b*t +a));//вычисляю y
    double w = b * sin(a*pow(t,2) * cos(2*t)) - 1;//вычисляю w
    cout << fixed << setprecision(2) << "y = " << y << endl  << "w = " << w << endl;
    return 0;
}