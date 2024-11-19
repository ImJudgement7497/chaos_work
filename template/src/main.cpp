#include <iostream>
#include <vector>
#include "vtimer_t.h"

using namespace std;

int main()
{
    vtimer_t timer;
    timer.start();
    
    cout << "This is a test line" << endl;

    timer.stop();

    cout << timer.elapsed_time() << endl;

    return 0;
}