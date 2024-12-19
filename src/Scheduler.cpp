#include <stack>
#include <vector>
#include "Schedulable.h"

class Scheduler {
    std::stack<Schedulable> p_stack;
    std::vector<Schedulable> p_bg_vector;
    void loop() {
        if (!p_stack.empty()) {
            Schedulable* p = &(p_stack.top());
            if (p != nullptr) {
                Schedulable* next_p = p->call();
                if (p == nullptr) p_stack.pop();
                else p_stack.push(*next_p);
            }
        }
    }
};