#ifndef _SCHEDULABLE_H_
#define _SCHEDULABLE_H_

class Schedulable {
    private:
        bool _is_busy;
    public:
        virtual bool init();
        virtual Schedulable* call();
        bool is_busy();
};

#endif