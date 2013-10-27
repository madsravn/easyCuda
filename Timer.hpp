#include <chrono>


class Timer {
    private:
        std::chrono::high_resolution_clock::time_point t1, t2;
        std::chrono::milliseconds elapsed;
    public:
        Timer() { elapsed = std::chrono::milliseconds(0); }
        void start() { t1 = std::chrono::high_resolution_clock::now(); }
        void stop() { t2 = std::chrono::high_resolution_clock::now(); elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1); };
        std::chrono::milliseconds duration() { return elapsed; /*std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);*/ }
        void reset() { elapsed = std::chrono::milliseconds(0); }

};

