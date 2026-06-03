#pragma once

// A single background thread that runs submitted tasks in order — the C++
// stand-in for Python's ThreadPoolExecutor(max_workers=1). Used for off-thread
// mesh parsing and frame array building so loads queue instead of fighting over
// the CPU, and the result is uploaded to the GPU later on the main thread.

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

class WorkerQueue {
public:
    WorkerQueue() : worker_([this] { run(); }) {}

    ~WorkerQueue() { shutdown(); }

    WorkerQueue(const WorkerQueue&) = delete;
    WorkerQueue& operator=(const WorkerQueue&) = delete;

    /// Queue ``task`` to run on the worker thread.
    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (stop_) {
                return;
            }
            tasks_.push(std::move(task));
        }
        condition_.notify_one();
    }

    /// Stop accepting tasks and join the worker (also called by the destructor).
    void shutdown() {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (stop_) {
                return;
            }
            stop_ = true;
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> guard(mutex_);
                condition_.wait(guard, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<std::function<void()>> tasks_;
    bool stop_ = false;
    std::thread worker_;
};
