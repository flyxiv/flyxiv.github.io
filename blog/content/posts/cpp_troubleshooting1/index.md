---
title: <C++ Server Programming> Troubleshooting 1. Fixing Thread Logic
date: 2025-04-28T19:47:20Z
lastmod: 2025-03-30T19:47:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: cpp.png
categories:
  - computer vision
tags:
  - c++
  - cpp
  - network programming
  - threads
  - multithreading
# nolastmod: true
draft: false
---

# Situation

I was writing a unit test for my GlobalThreadManager Singleton Class:

```cpp
TEST_F(ThreadManagerTest, ThreadIdGeneration) {
    std::vector<std::thread> jobs;
    std::atomic<int> completion_counter{ 0 };

	// Given: 10 threads are generated asynchronously by 10 different threads using GlobalThreadManager, total 100 threads
    for (int i = 0; i < 10; i++) {
        jobs.push_back(std::thread([this, &completion_counter]() {
            for (int j = 0; j < 10; j++) {
                GlobalThreadManager->Launch([this, &completion_counter]() {
                    std::lock_guard<std::mutex> lock(thread_id_mutex);
                    thread_ids.insert(CurrentThreadId);
                    completion_counter++;
                });
            }
        }));
    }

    for (auto& job : jobs) {
        if (job.joinable()) {
            job.join();
        }
    }

    // job is registered by join, but not guaranteed to complete.
    // need to wait until all jobs are done.
    while (completion_counter < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

	// Then: Number of threads in the GlobalThreadManager thread pool should be 100, and they must all have unique ids.
    ASSERT_EQ(GlobalThreadManager->GetThreadPool().size(), 100);
    ASSERT_EQ(thread_ids.size(), 100);
}
```

After running the test, I figured out that **all the assert stations pass, but the test itself terminates with error code 3.**

# Reason For Bug

1. By waiting until completion_counter = 100, we guarantee that **the 100 threads "Launch"ed by GlobalThreadManager has incremented their completion counter.**
2. However, this does not guarantee that **All the threads "terminated" - it just means that all 100 threads executed its final statement and is ready to terminate.**
3. The join() statement guarantees that the **10 subthreads in the "jobs" vector terminated, but it doesn't guarantee the termination of its 10 subthreads.**

# Fix

Add `GlobalThreadManager.Join()` at the end of the test to guarantee all threads are terminated before Destructor is called.

```cpp
TEST_F(ThreadManagerTest, ThreadIdGeneration) {
    std::vector<std::thread> jobs;
    std::atomic<int> completion_counter{ 0 };

	// Given: 10 threads are generated asynchronously by 10 different threads using GlobalThreadManager, total 100 threads
    for (int i = 0; i < 10; i++) {
        jobs.push_back(std::thread([this, &completion_counter]() {
            for (int j = 0; j < 10; j++) {
                GlobalThreadManager->Launch([this, &completion_counter]() {
                    std::lock_guard<std::mutex> lock(thread_id_mutex);
                    thread_ids.insert(CurrentThreadId);
                    completion_counter++;
                });
            }
        }));
    }

    for (auto& job : jobs) {
        if (job.joinable()) {
            job.join();
        }
    }

    // job is registered by join, but not guaranteed to complete.
    // need to wait until all jobs are done.
    while (completion_counter < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

	// Then: Number of threads in the GlobalThreadManager thread pool should be 100, and they must all have unique ids.
    ASSERT_EQ(GlobalThreadManager->GetThreadPool().size(), 100);
    ASSERT_EQ(thread_ids.size(), 100);

	// add join!!!
	GlobalThreadManager->Join();
}
```
