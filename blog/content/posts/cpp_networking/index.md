---
title: <Server> Understanding Game Servers(C++)
date: 2025-04-14T18:33:20Z
lastmod: 2025-04-14T18:33:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: cpp.png
categories:
  - networking
tags:
  - cpp
  - network programming
  - game server
  - tcp
  - udp
# nolastmod: true
draft: false
---

# Web Server vs Game Server

Web Server:

- small # of requests per user
- don't need real-time interaction
- Server doesn't approach client first
- Stateless - forgets user after request is done
- Usually uses web frameworks
- ex) drivethrough

Game Server:

- a lot of request/response
- real-time interaction
- stateful
- cannot use web framework (different needs for each game)
- ex) restaurant - constant service to the user

# C++ Thread

`#include <thread>`

```cpp
#include <thread>

int main()
{
	std::thread t;

	// CPU core number
	int t.hardware_concurrency();

	// thread id
	auto id = t.get_id();

	// detach the actual thread from std::thread, make it run independently as background thread
	t.detach();

	// check if the thread is valid and can join
	if (t.joinable()) {
		t.join();
	}
}
```

# Atomic Type

all-or-nothing

- ## lock behavior for simple types

```cpp
#include <atomic>

atomic<int> sum = 0;
```

- stl containers are probably not multi-thread safe.

# Mutex

Basic C++ lock

```cpp
#include <vector>
#include <atomic>
#include <mutex>

mutex m;
vector<int32> v;

void Push()
{
    for (int32 i = 0; i < 10'000; i++) {
        m.lock();
        v.push_back(i);

        m.unlock();
    }
}

int main()
{
    auto t1 = std::thread(Push);
    auto t2 = std::thread(Push);

    t1.join();
    t2.join();
    cout << v.size() << endl;
}
```

if there's unusual jump before m.unlock(), deadlock happens

- we can't be checking every branch and unlocking; too error-prone.

## Types of Lock

- Spin Lock(Infinite wait)
- Sleep
- Event based lock

# Lock Guard - RAII Pattern

```cpp

// Wrapper class to automate lock acquisition/release
template<typename T>
class LockGuard
{
public:
	LockGuard(T& m)
	{
		_mutex = m;
		_mutex->lock();
	}

	~LockGuard()
	{
		_mutex->unlock();
	}

private:
	T* _mutex;
};


#include <vector>
#include <atomic>
#include <mutex>

mutex m;
vector<int32> v;

void Push()
{
    for (int32 i = 0; i < 10'000; i++) {
		// acquires lock on initialization
		auto lock_guard = LockGuard(m);

		// std C++ version. Same function as above.
		std::lock_guard<std::mutex> lock_guard(m);

		// lock_guard + doesn't lock instantly
		std::unique_lock<std::mutex> lock_guard_unique(m, std::defer_lock);
		lock_guard_unique.lock();


        v.push_back(i);

		// lock_guard is destroyed, automatically releasing lock(even if there's goto)
    }
}


```

# SpinLock

- voilatile: don't do compile optimization.

```cpp
// in release, runs int a = 4 directly!
int a = 0;
a = 1;
a = 2;
a = 3;
a = 4;

// doesn't cut optimizations and run every line
volatile int a = 0;
a = 1;
a = 2;
a = 3;
a = 4;
```

Needs to be used for lock

```cpp
class SpinLock {
public:
    void lock()
    {
        // Compare and swap
        bool expected = false;
        bool desired = true;

        while (_locked.compare_exchange_strong(expected, desired) == false) {
            expected = false;
        }
    }

    void unlock()
    {
        _locked.store(false);
    }

private:
    atomic<bool> _locked = false;
};
```

# SleepLock

```cpp
class SleepLock {
public:
    void lock()
    {
        // Compare and swap
        bool expected = false;
        bool desired = true;

        while (_locked.compare_exchange_strong(expected, desired) == false) {
            expected = false;

			this_thread::sleep_for(0ms);
        }
    }

    void unlock()
    {
        _locked.store(false);
    }

private:
    atomic<bool> _locked = false;
};
```

# EventLock

Sleep until event occurs. Use window/linux system call

```cpp
// GameServer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "AccountManager.h"
#include "UserManager.h"

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <windows.h>

mutex m;
queue<int32> q;
HANDLE handle;

void Producer()
{
    while (true)
    {
        unique_lock<mutex> lock(m);
        q.push(100);
    }

    // turns signal on
    ::SetEvent(handle);

    this_thread::sleep_for(100ms);
}

void Consumer()
{
    while (true)
    {
        ::WaitForSingleObject(handle, INFINITE);

        unique_lock<mutex> lock(m);

        if (q.empty() == false)
        {
            int32 data = q.front();
            q.pop();
            cout << data << endl;
        }
    }

}

int main()
{
    // Kernel Object
    // Usage Count, Signal on/off(bool), Auto/Manual(bool)
    handle = ::CreateEvent(NULL, FALSE, FALSE, NULL);
    std::thread t1(Producer);
    std::thread t2(Consumer);

    t1.join();
    t2.join();

    ::CloseHandle(handle);
}
```

# Condition Variable

The event above doesn't actually give 1 to 1 mapping of Producer and Consumer:

- Another Producer thread can be called right after SetEvent before any Consumer is called

To guarantee an ordered approach, we need to use `condition variable`

- wakes up waiting thread on condition
- acquires lock if condition is met

```cpp
// GameServer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "AccountManager.h"
#include "UserManager.h"

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <windows.h>

mutex m;
queue<int32> q;

condition_variable cv;

void Producer()
{
    while (true)
    {
        {
			unique_lock<mutex> lock(m);
			q.push(100);
        }
    }

    // wake only one waiting thread
    cv.notify_one();

    this_thread::sleep_for(100ms);
}

void Consumer()
{
    while (true)
    {
        unique_lock<mutex> lock(m);

        // why check empty()
        // Spurious Wakeup
        // We don't have the lock when we're waiting, so another consumer could use the queue simultaneously with notify_one.
        cv.wait(lock, []() { return q.empty() == false; });

        if (q.empty() == false)
        {
            int32 data = q.front();
            q.pop();
            cout << data << endl;
        }
    }

}

int main()
{
    // Kernel Object
    // Usage Count, Signal on/off(bool), Auto/Manual(bool)
    std::thread t1(Producer);
    std::thread t2(Consumer);

    t1.join();
    t2.join();

}
```

# One-time only tasks

We can use `future/promise/packaged_task`

- future async: simplest
- promise: more control over start timing, set_exception
- packaged_task: used for managing big tasks that require multiple thread or thread pool.

```cpp
// GameServer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "AccountManager.h"
#include "UserManager.h"

#include <iostream>
#include <future>


int64 Calculate()
{
	int64 sum = 0;

	for (int32 i = 0; i < 100'00; i++)
		sum += i;

	return sum;
}

void PromiseWorker(std::promise<string>&& promise)
{
	promise.set_value("Secret message");
}


void TaskWorker(std::packaged_task < int64(void)> && task)
{
	task();
}

int main()
{
	// std::future: midpoint btw thread and synchronous
	{
		// 1) deferred -> lazy evaluation in current thread
		// 2) async -> create separate thread and run
		// 3) deferred | async: choose any of the two

		std::future<int64> future = std::async(std::launch::async, Calculate);

		future.wait_for(0ms);
		int64 result = future.get();
	}

	// std::promise
	// more control over exactly when to start thread, can deliver exceptions
	{
		std::promise<string> promise;
		std::future<string> future = promise.get_future();

		thread t(PromiseWorker, std::move(promise));

		string message = future.get();
		cout << message << endl;

		t.join();
	}

	// std::packaged_task
	{
		std::packaged_task<int64(void)> task(Calculate);
		std::future<int64> future = task.get_future();

		std::thread t(TaskWorker, std::move(task));
		int64 sum = future.get();
		cout << sum << endl;

		t.join();
	}
}
```
