// 小彭老师作业05：假装是多线程 HTTP 服务器 - 富连网大厂面试官觉得很赞
#include <functional>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <queue>
#include <future>
#include <mutex>
#include <shared_mutex>


struct User {
    std::string password;
    std::string school;
    std::string phone;
};

std::unordered_map<std::string, User> users;
std::unordered_map<std::string, std::chrono::steady_clock::time_point> has_login;  // 换成 std::chrono::seconds 之类的
std::shared_mutex users_mtx, login_mtx;

// 作业要求1：把这些函数变成多线程安全的
// 提示：能正确利用 shared_mutex 加分，用 lock_guard 系列加分
std::string do_register(std::string username, std::string password, std::string school, std::string phone) {
    User user = {password, school, phone};
    std::lock_guard grd(users_mtx);
    if (users.emplace(username, user).second)
        return "注册成功";
    else
        return "用户名已被注册";
}

std::string do_login(std::string username, std::string password) {
    // 作业要求2：把这个登录计时器改成基于 chrono 的
    auto now = std::chrono::steady_clock::now();
    {
        std::unique_lock grd(users_mtx);
        if (has_login.find(username) != has_login.end()) {
            int msec = std::chrono::duration_cast<std::chrono::milliseconds>(now - has_login.at(username)).count();
            return std::to_string(msec) + "毫秒内登录过";
        }
        has_login[username] = now;
    }
    {
        std::shared_lock grd(login_mtx);
        if (users.find(username) == users.end())
            return "用户名错误";
        if (users.at(username).password != password)
            return "密码错误";
    }
    return "登录成功";
}

std::string do_queryuser(std::string username) {
    std::shared_lock grd(users_mtx);
    if (users.find(username) == users.end())
        return "用户名错误";
    auto &user = users.at(username);
    grd.unlock();
    std::stringstream ss;
    ss << "用户名: " << username << std::endl;
    ss << "学校:" << user.school << std::endl;
    ss << "电话: " << user.phone << std::endl;
    return ss.str();
}


class ThreadPool {
private:
    using Task = std::function<void()>;
    constexpr static size_t num = 128;
    std::thread pool[num];
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<Task> tasks;
    std::atomic<bool> stopped;
public:
    void create(const Task &start) {
        // 作业要求3：如何让这个线程保持在后台执行不要退出？
        // 提示：改成 async 和 future 且用法正确也可以加分
        std::lock_guard grd(mtx);
        tasks.push(start);
        cv.notify_one();
    }
    ThreadPool(): stopped(false) {
        for (size_t i = 0; i < num; ++i)
            pool[i] = std::move(std::thread([this]{
                while (!stopped) {
                    Task task;
                    {
                        std::unique_lock grd(mtx);
                        cv.wait(grd, [this]{ return stopped.load() || !tasks.empty(); });
                        if (stopped)
                            return;
                        task = tasks.front();
                        tasks.pop();
                    }
                    task();
                }
            }));
    }
    ~ThreadPool() {
        stopped = true;
        cv.notify_all();
        for (size_t i = 0; i < num; ++i)
            pool[i].join();
    }
};

ThreadPool tpool;


namespace test {  // 测试用例？出水用力！
std::string username[] = {"张心欣", "王鑫磊", "彭于斌", "胡原名"};
std::string password[] = {"hellojob", "anti-job42", "cihou233", "reCihou_!"};
std::string school[] = {"九百八十五大鞋", "浙江大鞋", "剑桥大鞋", "麻绳理工鞋院"};
std::string phone[] = {"110", "119", "120", "12315"};
}

int main() {
    for (int i = 0; i < 262144; i++) {
        tpool.create([&] {
            std::cout << do_register(test::username[rand() % 4], test::password[rand() % 4], test::school[rand() % 4], test::phone[rand() % 4]) << std::endl;
        });
        tpool.create([&] {
            std::cout << do_login(test::username[rand() % 4], test::password[rand() % 4]) << std::endl;
        });
        tpool.create([&] {
            std::cout << do_queryuser(test::username[rand() % 4]) << std::endl;
        });
    }

    // 作业要求4：等待 tpool 中所有线程都结束后再退出
    return 0;
}
