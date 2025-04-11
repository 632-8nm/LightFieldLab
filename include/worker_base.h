#ifndef WORKER_BASE_H
#define WORKER_BASE_H

#include <QtCore/qobject.h>
#include <QtCore/qtmetamacros.h>

#include <QFuture>
#include <QObject>
#include <QPromise>
#include <QThread>
#include <iostream>

template <typename Core>
class WorkerBase : public QObject {
   public:
	explicit WorkerBase(QObject* parent = nullptr)
		: QObject(parent), core(std::make_unique<Core>()) {}
	~WorkerBase() {}

	// private:
	// template <typename Func, typename... Args>
	// void invoke(Func&& func, Args&&... args) {
	// 	QMetaObject::invokeMethod(
	// 		this, // 调用 Worker 的匿名槽函数
	// 		[this, func = std::forward<Func>(func),
	// 		 args = std::tuple(std::forward<Args>(args)...)]() mutable {
	// 			// 使用 core.get() 绑定 Core 的 this 指针
	// 			std::apply(
	// 				[&](auto&&... params) {
	// 					(core.get()->*func)(
	// 						std::forward<decltype(params)>(params)...);
	// 				},
	// 				std::move(args));
	// 		},
	// 		Qt::QueuedConnection);
	// }

	// std::unique_ptr<Core> core;
	// template <typename Func, typename... Args>
	// auto invoke(Func&& func, Args&&... args) -> std::future<
	// 	decltype((core.get()->*func)(std::forward<Args>(args)...))> {
	// 	using ResultType =
	// 		decltype((core.get()->*func)(std::forward<Args>(args)...));
	// 	// 创建 promise/future 对象
	// 	auto promise = std::make_shared<std::promise<ResultType>>();
	// 	auto future	 = promise->get_future();

	// 	QMetaObject::invokeMethod(
	// 		this,
	// 		[this, func = std::forward<Func>(func),
	// 		 args	 = std::tuple(std::forward<Args>(args)...),
	// 		 promise = std::move(promise)]() mutable {
	// 			// 使用 std::apply 调用成员函数
	// 			if constexpr (std::is_void_v<ResultType>) {
	// 				std::cout << "Function “invoke” called thread: "
	// 						  << QThread::currentThreadId() << std::endl;
	// 				// 返回类型为 void，无需存储结果
	// 				std::apply(
	// 					[&](auto&&... params) {
	// 						(core.get()->*func)(
	// 							std::forward<decltype(params)>(params)...);
	// 					},
	// 					std::move(args));
	// 				// 设置 promise 的值（无需参数）
	// 				promise->set_value();

	// 			} else {
	// 				std::cout << "Function “invoke” called thread: "
	// 						  << QThread::currentThreadId() << std::endl;
	// 				// 返回类型非 void，存储结果
	// 				ResultType result = std::apply(
	// 					[&](auto&&... params) {
	// 						return (core.get()->*func)(
	// 							std::forward<decltype(params)>(params)...);
	// 					},
	// 					std::move(args));
	// 				promise->set_value(std::move(result));
	// 			}
	// 		},
	// 		Qt::QueuedConnection);
	// 	return future;
	// }

	std::unique_ptr<Core> core;
	template <typename Func, typename... Args>
	auto invoke(Func&& func, Args&&... args)
		-> QFuture<decltype((core.get()->*func)(std::forward<Args>(args)...))> {
		using ResultType =
			decltype((core.get()->*func)(std::forward<Args>(args)...));

		// 创建 QPromise 和 QFuture
		QPromise<ResultType> promise;
		auto				 future = promise.future();

		QMetaObject::invokeMethod(
			this,
			[this, func = std::forward<Func>(func),
			 args	 = std::tuple(std::forward<Args>(args)...),
			 promise = std::move(promise)]() mutable {
				// 使用 std::apply 调用成员函数
				if constexpr (std::is_void_v<ResultType>) {
					std::cout << "invoke thread: " << QThread::currentThreadId()
							  << std::endl;
					std::apply(
						[&](auto&&... params) {
							(core.get()->*func)(
								std::forward<decltype(params)>(params)...);
						},
						std::move(args));
					// 对于 void 类型，直接标记完成
					promise.finish();
				} else {
					std::cout << "invoke thread: " << QThread::currentThreadId()
							  << std::endl;
					ResultType result = std::apply(
						[&](auto&&... params) {
							return (core.get()->*func)(
								std::forward<decltype(params)>(params)...);
						},
						std::move(args));
					// 设置结果并标记完成
					promise.addResult(std::move(result));
					promise.finish();
				}
			},
			Qt::QueuedConnection);

		return future;
	}
};

#endif