// pydds_binding.cpp - Python bindings for DDSCOMM
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include "DDSCOMM.h"

namespace py = pybind11;

// Python Callback Manager
class PyCallbackManager {
public:
  static PyCallbackManager& instance() {
    static PyCallbackManager mgr;
    return mgr;
  }

  void registerCallback(int topicId, py::object callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_[topicId] = callback;
  }

  void unregisterCallback(int topicId) {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.erase(topicId);
  }

  static void cppCallback(int topicId, char* msgBuffer, int msgLen) {
    std::lock_guard<std::mutex> lock(instance().mutex_);

    auto it = instance().callbacks_.find(topicId);
    if (it == instance().callbacks_.end()) {
      return;
    }

    try {
      py::gil_scoped_acquire acquire;
      py::bytes data(msgBuffer, msgLen);
      it->second(topicId, data);
    }
    catch (py::error_already_set& e) {
      std::cerr << "[PyDDS Error] " << e.what() << std::endl;
    }
  }

private:
  PyCallbackManager() = default;
  std::map<int, py::object> callbacks_;
  std::mutex mutex_;
};

// Wrapper functions
bool py_subscribe(int topicId, py::object callback) {
  PyCallbackManager::instance().registerCallback(topicId, callback);
  return DDSCOMM::SubTopic(topicId, PyCallbackManager::cppCallback);
}

bool py_subscribe_with_qos(int topicId, py::object callback, const DDSCOMM::QosConfig& qos) {
  PyCallbackManager::instance().registerCallback(topicId, callback);
  return DDSCOMM::SubTopic(topicId, PyCallbackManager::cppCallback, qos);
}

bool py_unsubscribe(int topicId) {
  PyCallbackManager::instance().unregisterCallback(topicId);
  return DDSCOMM::UnsubTopic(topicId);
}

bool py_publish(int topicId, py::bytes data) {
  char* buffer;
  ssize_t length;
  PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length);
  return DDSCOMM::PubTopic(topicId, buffer, static_cast<int>(length));
}

bool py_publish_with_qos(int topicId, py::bytes data, const DDSCOMM::QosConfig& qos) {
  char* buffer;
  ssize_t length;
  PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length);
  return DDSCOMM::PubTopic(topicId, buffer, static_cast<int>(length), qos);
}

bool py_publish_with_retry(int topicId, py::bytes data, int maxRetries, int retryDelayMs) {
  char* buffer;
  ssize_t length;
  PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length);
  return DDSCOMM::PubTopicWithRetry(topicId, buffer, static_cast<int>(length),
    maxRetries, retryDelayMs);
}

bool py_publish_with_priority(int topicId, py::bytes data, int priority) {
  char* buffer;
  ssize_t length;
  PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length);
  return DDSCOMM::PubTopicWithPriority(topicId, buffer, static_cast<int>(length), priority);
}

PYBIND11_MODULE(pydds_bindings, m) {
  m.doc() = "Python bindings for DDSCOMM";

  py::enum_<DDSCOMM::ReliabilityLevel>(m, "ReliabilityLevel")
    .value("BEST_EFFORT", DDSCOMM::BEST_EFFORT)
    .value("RELIABLE", DDSCOMM::RELIABLE)
    .value("STRICT_RELIABLE", DDSCOMM::STRICT_RELIABLE)
    .export_values();

  py::class_<DDSCOMM::QosConfig>(m, "QosConfig")
    .def(py::init<>())
    .def_readwrite("reliability", &DDSCOMM::QosConfig::reliability)
    .def_readwrite("history_depth", &DDSCOMM::QosConfig::historyDepth)
    .def_readwrite("max_samples", &DDSCOMM::QosConfig::maxSamples)
    .def_readwrite("keep_all_samples", &DDSCOMM::QosConfig::keepAllSamples)
    .def_readwrite("priority", &DDSCOMM::QosConfig::priority)
    .def_readwrite("timeout_sec", &DDSCOMM::QosConfig::timeoutSec)
    .def_readwrite("wait_for_ack", &DDSCOMM::QosConfig::waitForAck)
    .def_readwrite("transient_local", &DDSCOMM::QosConfig::transientLocal)
    .def_readwrite("max_retries", &DDSCOMM::QosConfig::maxRetries)
    .def_readwrite("retry_delay_ms", &DDSCOMM::QosConfig::retryDelayMs);

  m.def("init", py::overload_cast<int>(&DDSCOMM::MsgBusInit), py::arg("domain_id"));
  m.def("shutdown", &DDSCOMM::MsgBusShutdown);

  m.def("publish", &py_publish, py::arg("topic_id"), py::arg("data"));
  m.def("publish_with_qos", &py_publish_with_qos, py::arg("topic_id"), py::arg("data"), py::arg("qos"));
  m.def("publish_with_retry", &py_publish_with_retry, py::arg("topic_id"), py::arg("data"),
    py::arg("max_retries") = 3, py::arg("retry_delay_ms") = 500);
  m.def("publish_with_priority", &py_publish_with_priority, py::arg("topic_id"), py::arg("data"), py::arg("priority"));

  m.def("subscribe", &py_subscribe, py::arg("topic_id"), py::arg("callback"));
  m.def("subscribe_with_qos", &py_subscribe_with_qos, py::arg("topic_id"), py::arg("callback"), py::arg("qos"));
  m.def("unsubscribe", &py_unsubscribe, py::arg("topic_id"));

  m.def("wait_for_subscribers", &DDSCOMM::WaitForSubscribers, py::arg("topic_id"), py::arg("timeout_sec") = 3);
  m.def("wait_for_publishers", &DDSCOMM::WaitForPublishers, py::arg("topic_id"), py::arg("timeout_sec") = 3);
  m.def("get_subscriber_count", &DDSCOMM::GetSubscriberCount, py::arg("topic_id"));
  m.def("get_publisher_count", &DDSCOMM::GetPublisherCount, py::arg("topic_id"));
  m.def("is_initialized", &DDSCOMM::IsInitialized);
  m.def("get_last_error", &DDSCOMM::GetLastError);

  m.attr("__version__") = "1.0.0";
}
