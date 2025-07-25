#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>

namespace hopsy::GPU {
    void set_device(int device);
    std::vector<std::string> list_devices();
}
