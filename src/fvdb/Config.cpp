// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/Config.h>

namespace fvdb {

Config::Config() = default;

Config &
Config::global() {
    static Config _config;
    return _config;
}

void
Config::setUltraSparseAcceleration(bool enabled) {
    mUltraSparseAcceleration = enabled;
}

bool
Config::ultraSparseAccelerationEnabled() const {
    return mUltraSparseAcceleration;
}

void
Config::setPedanticErrorChecking(bool enabled) {
    mPedanticErrorChecking = enabled;
}
bool
Config::pedanticErrorCheckingEnabled() const {
    return mPedanticErrorChecking;
}

} // namespace fvdb
