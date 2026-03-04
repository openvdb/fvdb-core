// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_CONFIG_H
#define FVDB_CONFIG_H

namespace fvdb {

class Config {
  public:
    Config();

    void setUltraSparseAcceleration(bool enabled);
    bool ultraSparseAccelerationEnabled() const;

    void setPedanticErrorChecking(bool enabled);
    bool pedanticErrorCheckingEnabled() const;

    static Config &global();

  private:
    bool mUltraSparseAcceleration = false;
    bool mPedanticErrorChecking   = false;
};

} // namespace fvdb

#endif // FVDB_CONFIG_H
