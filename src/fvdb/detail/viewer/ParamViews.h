// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_VIEWER_PARAMVIEWS_H
#define FVDB_DETAIL_VIEWER_PARAMVIEWS_H

#include <cstdint>
#include <string>

namespace fvdb::detail::viewer {

// Forward declaration
class Viewer;

inline constexpr const char *kSubmitCounterSuffix = "__fvdb_submit__";

class SliderView {
    friend class Viewer;

    std::string mSceneName;
    std::string mName;
    Viewer *mViewer = nullptr;

    float mMin     = 0.f;
    float mMax     = 1.f;
    float mStep    = 0.01f;
    float mInitial = 0.f;

  public:
    SliderView(const std::string &sceneName,
               const std::string &name,
               Viewer *viewer,
               float min,
               float max,
               float step,
               float initial)
        : mSceneName(sceneName), mName(name), mViewer(viewer), mMin(min), mMax(max), mStep(step),
          mInitial(initial) {}

    const std::string &
    getName() const {
        return mName;
    }
    const std::string &
    getSceneName() const {
        return mSceneName;
    }

    float
    getMin() const {
        return mMin;
    }
    float
    getMax() const {
        return mMax;
    }
    float
    getStep() const {
        return mStep;
    }

    float getValue() const;
    void setValue(float value);
};

class NumberView {
    friend class Viewer;

    std::string mSceneName;
    std::string mName;
    Viewer *mViewer = nullptr;

    bool mHasMin   = false;
    bool mHasMax   = false;
    float mMin     = 0.f;
    float mMax     = 0.f;
    float mStep    = 0.01f;
    float mInitial = 0.f;

  public:
    NumberView(const std::string &sceneName,
               const std::string &name,
               Viewer *viewer,
               bool hasMin,
               float min,
               bool hasMax,
               float max,
               float step,
               float initial)
        : mSceneName(sceneName), mName(name), mViewer(viewer), mHasMin(hasMin), mHasMax(hasMax),
          mMin(min), mMax(max), mStep(step), mInitial(initial) {}

    const std::string &
    getName() const {
        return mName;
    }
    const std::string &
    getSceneName() const {
        return mSceneName;
    }

    bool
    hasMin() const {
        return mHasMin;
    }
    bool
    hasMax() const {
        return mHasMax;
    }
    float
    getMin() const {
        return mMin;
    }
    float
    getMax() const {
        return mMax;
    }
    float
    getStep() const {
        return mStep;
    }

    float getValue() const;
    void setValue(float value);
};

class TextView {
    friend class Viewer;

    std::string mSceneName;
    std::string mName;
    Viewer *mViewer = nullptr;

    int32_t mMaxLength = 256;
    std::string mInitial;
    bool mCommitOnEnter = false;

  public:
    TextView(const std::string &sceneName,
             const std::string &name,
             Viewer *viewer,
             int32_t maxLength,
             const std::string &initial,
             bool commitOnEnter)
        : mSceneName(sceneName), mName(name), mViewer(viewer), mMaxLength(maxLength),
          mInitial(initial), mCommitOnEnter(commitOnEnter) {}

    const std::string &
    getName() const {
        return mName;
    }
    const std::string &
    getSceneName() const {
        return mSceneName;
    }

    int32_t
    getMaxLength() const {
        return mMaxLength;
    }

    bool
    getCommitOnEnter() const {
        return mCommitOnEnter;
    }

    std::string getValue() const;
    void setValue(const std::string &value);

    // Returns 0 when the widget was created with ``commit_on_enter=False``.
    uint32_t getSubmitCounter() const;
};

class CheckboxView {
    friend class Viewer;

    std::string mSceneName;
    std::string mName;
    Viewer *mViewer = nullptr;

    bool mInitial = false;

  public:
    CheckboxView(const std::string &sceneName,
                 const std::string &name,
                 Viewer *viewer,
                 bool initial)
        : mSceneName(sceneName), mName(name), mViewer(viewer), mInitial(initial) {}

    const std::string &
    getName() const {
        return mName;
    }
    const std::string &
    getSceneName() const {
        return mSceneName;
    }

    bool getValue() const;
    void setValue(bool value);
};

} // namespace fvdb::detail::viewer
#endif // FVDB_DETAIL_VIEWER_PARAMVIEWS_H
