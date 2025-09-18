// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/io/GaussianPlyIO.h>

// Ops headers
#include <fvdb/detail/ops/gsplat/GaussianComputeNanInfMask.h>

// Utils headers
#include <fvdb/detail/utils/Utils.h>

#include <c10/core/ScalarType.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/api/include/torch/types.h>

#include <exception>

#define TINYPLY_IMPLEMENTATION
#include <tinyply.h>

#include <fstream>
#include <ostream>

namespace fvdb::detail::io {

/// @brief Get a uint8_t pointer to the data of a tensor
/// @param tensor The tensor to get the pointer to
/// @return A uint8_t pointer to the data of the tensor
inline uint8_t *
tensorBytePointer(const torch::Tensor &tensor) {
    return static_cast<uint8_t *>(tensor.data_ptr());
}

/// @brief Convert a torch::ScalarType to its corresponding tinyply::Type
/// @param dtype The torch::ScalarType to convert
/// @return The corresponding tinyply::Type
tinyply::Type
tensorDtypeToPlyType(const torch::ScalarType &dtype) {
    using namespace tinyply;

    switch (dtype) {
    case torch::kFloat32: return Type::FLOAT32;
    case torch::kFloat64: return Type::FLOAT64;
    case torch::kInt32: return Type::INT32;
    case torch::kUInt32: return Type::UINT32;
    case torch::kInt16: return Type::INT16;
    case torch::kUInt16: return Type::UINT16;
    case torch::kInt8: return Type::INT8;
    case torch::kUInt8: return Type::UINT8;
    default: return Type::INVALID;
    }
}

/// @brief Convert a tinyply::Type to its corresponding torch::ScalarType
/// @param plyType The tinyply::Type to convert
/// @return The corresponding torch::ScalarType
torch::ScalarType
plyTypeToTensorDtype(const tinyply::Type &plyType) {
    using namespace tinyply;
    switch (plyType) {
    case Type::FLOAT32: return torch::kFloat32;
    case Type::FLOAT64: return torch::kFloat64;
    case Type::INT32: return torch::kInt32;
    case Type::UINT32: return torch::kUInt32;
    case Type::INT16: return torch::kInt16;
    case Type::UINT16: return torch::kUInt16;
    case Type::INT8: return torch::kInt8;
    case Type::UINT8: return torch::kUInt8;
    default: TORCH_CHECK(false, "Unsupported PLY tensor data type"); // Invalid type
    }
}

/// @brief Parse PLY metadata comments to extract either values (for string, int64, and double
/// metadata types) or tensor shapes for tensor metadata types.
/// @param plyComments A vector of PLY comments to be parsed.
/// @return A tuple containing two maps: one for scalar metadata and one for tensor metadata.
/// Each map has keys matching the names of the metadata. The scalar metadata map, contains the
/// values of the metadata and the tensor map contains the shape of the tensors to load.
std::tuple<
    std::unordered_map<std::string, PlyMetadataTypes>,
    std::unordered_map<std::string, std::tuple<std::shared_ptr<PlyData>, std::vector<int64_t>>>>
parsePlyMetadataComments(tinyply::PlyFile &plyf) {
    std::unordered_map<std::string, PlyMetadataTypes> retMetadata;
    std::unordered_map<std::string, std::tuple<std::shared_ptr<PlyData>, std::vector<int64_t>>>
        retTensorMetadata;
    // A metadata comment has the form:
    // <PLY_MAGIC><key>|<type>|<value_or_tensor_shape>
    for (auto comment: plyf.get_comments()) {
        // Check if the comment starts with PLY_MAGIC
        if (comment.rfind(PLY_MAGIC) != 0) {
            continue;
        }

        // Find the first separator to extract the key
        const auto firstSep = comment.find("|");
        if (firstSep == std::string::npos) {
            continue; // Not a metadata comment
        }
        const auto key = comment.substr(PLY_MAGIC.size(), firstSep - PLY_MAGIC.size());

        const auto secondSep = comment.find("|", firstSep + 1);
        const auto type      = comment.substr(firstSep + 1, secondSep - firstSep - 1);

        const auto value = comment.substr(secondSep + 1);

        if (type == "str") {
            retMetadata[key] = value;
        } else if (type == "double") {
            retMetadata[key] = std::stod(value);
        } else if (type == "int64") {
            retMetadata[key] = std::stoll(value);
        } else if (type == "tensor") {
            auto commaPos = value.find(",");

            const int64_t ndim = std::stoll(value.substr(0, commaPos));
            std::vector<int64_t> tensorShape;
            tensorShape.reserve(ndim);
            for (int64_t i = 0; i < ndim; ++i) {
                const auto nextComma = value.find(",", commaPos + 1);
                tensorShape.push_back(
                    std::stoll(value.substr(commaPos + 1, nextComma - commaPos - 1)));
                commaPos = nextComma;
            }
            std::shared_ptr<PlyData> tensorData =
                plyf.request_properties_from_element(key, {"value"});
            TORCH_CHECK(tensorData != nullptr,
                        "Failed to read tensor metadata '" + key +
                            "'. Make sure it was written with fvdb::GaussianSplat3d::savePly");
            retTensorMetadata[key] = std::make_tuple(tensorData, tensorShape);
        } else {
            continue; // Not a metadata comment
        }
    }

    return std::make_tuple(retMetadata, retTensorMetadata);
}

/// @brief Generate a comment encoding metadata with the given name (key) and value as a string.
/// If the value is a scalar type (int64, double, str), then the comment will include the value
/// directly. If the value is a tensor, then the comment will include the number of dimensions and
/// shape of the tensor separated by commas (i.e. 'ndim,size0,..., sizeD').
/// Each comment has the form "<PLY_MAGIC>|<TYPE>|<VALUE_OR_TENSOR_SIZE>"
/// @param key The name of the metadata attribute
/// @param value The value of the metadata attribute (int64, double, str, or tensor)
/// @return A string encoding the metadata comment of the form
/// "<PLY_MAGIC>|<TYPE>|<VALUE_OR_TENSOR_SIZE>"
std::string
plyMetadataComment(const std::string &key, const PlyMetadataTypes &value) {
    TORCH_CHECK_VALUE(key.size() <= MAX_PLY_KEY_LENGTH,
                      "Invalid PLY key '" + key + ": exceeds maximum length of " +
                          std::to_string(MAX_PLY_KEY_LENGTH) + " characters");
    TORCH_CHECK_VALUE(
        std::all_of(key.begin(),
                    key.end(),
                    [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }),
        "key '" + key + "' can only contain alphanumeric characters and underscores.");

    if (std::holds_alternative<std::string>(value)) {
        return PLY_MAGIC + key + "|str|" + std::get<std::string>(value);
    } else if (std::holds_alternative<int64_t>(value)) {
        return PLY_MAGIC + key + "|int64|" + std::to_string(std::get<int64_t>(value));
    } else if (std::holds_alternative<double>(value)) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10)
           << std::get<double>(value);
        return PLY_MAGIC + key + "|double|" + ss.str();
    } else if (std::holds_alternative<torch::Tensor>(value)) {
        const auto tensorValue = std::get<torch::Tensor>(value);
        std::stringstream ss;
        ss << std::to_string(tensorValue.dim()) << ",";
        for (auto i = 0; i < tensorValue.dim(); ++i) {
            ss << std::to_string(tensorValue.size(i));
            if (i < tensorValue.dim() - 1) {
                ss << ",";
            }
        }
        return PLY_MAGIC + key + "|tensor|" + ss.str();
    } else {
        TORCH_CHECK(false, "Invalid metadata type. Must be str, int, float, or torch.Tensor.");
    }
}

std::tuple<GaussianSplat3d, std::unordered_map<std::string, PlyMetadataTypes>>
loadGaussianPly(const std::string &filename, torch::Device device) {
    using namespace tinyply;

    std::ifstream instream(filename, std::ios::binary);

    PlyFile plyf;
    plyf.parse_header(instream);

    std::vector<PlyElement> elements = plyf.get_elements();

    // Find the vertex element
    auto vertex_element_iter = std::find_if(
        elements.begin(), elements.end(), [](const PlyElement &e) { return e.name == "vertex"; });

    if (vertex_element_iter == elements.end()) {
        throw std::runtime_error("No vertex element found in PLY file");
    }

    // Request position data (x, y, z)
    std::shared_ptr<PlyData> meansData =
        plyf.request_properties_from_element("vertex", {"x", "y", "z"});
    std::shared_ptr<PlyData> logitOpacitiesData =
        plyf.request_properties_from_element("vertex", {"opacity"});
    std::shared_ptr<PlyData> logScalesData =
        plyf.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
    std::shared_ptr<PlyData> quatsData =
        plyf.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});

    // Find all SH coefficient properties
    std::vector<std::string> sh0PlyPropertyNames;
    std::vector<std::string> shNPlyPropertyNames;

    for (const auto &prop: vertex_element_iter->properties) {
        if (prop.name.substr(0, 5) == "f_dc_") {
            sh0PlyPropertyNames.push_back(prop.name);
        } else if (prop.name.substr(0, 7) == "f_rest_") {
            shNPlyPropertyNames.push_back(prop.name);
        }
    }

    // Request SH coefficient data
    std::shared_ptr<PlyData> sh0Data =
        plyf.request_properties_from_element("vertex", sh0PlyPropertyNames);
    std::shared_ptr<PlyData> shNData;
    try {
        shNData = plyf.request_properties_from_element("vertex", shNPlyPropertyNames);
    } catch (std::exception &e) {
        // If there are no SH N coefficients, tinyply will throw an exception. We can ignore this
        // and just set shNData to nullptr
        shNData.reset();
    }

    // Read out metadata from comment strings
    auto [retMetadata, retTensorMetadata] = parsePlyMetadataComments(plyf);

    // Read the file
    plyf.read(instream);

    // Get the number of vertices
    size_t vertex_count = meansData->count;

    // Create tensors to hold the data
    torch::Tensor means = torch::from_blob(
        meansData->buffer.get(), {static_cast<int64_t>(vertex_count), 3}, torch::kFloat32);
    torch::Tensor logitOpacities = torch::from_blob(
        logitOpacitiesData->buffer.get(), {static_cast<int64_t>(vertex_count)}, torch::kFloat32);
    torch::Tensor logScales = torch::from_blob(
        logScalesData->buffer.get(), {static_cast<int64_t>(vertex_count), 3}, torch::kFloat32);
    torch::Tensor quats = torch::from_blob(
        quatsData->buffer.get(), {static_cast<int64_t>(vertex_count), 4}, torch::kFloat32);

    // Create tensor to hold SH coefficients
    const int numChannels  = static_cast<int>(sh0PlyPropertyNames.size());
    const int numShNCoeffs = static_cast<int>(shNPlyPropertyNames.size());
    const int nShNBases    = numShNCoeffs / numChannels;

    torch::Tensor sh0Coeffs; // (N, 1, D)
    if (sh0Data && sh0Data->count > 0) {
        sh0Coeffs = torch::from_blob(sh0Data->buffer.get(),
                                     {static_cast<int64_t>(vertex_count), 1, numChannels},
                                     torch::kFloat32);
    }
    torch::Tensor shNCoeffs; // (N, K-1, D)
    if (shNData && shNData->count > 0) {
        shNCoeffs = torch::from_blob(shNData->buffer.get(),
                                     {static_cast<int64_t>(vertex_count), nShNBases, numChannels},
                                     torch::kFloat32);
    } else {
        shNCoeffs =
            torch::empty({static_cast<int64_t>(vertex_count), 0, numChannels}, torch::kFloat32);
    }

    for (auto kv: retTensorMetadata) {
        const auto key         = kv.first;
        const auto plyData     = std::get<0>(kv.second);
        const auto tensorShape = std::get<1>(kv.second);
        const auto tensorDtype = plyTypeToTensorDtype(plyData->t);
        const auto tensor =
            torch::from_blob(plyData->buffer.get(), tensorShape, tensorDtype).to(device);

        retMetadata[key] = tensor;
    }

    return std::make_tuple(GaussianSplat3d(means.to(device),
                                           quats.to(device),
                                           logScales.to(device),
                                           logitOpacities.to(device),
                                           sh0Coeffs.to(device),
                                           shNCoeffs.to(device),
                                           false,
                                           false,
                                           false),
                           retMetadata);
}

void
saveGaussianPly(const std::string &filename,
                const GaussianSplat3d &gaussians,
                std::optional<std::unordered_map<std::string, PlyMetadataTypes>> trainingMetadata) {
    using namespace tinyply;

    const fvdb::JaggedTensor validMask =
        FVDB_DISPATCH_KERNEL_DEVICE(gaussians.means().device(), [&]() {
            return detail::ops::dispatchGaussianNanInfMask<DeviceTag>(gaussians.means(),
                                                                      gaussians.quats(),
                                                                      gaussians.logScales(),
                                                                      gaussians.logitOpacities(),
                                                                      gaussians.sh0(),
                                                                      gaussians.shN());
        });

    std::filebuf fb;
    fb.open(filename, std::ios::out | std::ios::binary);

    std::ostream outstream(&fb);
    TORCH_CHECK(!outstream.fail(), "failed to open " + filename);

    PlyFile plyf;

    const torch::Tensor meansCPU =
        gaussians.means().index({validMask.jdata(), torch::indexing::Ellipsis}).cpu().contiguous();
    const torch::Tensor quatsCPU =
        gaussians.quats().index({validMask.jdata(), torch::indexing::Ellipsis}).cpu().contiguous();
    const torch::Tensor scalesCPU = gaussians.logScales()
                                        .index({validMask.jdata(), torch::indexing::Ellipsis})
                                        .cpu()
                                        .contiguous();
    const torch::Tensor opacitiesCPU =
        gaussians.logitOpacities().index({validMask.jdata()}).cpu().contiguous();

    // [N, D]
    const torch::Tensor shCoeffs0CPU =
        gaussians.sh0().index({validMask.jdata(), 0, torch::indexing::Ellipsis}).cpu().contiguous();
    // [N, K-1, D]
    const torch::Tensor shCoeffsNCPU = [&]() {
        if (gaussians.shN().numel() <= 0) {
            return torch::zeros({meansCPU.size(0), 0},
                                gaussians.shN().options().device(torch::kCPU));
        } else {
            return gaussians.shN()
                .index({validMask.jdata(), torch::indexing::Slice(), torch::indexing::Ellipsis})
                .cpu()
                .contiguous()
                .permute({0, 2, 1})
                .reshape({meansCPU.size(0), -1});
        }
    }();

    plyf.add_properties_to_element("vertex",
                                   {"x", "y", "z"},
                                   Type::FLOAT32,
                                   meansCPU.size(0),
                                   tensorBytePointer(meansCPU),
                                   Type::INVALID,
                                   0);
    plyf.add_properties_to_element("vertex",
                                   {"opacity"},
                                   Type::FLOAT32,
                                   opacitiesCPU.size(0),
                                   tensorBytePointer(opacitiesCPU),
                                   Type::INVALID,
                                   0);
    plyf.add_properties_to_element("vertex",
                                   {"scale_0", "scale_1", "scale_2"},
                                   Type::FLOAT32,
                                   scalesCPU.size(0),
                                   tensorBytePointer(scalesCPU),
                                   Type::INVALID,
                                   0);
    plyf.add_properties_to_element("vertex",
                                   {"rot_0", "rot_1", "rot_2", "rot_3"},
                                   Type::FLOAT32,
                                   quatsCPU.size(0),
                                   tensorBytePointer(quatsCPU),
                                   Type::INVALID,
                                   0);

    std::vector<std::string> shCoeff0Names(shCoeffs0CPU.size(1));
    std::generate(shCoeff0Names.begin(), shCoeff0Names.end(), [i = 0]() mutable {
        return "f_dc_" + std::to_string(i++);
    });
    plyf.add_properties_to_element("vertex",
                                   shCoeff0Names,
                                   Type::FLOAT32,
                                   shCoeffs0CPU.size(0),
                                   tensorBytePointer(shCoeffs0CPU),
                                   Type::INVALID,
                                   0);

    std::vector<std::string> shCoeffNNames(shCoeffsNCPU.size(1));
    std::generate(shCoeffNNames.begin(), shCoeffNNames.end(), [i = 0]() mutable {
        return "f_rest_" + std::to_string(i++);
    });
    plyf.add_properties_to_element("vertex",
                                   shCoeffNNames,
                                   Type::FLOAT32,
                                   shCoeffsNCPU.size(0),
                                   tensorBytePointer(shCoeffsNCPU),
                                   Type::INVALID,
                                   0);

    plyf.get_comments().push_back("fvdb_gs_ply_version " + PLY_VERSION_STRING);

    // We need to keep copies of the values we will write out (allocated in the if block)
    // so they don't get freed before calling plyf.write()
    std::vector<torch::Tensor> valuesToWrite;
    if (trainingMetadata.has_value()) {
        for (auto kv: trainingMetadata.value()) {
            plyf.get_comments().push_back(plyMetadataComment(kv.first, kv.second));

            // If the value is a tensor, we need to also write the tensor data as an attribute to
            // the PLY file
            if (std::holds_alternative<torch::Tensor>(kv.second)) {
                const auto tensorValue = std::get<torch::Tensor>(kv.second);
                const auto plyType     = tensorDtypeToPlyType(tensorValue.scalar_type());
                TORCH_CHECK_VALUE(
                    plyType != Type::INVALID,
                    "Unsupported dtype for key '" + kv.first +
                        "'. Must be one of (float32, float64, int32, uint32, int16, uint16, int8, uint8)");
                const auto valueData  = tensorValue.reshape({-1}).cpu().contiguous();
                const auto numScalars = valueData.numel();
                valuesToWrite.push_back(valueData);

                plyf.add_properties_to_element(kv.first,
                                               {"value"},
                                               plyType,
                                               numScalars,
                                               tensorBytePointer(valueData),
                                               Type::INVALID,
                                               0);
            }
        }
    }

    plyf.write(outstream, true);
}

} // namespace fvdb::detail::io
