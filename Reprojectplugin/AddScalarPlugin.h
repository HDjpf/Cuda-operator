/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cookbookHelper.hpp"
#include <cublas_v2.h>

namespace
{
static const char *PLUGIN_NAME {"Reproject"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{
class AddScalarPlugin : public IPluginV2DynamicExt
{
private:
    const std::string name_;
    std::string       namespace_;
    // struct
    // {
    //     float scalar;
    // } m_;

public:
    AddScalarPlugin() = delete;
    AddScalarPlugin(const std::string &name);
    // AddScalarPlugin(const std::string &name, float scalar);
    // AddScalarPlugin(const std::string &name, const void *buffer, size_t length);
    ~AddScalarPlugin();

    //Method myself
    float * h_src_proj_ {nullptr};
    float * h_regs_ {nullptr};
    float * cdhw {nullptr};
    float * intr_mat_ {nullptr};
    float * intr_mat_inv_ {nullptr};
    cudaStream_t stream_;
    cublasHandle_t handle;
    void reprojection(const float *intr_mat,const float *intr_mat_inv,float *trans,float *rot,const float *src_fea,
                            const float *src_proj,const float *ref_proj,const float *depth_samples,int width,int height,int channels,int depth_num,float *ret,
                            float *wa,float *wb,float *wc,float *wd,int *x0,int *x1,int *y0,int *y1,float *d_xyz,float *d_rot_xyz,float *depth_xyz,float* d_src_proj,
                            float* d_ref_proj,float* d_ref_proj_trans,float* d_proj,float* d_regs,float* h_src_proj,float* h_regs,float* src_proj_cache,float* ref_proj_cache,
                            cudaStream_t stream);
    void GetRotTrans(const float *intr_mat,const float *intr_mat_inv,float *rot,float *trans,const float *src_proj,const float *ref_proj,float* d_src_proj_,float* d_ref_proj_,
                    float* d_ref_proj_trans,float* d_proj_,float* d_regs_,float* h_src_proj_,float* h_regs_,float* src_proj_cache,float* ref_proj_cache,
                            cudaStream_t stream);
    void reprojection_half(const half *intr_mat,const half *intr_mat_inv,float *trans,float *rot,const half *src_fea,
                            const half *src_proj,const half *ref_proj,const half *depth_samples,int width,int height,int channels,int depth_num,half *ret,
                            float *wa,float *wb,float *wc,float *wd,int *x0,int *x1,int *y0,int *y1,float *d_xyz,float *d_rot_xyz,float *depth_xyz,float* d_src_proj,
                            float* d_ref_proj,float* d_ref_proj_trans,float* d_proj,float* d_regs,float* src_proj_cache,float* ref_proj_cache,
                            cudaStream_t stream);
    void GetRotTrans_half(const half *intr_mat,const half *intr_mat_inv,float *rot,float *trans,const half *src_proj,const half *ref_proj,float* d_src_proj_,float* d_ref_proj_,
                            float* d_ref_proj_trans,float* d_proj_,float* d_regs_,float* src_proj_cache,float* ref_proj_cache,
                            cudaStream_t stream);

    // Method inherited from IPluginV2
    const char *getPluginType() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    size_t      getSerializationSize() const noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    void        setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    void     attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void     detachFromContext() noexcept override;

    //Method inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt *clone() const noexcept override;
    DimsExprs            getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    bool                 supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void                 configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;
    size_t               getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;
    int32_t              enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

private:


};

class AddScalarPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    AddScalarPluginCreator();
    ~AddScalarPluginCreator();
    const char *                 getPluginName() const noexcept override;
    const char *                 getPluginVersion() const noexcept override;
    const PluginFieldCollection *getFieldNames() noexcept override;
    IPluginV2 *                  createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;
    IPluginV2 *                  deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;
    void                         setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *                 getPluginNamespace() const noexcept override;
};

} // namespace nvinfer1
