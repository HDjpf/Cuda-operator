/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless renqueueuired by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "AddScalarPlugin.h"

#include <iostream>
using namespace std;


#define ROTATE_ROWS 3
#define ROTATE_COLS 3
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}



typedef unsigned short ushort;//占用2个字节
typedef unsigned int uint;    //占用4个字节
 
uint as_uint(const float x) {
    return *(uint*)&x;
}
float as_float(const uint x) {
    return *(float*)&x;
}
 
float half_to_float(const ushort x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint e = (x&0x7C00)>>10; // exponent
    const uint m = (x&0x03FF)<<13; // mantissa
    const uint v = as_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000))); // sign : normalized : denormalized
}
ushort float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint b = as_uint(x)+0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const uint e = (b&0x7F800000)>>23; // exponent
    const uint m = b&0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    return (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7C00)|m>>13) | ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF; // sign : normalized : denormalized : saturate
}

//thread [width,height]
//xyz [x0,y0,z0,x1,y1,z1,x2,y2,z2......]
//template<class Data_type0>
__global__ void intial_xyz(float* d_xyz,int width,int height){
    int Idx = threadIdx.x + blockDim.x*blockIdx.x;
    int Idy = threadIdx.y + blockDim.y*blockIdx.y;
    int row_size = gridDim.x*blockDim.x;
    if(Idx >= width || Idy >= height){
        return;
    }
    if(row_size > width){
        row_size = width;
    }
    
    d_xyz[Idy*3*row_size + Idx*3  ] = Idx;
    d_xyz[Idy*3*row_size + Idx*3+1] = Idy;
    d_xyz[Idy*3*row_size + Idx*3+2] = 1.0;
    
}

//thread block[32,32]  grid[width/32,height/32,depth_num]
template<class Data_type1>
__global__ void depthmulxyz(int *x1,int *x0,int *y1,int *y0,float *wa,float *wb,float *wc,float *wd,
                            const Data_type1* src_fea,const Data_type1 *depth,float *rot_xyz,int width,int height,int channels, int depth_num,
                            float *depth_xyz,float *trans,Data_type1* ret){
    int Idx = threadIdx.x + blockDim.x*blockIdx.x;
    int Idy = blockDim.x*gridDim.x*blockDim.y*blockIdx.y + blockDim.x*gridDim.x*threadIdx.y;
    int Idz = threadIdx.z + blockDim.z*blockIdx.z;
    int row_size = gridDim.x*blockDim.x;
    // int Id_depth = Idz/channels;
    // int Id_channel = Idz%channels;
    if(Idx >= width || threadIdx.y + blockDim.y*blockIdx.y >= height || (Idz >= depth_num)){ //Id_depth
        return;
    }
    if(row_size > width){
        row_size = width;
        Idy = row_size*(threadIdx.y + blockDim.y*blockIdx.y);
    }


    float Ia;
    float Ib;
    float Ic;
    float Id;
    
    int per_pic_size = height*width;
    int patchid_depth = Idz;
    int depth_pixel_loca = patchid_depth*per_pic_size+Idy+Idx;
    int pic_y = 3*Idy;
    int pic_x = 3*Idx;
    int depth_pic_z = 3*Idz*per_pic_size;
 

    depth_xyz[depth_pic_z+pic_y+pic_x  ] = (float)depth[depth_pixel_loca]*rot_xyz[pic_y+pic_x  ] + trans[0];
    depth_xyz[depth_pic_z+pic_y+pic_x+1] = (float)depth[depth_pixel_loca]*rot_xyz[pic_y+pic_x+1] + trans[1];
    depth_xyz[depth_pic_z+pic_y+pic_x+2] = (float)depth[depth_pixel_loca]*rot_xyz[pic_y+pic_x+2] + trans[2];
  

    if(depth_xyz[depth_pic_z+pic_y+pic_x+2] < (float)1e-5){
        depth_xyz[depth_pic_z+pic_y+pic_x+2] = 1;
        depth_xyz[depth_pic_z+pic_y+pic_x+1] = 0;
        depth_xyz[depth_pic_z+pic_y+pic_x  ] = 0;
    }else{
        depth_xyz[depth_pic_z+pic_y+pic_x  ] /= depth_xyz[depth_pic_z+pic_y+pic_x+2];
        depth_xyz[depth_pic_z+pic_y+pic_x+1] /= depth_xyz[depth_pic_z+pic_y+pic_x+2];
    }
     
    if (depth_xyz[depth_pic_z+pic_y+pic_x] >= (float)width
            || depth_xyz[depth_pic_z+pic_y+pic_x  ] < (float)0 ){
        depth_xyz[depth_pic_z+pic_y+pic_x  ] = 0;
    }
    if (depth_xyz[depth_pic_z+pic_y+pic_x+1] >= (float)height 
            || depth_xyz[depth_pic_z+pic_y+pic_x+1] < (float)0){
        depth_xyz[depth_pic_z+pic_y+pic_x+1] = 0;
    }


    //bilinear interpolate
    //test
    // wa[per_pic_size*patchid_depth + Idy + Idx] = depth_xyz[depth_pic_z+pic_y+pic_x  ];
    // wb[per_pic_size*patchid_depth + Idy + Idx] = depth_xyz[depth_pic_z+pic_y+pic_x+1];
    // wc[per_pic_size*patchid_depth + Idy + Idx] = depth_xyz[depth_pic_z+pic_y+pic_x+2];
    
    // wa[per_pic_size*patchid_depth + Idy + Idx] = 99999;
    // wb[per_pic_size*patchid_depth + Idy + Idx] = 99999;
    // wc[per_pic_size*patchid_depth + Idy + Idx] = 99999;
   
    // return;
    //
    
    x1[per_pic_size*patchid_depth + Idy + Idx] = (int)depth_xyz[depth_pic_z+pic_y+pic_x  ] + 1;
    y1[per_pic_size*patchid_depth + Idy + Idx] = (int)depth_xyz[depth_pic_z+pic_y+pic_x+1] + 1;
    x0[per_pic_size*patchid_depth + Idy + Idx] = (int)depth_xyz[depth_pic_z+pic_y+pic_x  ];
    y0[per_pic_size*patchid_depth + Idy + Idx] = (int)depth_xyz[depth_pic_z+pic_y+pic_x+1];
        
    
    
    wa[per_pic_size*patchid_depth + Idy + Idx] = ((float)x1[per_pic_size*patchid_depth + Idy + Idx] - depth_xyz[depth_pic_z+pic_y+pic_x])*
                                                        ((float)y1[per_pic_size*patchid_depth + Idy + Idx] - depth_xyz[depth_pic_z+pic_y+pic_x+1]);
    wb[per_pic_size*patchid_depth + Idy + Idx] = ((float)x1[per_pic_size*patchid_depth + Idy + Idx] - depth_xyz[depth_pic_z+pic_y+pic_x])*
                                                        (depth_xyz[depth_pic_z+pic_y+pic_x + 1] - (float)y0[per_pic_size*patchid_depth + Idy + Idx]);
    wc[per_pic_size*patchid_depth + Idy + Idx] = (depth_xyz[depth_pic_z+pic_y+pic_x] - (float)x0[per_pic_size*patchid_depth + Idy + Idx])*
                                                        ((float)y1[per_pic_size*patchid_depth + Idy + Idx] - depth_xyz[depth_pic_z+pic_y+pic_x+1]);
    wd[per_pic_size*patchid_depth + Idy + Idx] = (depth_xyz[depth_pic_z+pic_y+pic_x] - (float)x0[per_pic_size*patchid_depth + Idy + Idx])*
                                                        (depth_xyz[depth_pic_z+pic_y+pic_x+1] - (float)y0[per_pic_size*patchid_depth + Idy + Idx]);

    //clamp
    x1[per_pic_size*patchid_depth + Idy + Idx] = x1[per_pic_size*patchid_depth + Idy + Idx] - (int)(x1[per_pic_size*patchid_depth + Idy + Idx]/width);    
    y1[per_pic_size*patchid_depth + Idy + Idx] = y1[per_pic_size*patchid_depth + Idy + Idx] - (int)(y1[per_pic_size*patchid_depth + Idy + Idx]/height);    


    // Ia = src_fea[Id_channel*per_pic_size + y0[per_pic_size*patchid_depth + Idy + Idx]*width + x0[per_pic_size*patchid_depth + Idy + Idx]]*
    //             wa[per_pic_size*patchid_depth + Idy + Idx];
    // Ib = src_fea[Id_channel*per_pic_size + y1[per_pic_size*patchid_depth + Idy + Idx]*width + x0[per_pic_size*patchid_depth + Idy + Idx]]*
    //             wb[per_pic_size*patchid_depth + Idy + Idx];
    // Ic = src_fea[Id_channel*per_pic_size + y0[per_pic_size*patchid_depth + Idy + Idx]*width + x1[per_pic_size*patchid_depth + Idy + Idx]]*
    //             wc[per_pic_size*patchid_depth + Idy + Idx];
    // Id = src_fea[Id_channel*per_pic_size + y1[per_pic_size*patchid_depth + Idy + Idx]*width + x1[per_pic_size*patchid_depth + Idy + Idx]]*
    //             wd[per_pic_size*patchid_depth + Idy + Idx];
    // ret[Id_channel*per_pic_size*depth_num + patchid_depth*per_pic_size + Idy + Idx] = Ia + Ib + Ic + Id; 

    for(int i=0; i<channels; i++){
        Ia = (float)src_fea[i*per_pic_size + y0[per_pic_size*patchid_depth + Idy + Idx]*width + x0[per_pic_size*patchid_depth + Idy + Idx]]*
                wa[per_pic_size*patchid_depth + Idy + Idx];
        Ib = (float)src_fea[i*per_pic_size + y1[per_pic_size*patchid_depth + Idy + Idx]*width + x0[per_pic_size*patchid_depth + Idy + Idx]]*
                wb[per_pic_size*patchid_depth + Idy + Idx];
        Ic = (float)src_fea[i*per_pic_size + y0[per_pic_size*patchid_depth + Idy + Idx]*width + x1[per_pic_size*patchid_depth + Idy + Idx]]*
                wc[per_pic_size*patchid_depth + Idy + Idx];
        Id = (float)src_fea[i*per_pic_size + y1[per_pic_size*patchid_depth + Idy + Idx]*width + x1[per_pic_size*patchid_depth + Idy + Idx]]*
                wd[per_pic_size*patchid_depth + Idy + Idx];
        ret[i*per_pic_size*depth_num + patchid_depth*per_pic_size + Idy + Idx] = (Data_type1)(Ia + Ib + Ic + Id); 
    }

    //*/
}

template<class Data_type2>
__global__ void sub(Data_type2 *d_regs_,Data_type2 *d_src_proj_){
    int Idx = threadIdx.x + blockDim.x*blockIdx.x;
    d_regs_[Idx] = d_src_proj_[9+Idx] - d_regs_[Idx];
}

void AddScalarPlugin::GetRotTrans(const float *intr_mat,const float *intr_mat_inv,float *rot,float *trans,
                                const float *src_proj,const float *ref_proj,float* d_src_proj_,float* d_ref_proj_,
    float* d_ref_proj_trans,
    float* d_proj_,
    float* d_regs_,
    float* h_src_proj_,
    float* h_regs_,
    float* src_proj_cache,
    float* ref_proj_cache,
    cudaStream_t stream){
    // int x = threadIdx.x + blockDim.x*blockIdx.x;
    // int y = threadIdx.y + blockDim.y*blockIdx.y;

    float alpha = 1;
    float beta = 0;


    
    cublasStatus_t ret;
    cudaError_t ret_mem;
    cublasSetStream(handle, stream);
    // printf("getrottrans mul1 mat\n");
    cudaMemcpy((void*)&src_proj_cache[0],(void*)&src_proj[0],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[1],(void*)&src_proj[4],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[2],(void*)&src_proj[8],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[3],(void*)&src_proj[1],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[4],(void*)&src_proj[5],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[5],(void*)&src_proj[9],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[6],(void*)&src_proj[2],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[7],(void*)&src_proj[6],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[8],(void*)&src_proj[10],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[9],(void*)&src_proj[3],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[10],(void*)&src_proj[7],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&src_proj_cache[11],(void*)&src_proj[11],sizeof(float),cudaMemcpyDeviceToDevice);

    // cudaMemcpy((void*)mmm,(void*)src_proj_cache,sizeof(float)*3*4,cudaMemcpyDeviceToHost);cout<<"src_proj_cache:";
    // for(int i = 0;i<12;i++){
    //     cout<<mmm[i]<<"  ";
    // }cout<<endl;

    // for(int i =0;i<12;i++){
    //     cudaMemcpy((void*)mmm,(void*)&src_proj[i],sizeof(float),cudaMemcpyDeviceToHost);
    //     cout<<"mmm:"<<*mmm<<endl;
    // }
    
    cudaMemcpy((void*)&ref_proj_cache[0],(void*)&ref_proj[0],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[1],(void*)&ref_proj[4],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[2],(void*)&ref_proj[8],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[3],(void*)&ref_proj[1],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[4],(void*)&ref_proj[5],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[5],(void*)&ref_proj[9],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[6],(void*)&ref_proj[2],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[7],(void*)&ref_proj[6],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[8],(void*)&ref_proj[10],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[9],(void*)&ref_proj[3],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[10],(void*)&ref_proj[7],sizeof(float),cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)&ref_proj_cache[11],(void*)&ref_proj[11],sizeof(float),cudaMemcpyDeviceToDevice);

    // cudaMemcpy((void*)mmm,(void*)ref_proj_cache,sizeof(float)*3*4,cudaMemcpyDeviceToHost);
    // for(int i = 0;i<12;i++){
    //     cout<<mmm[i]<<"  ";
    // }cout<<endl;

    
    ret = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,4,3,&alpha,intr_mat_inv,3,src_proj_cache,3,&beta,d_src_proj_,3);
    if(ret != CUBLAS_STATUS_SUCCESS){cout<<"return 1:"<<ret<<endl;}
    // cudaMemcpy((void*)mmm,(void*)d_src_proj_,sizeof(float)*3*4,cudaMemcpyDeviceToHost);cout<<"d_src_proj_:";
    // for(int i = 0;i<12;i++){
    //     cout<<mmm[i]<<"  ";
    // }cout<<endl;
    ret = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,4,3,&alpha,intr_mat_inv,3,ref_proj_cache,3,&beta,d_ref_proj_,3);
    if(ret != CUBLAS_STATUS_SUCCESS){cout<<"return 2:"<<ret<<endl;}
    // cudaMemcpy((void*)mmm,(void*)d_ref_proj_,sizeof(float)*3*4,cudaMemcpyDeviceToHost);cout<<"d_ref_proj_:";
    // for(int i = 0;i<12;i++){
    //     cout<<mmm[i]<<"  ";
    // }cout<<endl;

    cudaMemcpy((void*)d_ref_proj_trans,(void*)&d_ref_proj_[9],sizeof(float)*3,cudaMemcpyDeviceToDevice);
    // cudaMemcpy((void*)mmm,(void*)d_ref_proj_trans,sizeof(float)*3,cudaMemcpyDeviceToHost);cout<<"d_ref_proj_trans: ";
    // for(int i = 0;i<3;i++){
    //     cout<<mmm[i]<<"  ";
    // }cout<<endl;
    ret = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,3,3,3,&alpha,d_src_proj_,3,d_ref_proj_,3,&beta,d_proj_,3);
    if(ret != CUBLAS_STATUS_SUCCESS){cout<<"return 3:"<<ret<<endl;}
    
    ret = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,3,1,3,&alpha,d_proj_,3,d_ref_proj_trans,3,&beta,d_regs_,3);
    if(ret != CUBLAS_STATUS_SUCCESS){cout<<"return 4:"<<ret<<endl;}
    // cudaMemcpy((void*)mmm,(void*)d_regs_,sizeof(float)*3,cudaMemcpyDeviceToHost);cout<<"d_regs_: ";
    // for(int i = 0;i<3;i++){
    //     cout<<mmm[i]<<"  ";
    // }cout<<endl;
    // cudaMemcpy((void*)mmm,(void*)d_proj_,sizeof(float)*3*3,cudaMemcpyDeviceToHost);cout<<"d_proj_: ";
    // for(int i = 0;i<9;i++){
    //     cout<<mmm[i]<<"  ";
    // }cout<<endl;
    // cudaMemcpy((void*)mmm,(void*)d_ref_proj_trans,sizeof(float)*3,cudaMemcpyDeviceToHost);cout<<"d_ref_proj_trans: ";
    // for(int i = 0;i<3;i++){
    //     cout<<mmm[i]<<"  ";
    // }cout<<endl;
    
    cudaMemcpy((void*)h_regs_,(void*)d_regs_,sizeof(float)*3,cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)h_src_proj_,(void*)d_src_proj_,sizeof(float)*3*4,cudaMemcpyDeviceToHost);

    h_regs_[0] = h_src_proj_[9] - h_regs_[0];
    h_regs_[1] = h_src_proj_[10] - h_regs_[1];
    h_regs_[2] = h_src_proj_[11] - h_regs_[2];
    // cout<<"h_reg_"<<h_regs_[0]<<"  "<<h_regs_[1]<<"  "<<h_regs_[2]<<"  "<<endl;
    cudaMemcpy((void*)d_regs_,(void*)h_regs_,sizeof(float)*3,cudaMemcpyHostToDevice);

    //sub<<<1,3,0,stream>>>(d_regs_,d_src_proj_);
    
   // printf("getrottrans mul2 mat \n");
    ret = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,1,3,&alpha,intr_mat,3,d_regs_,3,&beta,trans,3);
    ret = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,3,3,&alpha,intr_mat,3,d_proj_,3,&beta,d_proj_,3);
    ret = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,3,3,3,&alpha,d_proj_,3,intr_mat_inv,3,&beta,rot,3);
    
}

template<class type0>
__global__ void row_to_column(float *out,const type0 *in){
    int Idx = threadIdx.x + blockDim.x*blockIdx.x;
    int Idy = threadIdx.y + blockDim.y*blockIdx.y;
    if(Idy == 3)
    {
        if(sizeof(type0) == sizeof(half)){
            out[Idy*3+Idx] = ((float)in[Idx*4+Idy])*1000;
        }else{
            out[Idy*3+Idx] = ((float)in[Idx*4+Idy]);
        }
    }else{
        out[Idy*3+Idx] = ((float)in[Idx*4+Idy]);
    }
}

template<class type0>
__global__ void copy_mat(float *out,const type0 *in){
    int Idx = threadIdx.x + blockDim.x*blockIdx.x;
    int Idy = threadIdx.y + blockDim.y*blockIdx.y;
    out[Idx*3+Idy] = (float)in[Idx*3+Idy];
}

void AddScalarPlugin::GetRotTrans_half(const half *intr_mat,const half *intr_mat_inv,float *rot,float *trans,
                                const half *src_proj,const half *ref_proj,float* d_src_proj_,float* d_ref_proj_,
    float* d_ref_proj_trans,
    float* d_proj_,
    float* d_regs_,
    float* src_proj_cache,
    float* ref_proj_cache,
    cudaStream_t stream){
    // int x = threadIdx.x + blockDim.x*blockIdx.x;
    // int y = threadIdx.y + blockDim.y*blockIdx.y;

    float alpha = 1;
    float beta = 0;
    dim3 block2(3,4);
    dim3 block1(3,3);
    // half *_regs_ = (half*)malloc(sizeof(half)*3);
    // half *_src_proj_ = (half*)malloc(sizeof(half)*3*4);
    //test return
    // half * mmm;
    // float * mmmf;
    // half *h_rot;
    // half *h_trans;
    // h_rot = (half*)malloc(sizeof(half)*3*3);
    // h_trans = (half*)malloc(sizeof(half)*3);

    // cudaMemcpy((void*)mmm,(void*)d_regs_,sizeof(half)*3,cudaMemcpyDeviceToHost);
    // for(int i = 0;i<3;i++){
    //     cout<<(float)mmm[i]<<"  ";
    // }cout<<endl;

    //----------

    
    cublasStatus_t ret;
    cudaError_t ret_mem;
    cublasSetStream(handle, stream);
    // printf("getrottrans mul1 mat\n");
    (row_to_column<half>)<<<1,block2,0,stream>>>(src_proj_cache,src_proj);
 

    // cudaMemcpy((void*)mmmf,(void*)src_proj_cache,sizeof(float)*3*4,cudaMemcpyDeviceToHost);cout<<"src_proj_cache:";
    // for(int i = 0;i<12;i++){
    //     cout<<mmmf[i]<<"  ";
    // }cout<<endl;

    // for(int i =0;i<12;i++){
    //     cudaMemcpy((void*)mmm,(void*)&src_proj[i],sizeof(half),cudaMemcpyDeviceToHost);
    //     cout<<"mmm:"<<*mmm<<endl;
    // }
    (row_to_column<half>)<<<1,block2,0,stream>>>(ref_proj_cache,ref_proj);

    cudaMemcpy((void*)mmmf,(void*)ref_proj_cache,sizeof(float)*3*4,cudaMemcpyDeviceToHost);cout<<"ref_proj_cache:";
    for(int i = 0;i<12;i++){
        cout<<(float)mmmf[i]<<"  ";
    }cout<<endl;

    (copy_mat<half>)<<<1,block1,0,stream_>>>(intr_mat_inv_,intr_mat_inv);
    (copy_mat<half>)<<<1,block1,0,stream_>>>(intr_mat_,intr_mat);

    ret = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,4,3,&alpha,intr_mat_inv_,3,src_proj_cache,3,&beta,d_src_proj_,3);
    if(ret != CUBLAS_STATUS_SUCCESS){cout<<"return 1:"<<ret<<endl;}
    cudaMemcpy((void*)mmmf,(void*)d_src_proj_,sizeof(float)*3*4,cudaMemcpyDeviceToHost);cout<<"d_src_proj_:";
    for(int i = 0;i<12;i++){
        cout<<mmmf[i]<<"  ";
    }cout<<endl;
    ret = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,4,3,&alpha,intr_mat_inv_,3,ref_proj_cache,3,&beta,d_ref_proj_,3);
    if(ret != CUBLAS_STATUS_SUCCESS){cout<<"return 2:"<<ret<<endl;}
    cudaMemcpy((void*)mmmf,(void*)d_ref_proj_,sizeof(float)*3*4,cudaMemcpyDeviceToHost);cout<<"d_ref_proj_:";
    for(int i = 0;i<12;i++){
        cout<<(float)mmmf[i]<<"  ";
    }cout<<endl;

    cudaMemcpy((void*)d_ref_proj_trans,(void*)&d_ref_proj_[9],sizeof(float)*3,cudaMemcpyDeviceToDevice);

    // cudaMemcpy((void*)mmm,(void*)d_ref_proj_trans,sizeof(half)*3,cudaMemcpyDeviceToHost);cout<<"d_ref_proj_trans: ";
    // for(int i = 0;i<3;i++){
    //     cout<<(float)mmm[i]<<"  ";
    // }cout<<endl;
    ret = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,3,3,3,&alpha,d_src_proj_,3,d_ref_proj_,3,&beta,d_proj_,3);
    if(ret != CUBLAS_STATUS_SUCCESS){cout<<"return 3:"<<ret<<endl;}
    
    ret = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,3,1,3,&alpha,d_proj_,3,d_ref_proj_trans,3,&beta,d_regs_,3);
    if(ret != CUBLAS_STATUS_SUCCESS){cout<<"return 4:"<<ret<<endl;}
    // cudaMemcpy((void*)mmmf,(void*)d_regs_,sizeof(float)*3,cudaMemcpyDeviceToHost);cout<<"d_regs_: ";
    // for(int i = 0;i<3;i++){
    //     cout<<(float)mmmf[i]<<"  ";
    // }cout<<endl;
    // cudaMemcpy((void*)mmmf,(void*)d_proj_,sizeof(float)*3*3,cudaMemcpyDeviceToHost);cout<<"d_proj_: ";
    // for(int i = 0;i<9;i++){
    //     cout<<(float)mmmf[i]<<"  ";
    // }cout<<endl;
    // cudaMemcpy((void*)mmmf,(void*)d_ref_proj_trans,sizeof(float)*3,cudaMemcpyDeviceToHost);cout<<"d_ref_proj_trans: ";
    // for(int i = 0;i<3;i++){
    //     cout<<(float)mmmf[i]<<"  ";
    // }cout<<endl;
    
    // cudaMemcpy((void*)_regs_,(void*)d_regs_,sizeof(half)*3,cudaMemcpyDeviceToHost);
    // cudaMemcpy((void*)_src_proj_,(void*)d_src_proj_,sizeof(half)*3*4,cudaMemcpyDeviceToHost);

    // _regs_[0] = (float)_src_proj_[9] - (float)_regs_[0];
    // _regs_[1] = (float)_src_proj_[10] - (float)_regs_[1];
    // _regs_[2] = (float)_src_proj_[11] - (float)_regs_[2];
    // cout<<"_reg_"<<(float)_regs_[0]<<"  "<<(float)_regs_[1]<<"  "<<(float)_regs_[2]<<"  "<<endl;
    // cout<<"_src_proj_"<<(float)_src_proj_[0]<<" "<<(float)_src_proj_[1]<<" "<<(float)_src_proj_[2]<<" "<<(float)_src_proj_[3]<<endl;
    // cudaMemcpy((void*)d_regs_,(void*)h_regs_,sizeof(half)*3,cudaMemcpyHostToDevice);

    (sub<float>)<<<1,3,0,stream>>>(d_regs_,d_src_proj_);
    
   // printf("getrottrans mul2 mat \n");
    ret = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,1,3,&alpha,intr_mat_,3,d_regs_,3,&beta,trans,3);
    ret = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,3,3,&alpha,intr_mat_,3,d_proj_,3,&beta,d_proj_,3);
    ret = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,3,3,3,&alpha,d_proj_,3,intr_mat_inv_,3,&beta,rot,3);
    
}
 
void message() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
 
	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		int driver_version(0), runtime_version(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor = 9999 && deviceProp.major == 9999)
				printf("\n");
 
		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
		cudaDriverGetVersion(&driver_version);
		printf("CUDA Driver Version:                            %d.%d\n", 
			driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		printf("CUDA Runtime Version:                           %d.%d\n", 
			runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("Device Prop:                                    %d.%d\n", 
			deviceProp.major, deviceProp.minor);
		printf("Total amount of Global Memory:                  %u bytes\n", 
			deviceProp.totalGlobalMem);
		printf("Number of SMs:                                  %d\n", 
			deviceProp.multiProcessorCount);
		printf("Total amount of Constant Memory:                %u bytes\n", 
			deviceProp.totalConstMem);
		printf("Total amount of Shared Memory per block:        %u bytes\n", 
			deviceProp.sharedMemPerBlock);
		printf("Total number of registers available per block:  %d\n", 
			deviceProp.regsPerBlock);
		printf("Warp size:                                      %d\n", 
			deviceProp.warpSize);
		printf("Maximum number of threads per SM:               %d\n", 
			deviceProp.maxThreadsPerMultiProcessor);
		printf("Maximum number of threads per block:            %d\n", 
			deviceProp.maxThreadsPerBlock);
		printf("Maximum size of each dimension of a block:      %d x %d x %d\n", 
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", 
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Maximum memory pitch:                           %u bytes\n", 
			deviceProp.memPitch);
		printf("Texture alignmemt:                              %u bytes\n", 
			deviceProp.texturePitchAlignment);
		printf("Clock rate:                                     %.2f GHz\n", 
			deviceProp.clockRate * 1e-6f);
		printf("Memory Clock rate:                              %.0f MHz\n", 
			deviceProp.memoryClockRate * 1e-3f);
		printf("Memory Bus Width:                               %d-bit\n", 
			deviceProp.memoryBusWidth);
	}
 
	system("pause");
}
int getThreadNum()
{
    cudaDeviceProp prop;
    int count;
 
    cudaGetDeviceCount(&count);
    printf("gpu num %d\n", count);
    cudaGetDeviceProperties(&prop, 0);
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d, %d, %d)\n",
     prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}

void AddScalarPlugin::reprojection(const float *intr_mat,const float *intr_mat_inv,float *trans,float *rot,const float *src_fea,
                            const float *src_proj,const float *ref_proj,const float *depth_samples,int width,int height,int channels,int depth_num,float *ret,
                            float *wa,float *wb,float *wc,float *wd,int *x0,int *x1,int *y0,int *y1,float *d_xyz,float *d_rot_xyz,float *depth_xyz,float* d_src_proj_,
                            float* d_ref_proj_,float* d_ref_proj_trans,float* d_proj_,float* d_regs_,float* h_src_proj_,float* h_regs_,float* src_proj_cache,float* ref_proj_cache,
                            cudaStream_t stream){

    
    // width=800,height=600,channels=16,depth_num=16;
    float alpha = 1;
    float beta = 0;
    //test code
    // float * h_mem = (float*)malloc(9*sizeof(float)); 
    // float * test_mem = (float*)malloc(channels*depth_num*height*width*sizeof(float));
    // float * test_xyz = (float*)malloc(300*sizeof(float));
    // float * h_rot = (float*)malloc(9*sizeof(float)); 
    // float * h_trans = (float*)malloc(3*sizeof(float)); 
    // float * h_src_fea = (float*)malloc(200*sizeof(float));
    // cudaMemcpy((void*)h_src_fea,(void*)src_fea,200*sizeof(float),cudaMemcpyDeviceToHost);
    // for(int i =0;i<200;i++){
    //     cout<<h_src_fea[i]<<"  ";
    // }cout<<endl;
    
    // cudaMemcpy((void*)h_mem,(void*)intr_mat,9*sizeof(float),cudaMemcpyDeviceToHost);
    
    // CHECK(cublasSetStream(handle, stream));
    // for(int i = 0;i < 9;i++){
    //     printf("%.4f  ",h_mem[i]);
    // }printf("\n");

    dim3 block1(32,32); //width,height
    dim3 grid1(width/32 + 1,height/32 + 1);
    dim3 block2(32,32); //height,width
    // dim3 grid2(width/32 + 1,height/32 + 1,depth_num);
    dim3 grid2(width/32 + 1,height/32 + 1,depth_num);

    intial_xyz<<<grid1,block1,0,stream>>>(d_xyz,width,height);
    // cudaMemcpy((void*)test_xyz,(void*)d_xyz,width*height*3*sizeof(float),cudaMemcpyDeviceToHost);
    // for(int s = 0;s<width;s++){
    //     for(int i = 0;i<height;i++){
    //         printf("xyz[%d,%d]:(%.4f,%.4f,%.4f)  ",s,i,test_xyz[s*height*3 + i*3],test_xyz[s*height*3+i*3+1],test_xyz[s*height*3+i*3+2]);
    //     }printf("\n");
    // }
    cudaStreamSynchronize(stream);
    GetRotTrans(intr_mat,intr_mat_inv,rot,trans,src_proj,ref_proj,d_src_proj_,d_ref_proj_,d_ref_proj_trans,
                d_proj_,d_regs_,h_src_proj_,h_regs_,src_proj_cache,ref_proj_cache,stream);
    // cudaMemcpy((void*)h_rot,(void*)rot,9*sizeof(float),cudaMemcpyDeviceToHost);
    // cudaMemcpy((void*)h_trans,(void*)trans,3*sizeof(float),cudaMemcpyDeviceToHost);
    // printf("rot:[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\ntrans:[%.4f,%.4f,%.4f]\n",h_rot[0],h_rot[1],h_rot[2],h_rot[3],h_rot[4],h_rot[5],h_rot[6],
    //                     h_rot[7],h_rot[8],h_trans[0],h_trans[1],h_trans[2]);
    
    // printf("reproject---3\n");
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,3,width*height,3,&alpha,rot,3,d_xyz,3,&beta,d_rot_xyz,3);

    // cudaMemcpy((void*)h_mem,(void*)d_rot_xyz,9*sizeof(float),cudaMemcpyDeviceToHost);cout<<"rot_xyz:";
    // for(int i = 0;i < 9;i++){
    //     printf("%.4f  ",h_mem[i]);
    // }printf("\n");
    // fstream file0;fstream file1;
    // float * h_depth_samples = (float*)malloc(sizeof(float)*depth_num*height*width);
    // float * h_src_fea = (float*)malloc(sizeof(float)*channels*height*width);
    // cudaMemcpy((void*)h_depth_samples,(void*)depth_samples,sizeof(float)*depth_num*height*width,cudaMemcpyDeviceToHost);
    // cudaMemcpy((void*)h_src_fea,(void*)src_fea,sizeof(float)*channels*height*width,cudaMemcpyDeviceToHost);
    // file0.open("./src_fea.txt",ios::in|ios::out|ios::trunc);
    // for (size_t s = 0; s < channels; s++){
    //     for (size_t i = 0; i < width*height; i++)
    //     {
    //         file0<<h_src_fea[s*width*height + i]<<"  ";
    //     }
    //     file0<<"<<<>>>"<<"\n";
    // }
    // file0.close();
    // file1.open("./depth_samples.txt",ios::in|ios::out|ios::trunc);
    // for (size_t s = 0; s < depth_num; s++){
    //     for (size_t i = 0; i < width*height; i++)
    //     {
    //         file1<<h_depth_samples[s*width*height + i]<<"  ";
    //     }
    //     file1<<"<<<>>>"<<"\n";
    // }
    // file1.close();
    

    (depthmulxyz<float>)<<<grid2,block2,0,stream>>>(x1 ,x0 ,y1 ,y0 ,wa,wb,wc,wd,src_fea,depth_samples,d_rot_xyz,width,height,channels,
                                            depth_num,depth_xyz,trans,ret);
    cudaStreamSynchronize(stream);

    // cudaMemcpy((void*)test_mem,(void*)ret,channels*depth_num*height*width*sizeof(float),cudaMemcpyDeviceToHost);
    // fstream file;
    // file.open("./output.txt",ios::in|ios::out|ios::trunc);
    // test_mem = &test_mem[height*width*3];
    // for(int i =0 ;i<height;i++){
    //     for(int s =0;s<width;s++){
    //         file<<'['<<i<<','<<s<<']'<<test_mem[s+i*height]<<"  ";
    //        // printf("ret[%d,%d]:%.4f ",i,s,test_mem[s+i*height]);
    //     }
    //     file<<"\n";
    //     // printf("\n");
    // }
    // file.close();

}


void AddScalarPlugin::reprojection_half(const half *intr_mat,const half *intr_mat_inv,float *trans,float *rot,const half *src_fea,
                            const half *src_proj,const half *ref_proj,const half *depth_samples,int width,int height,int channels,int depth_num,half *ret,
                            float *wa,float *wb,float *wc,float *wd,int *x0,int *x1,int *y0,int *y1,float *d_xyz,float *d_rot_xyz,float *depth_xyz,float* d_src_proj_,
                            float* d_ref_proj_,float* d_ref_proj_trans,float* d_proj_,float* d_regs_,float* src_proj_cache,float* ref_proj_cache,
                            cudaStream_t stream){

    
    // width=800,height=600,channels=16,depth_num=16;
    float alpha = 1;
    float beta = 0;
    //test code
    // half * h_mem = (half*)malloc(9*sizeof(half)); 
    //half * test_mem = (half*)malloc(channels*depth_num*height*width*sizeof(half));
    //half * test_xyz = (half*)malloc(width*height*3*sizeof(half));
    // float * h_rot = (float*)malloc(9*sizeof(float)); 
    // float * h_trans = (float*)malloc(3*sizeof(float)); 
    // half * h_src_fea = (half*)malloc(200*sizeof(half));
    // cudaMemcpy((void*)h_src_fea,(void*)src_fea,200*sizeof(half),cudaMemcpyDeviceToHost);
    // for(int i =0;i<200;i++){
    //     cout<<h_src_fea[i]<<"  ";
    // }cout<<endl;
    
    // cudaMemcpy((void*)h_mem,(void*)intr_mat,9*sizeof(half),cudaMemcpyDeviceToHost);
    
    // CHECK(cublasSetStream(handle, stream));
    // for(int i = 0;i < 9;i++){
    //     printf("%.4f  ",h_mem[i]);
    // }printf("\n");

    dim3 block1(32,32); //width,height
    dim3 grid1(width/32 + 1,height/32 + 1);
    dim3 block2(32,32); //height,width
    // dim3 grid2(width/32 + 1,height/32 + 1,depth_num);
    dim3 grid2(width/32 + 1,height/32 + 1,depth_num);

    intial_xyz<<<grid1,block1,0,stream>>>(d_xyz,width,height);

    // cudaMemcpy((void*)test_xyz,(void*)d_xyz,width*height*3*sizeof(half),cudaMemcpyDeviceToHost);
    // for(int s = 0;s<width;s++){
    //     for(int i = 0;i<height;i++){
    //         printf("xyz[%d,%d]:(%.4f,%.4f,%.4f)  ",s,i,(float)(test_xyz[s*height*3 + i*3]),
    //                 (float)(test_xyz[s*height*3+i*3+1]),(float)(test_xyz[s*height*3+i*3+2]));
    //     }printf("\n");
    // }

    cudaStreamSynchronize(stream);
    GetRotTrans_half(intr_mat,intr_mat_inv,rot,trans,src_proj,ref_proj,d_src_proj_,d_ref_proj_,d_ref_proj_trans,
                d_proj_,d_regs_,src_proj_cache,ref_proj_cache,stream);
    // cudaMemcpy((void*)h_rot,(void*)rot,9*sizeof(float),cudaMemcpyDeviceToHost);
    // cudaMemcpy((void*)h_trans,(void*)trans,3*sizeof(float),cudaMemcpyDeviceToHost);
    // printf("rot:[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\ntrans:[%.4f,%.4f,%.4f]\n",(float)h_rot[0],(float)h_rot[1],(float)h_rot[2],(float)h_rot[3],
    //                             (float)h_rot[4],(float)h_rot[5],(float)h_rot[6],
    //                     (float)h_rot[7],(float)h_rot[8],(float)(h_trans[0]),(float)(h_trans[1]),(float)(h_trans[2]));
    
    // printf("reproject---3\n");
    
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,3,width*height,3,&alpha,rot,3,d_xyz,3,&beta,d_rot_xyz,3);

    // cudaMemcpy((void*)h_mem,(void*)d_rot_xyz,9*sizeof(half),cudaMemcpyDeviceToHost);cout<<"rot_xyz:";
    // for(int i = 0;i < 9;i++){
    //     printf("%.4f  ",h_mem[i]);
    // }printf("\n");
    // fstream file0;fstream file1;
    // half * h_depth_samples = (half*)malloc(sizeof(half)*depth_num*height*width);
    // half * h_src_fea = (half*)malloc(sizeof(half)*channels*height*width);
    // cudaMemcpy((void*)h_depth_samples,(void*)depth_samples,sizeof(half)*depth_num*height*width,cudaMemcpyDeviceToHost);
    // cudaMemcpy((void*)h_src_fea,(void*)src_fea,sizeof(half)*channels*height*width,cudaMemcpyDeviceToHost);
    // file0.open("./src_fea.txt",ios::in|ios::out|ios::trunc);
    // for (size_t s = 0; s < channels; s++){
    //     for (size_t i = 0; i < width*height; i++)
    //     {
    //         file0<<h_src_fea[s*width*height + i]<<"  ";
    //     }
    //     file0<<"<<<>>>"<<"\n";
    // }
    // file0.close();
    // file1.open("./depth_samples.txt",ios::in|ios::out|ios::trunc);
    // for (size_t s = 0; s < depth_num; s++){
    //     for (size_t i = 0; i < width*height; i++)
    //     {
    //         file1<<h_depth_samples[s*width*height + i]<<"  ";
    //     }
    //     file1<<"<<<>>>"<<"\n";
    // }
    // file1.close();
    

    (depthmulxyz<half>)<<<grid2,block2,0,stream>>>(x1 ,x0 ,y1 ,y0 ,wa,wb,wc,wd,src_fea,depth_samples,d_rot_xyz,width,height,channels,
                                            depth_num,depth_xyz,trans,ret);
    cudaStreamSynchronize(stream);

    // cudaMemcpy((void*)test_mem,(void*)ret,channels*depth_num*height*width*sizeof(half),cudaMemcpyDeviceToHost);
    // fstream file;
    // file.open("./output.txt",ios::in|ios::out|ios::trunc);
    // test_mem = &test_mem[height*width*3];
    // for(int i =0 ;i<height;i++){
    //     for(int s =0;s<width;s++){
    //         file<<'['<<i<<','<<s<<']'<<(float)(test_mem[s+i*height])<<"  ";
    //        // printf("ret[%d,%d]:%.4f ",i,s,test_mem[s+i*height]);
    //     }
    //     file<<"\n";
    //     // printf("\n");
    // }
    // file.close();

}



namespace nvinfer1
{
// 这里各成员函数按照被调用顺序或重要程度顺序排列
// class AddScalarPlugin
AddScalarPlugin::AddScalarPlugin(const std::string &name):
    name_(name)
{
  
    WHERE_AM_I();
    h_src_proj_ = (float*)malloc(sizeof(float)*3*4);
    h_regs_     = (float*)malloc(sizeof(float)*3);
    cdhw        = (float*)malloc(sizeof(float)*4);
    cudaMalloc((void**)&intr_mat_inv_,sizeof(float)*9);
    cudaMalloc((void**)&intr_mat_,sizeof(float)*9);
    cudaStreamCreate(&stream_);
    CHECK(cublasCreate(&handle));
}

// AddScalarPlugin::AddScalarPlugin(const std::string &name, float scalar):
//     name_(name)
// {
//     WHERE_AM_I();
//     m_.scalar = scalar;
// }

// AddScalarPlugin::AddScalarPlugin(const std::string &name, const void *buffer, size_t length):
//     name_(name)
// {
//     WHERE_AM_I();
//     memcpy(&m_, buffer, sizeof(m_));
// }

AddScalarPlugin::~AddScalarPlugin()
{
    WHERE_AM_I();
}

IPluginV2DynamicExt *AddScalarPlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new AddScalarPlugin(name_);
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t AddScalarPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType AddScalarPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I();
    return inputTypes[0];
}

DimsExprs AddScalarPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret;

    switch (outputIndex)
    {
    case 0:
        ret.nbDims = 5;
        ret.d[0]   = inputs[1].d[0];
        ret.d[1]   = inputs[1].d[1];
        ret.d[2]   = inputs[5].d[1];
        ret.d[3]   = inputs[1].d[2];
        ret.d[4]   = inputs[1].d[3];
        // printf("input[5] %d %d %d\n",inputs[5].d[0]->getConstantValue(),inputs[5].d[1]->getConstantValue(),
        //     inputs[5].d[2]->getConstantValue(),inputs[5].d[2]->getConstantValue());
        // printf("ret %d %d %d %d %d\n",ret.d[0]->getConstantValue(),ret.d[1]->getConstantValue(),
        //     ret.d[2]->getConstantValue(),ret.d[3]->getConstantValue(),ret.d[4]->getConstantValue());
        // if(inputs[5].d[1]->getConstantValue() > 50){ //
        // ret.d[0]   = inputs[1].d[1];
        // ret.d[1]   = inputs[5].d[0];
        // ret.d[2]   = inputs[1].d[2];
        // ret.d[3]   = inputs[1].d[3];
        // printf("input[5] %d %d %d\n",inputs[5].d[0]->getConstantValue(),inputs[5].d[1]->getConstantValue(),
        //     inputs[5].d[2]->getConstantValue(),inputs[5].d[2]->getConstantValue());
        // printf("ret %d %d %d %d\n",inputs[1].d[1]->getConstantValue(),inputs[5].d[0]->getConstantValue(),
        //     inputs[1].d[2]->getConstantValue(),inputs[1].d[3]->getConstantValue());
        // }else{
        // ret.d[0]   = inputs[1].d[1];
        // ret.d[1]   = inputs[5].d[1];
        // ret.d[2]   = inputs[1].d[2];
        // ret.d[3]   = inputs[1].d[3];
        // printf("input[5] %d %d %d\n",inputs[5].d[0]->getConstantValue(),inputs[5].d[1]->getConstantValue(),
        //     inputs[5].d[2]->getConstantValue(),inputs[5].d[2]->getConstantValue());
        // printf("ret %d %d %d %d\n",inputs[1].d[1]->getConstantValue(),inputs[5].d[1]->getConstantValue(),
        //     inputs[1].d[2]->getConstantValue(),inputs[1].d[3]->getConstantValue());
        // }
        
      
        return ret;
    
    default: // should NOT be here!
        return inputs[1];
    }
    
}

bool AddScalarPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
#if 0//DEBUG
    bool res;
    switch (pos)
    {
    case 0:
        res = inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        break;
    case 1:
        res = inOut[1].format == inOut[0].format && inOut[1].type == inOut[0].type;
        break;
    default: // should NOT be here!
        res = false;
    }

    std::cout << "\tpos=" << pos << ",res=" << res << "->[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << formatToString(inOut[i].format) << ",";
    }
    std::cout << "],[";
    for (int i = 0; i < nbInputs + nbOutputs; ++i)
    {
        std::cout << dataTypeToString(inOut[i].type) << ",";
    }
    std::cout << "]" << std::endl;
    return res;
#else

    if (inOut[pos].format != TensorFormat::kLINEAR){
        printf("supportsFormatCombination [%d]format[%d] ret false1\n",pos,inOut[pos].format);
        
        return false;
    }

    //cout<<"nbInputs:"<<nbInputs<<"   nbOutputs:"<<nbOutputs<<endl;

    switch (pos)
    {
    case 0:
        return inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF || inOut[0].type == DataType::kINT8;
    case 1:
        return inOut[1].type == inOut[0].type;
    case 2:
        return inOut[2].type == inOut[0].type;
    case 3:
        return inOut[3].type == inOut[0].type;
    case 4:
        return inOut[4].type == inOut[0].type;
    case 5:
        return inOut[5].type == inOut[0].type;
    case 6:
        return inOut[6].type == inOut[0].type;
    case 7:
        return inOut[7].type == inOut[0].type;
    default: // should NOT be here!
        printf("supportsFormatCombination pos[%d] ret false2\n",pos);
        return false;
    }
    printf("supportsFormatCombination ret false3\n");
    return false;
#endif
}

void AddScalarPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
    return;
}

size_t AddScalarPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I();
    int width = 400,height = 300,channels = 32,depth_num = 32;
    size_t size = 12*sizeof(float) + sizeof(float)*width*height*3 + sizeof(float)*width*height*3 + sizeof(float)*channels*depth_num*width*height + 3*depth_num*height*width*sizeof(float)
                    + depth_num*height*width*sizeof(int)*4 + depth_num*height*width*sizeof(float)*4 + sizeof(float)*3*4 + sizeof(float)*3*4 + sizeof(float)*3*3 + sizeof(float)*3
                    + sizeof(float)*3 + sizeof(float)*3*4 + sizeof(float)*3*4 + sizeof(float)*12*2;

    return size;
}

int32_t AddScalarPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
  


    if (inputDesc[0].type == DataType::kFLOAT){
          // int32_t inputs_offset[7] = {1,1,1,1,1,1,1};
    for(int s=0;s<7;s++){
    printf("input[%d] nbDims:%d\n",s,inputDesc[s].dims.nbDims);
    printf("input[%d] format:%d\n",s,inputDesc[s].format);
    printf("input[%d] type:%d\n",s,inputDesc[s].type);
    for(int i = 0;i < inputDesc[s].dims.nbDims;i++)
    {
        
        printf("idex[%d] long:%d\n",i,inputDesc[s].dims.d[i]);
    }
    // printf("%d-------------\n",inputs_offset[s]);
    }
    printf("output nbDims:%d\n",outputDesc[0].dims.nbDims);
    printf("output format:%d\n",outputDesc[0].format);
    printf("output type:%d\n",outputDesc[0].type);
    for(int i = 0;i < outputDesc[0].dims.nbDims;i++)
    {
        
        printf("idex[%d] long:%d\n",i,outputDesc[0].dims.d[i]);
        
    }
    
   
    
    
    cudaMemcpy((void*)cdhw,reinterpret_cast<const void *>(inputs[6]),sizeof(float)*4,cudaMemcpyDeviceToHost);
    int offset = 0;
    const float * intptr = (float *)inputs[0];

    char * cachespace = (char*)workspace;

    int  channels = (int)cdhw[0];
    int depth_num = (int)cdhw[1];
    int  height = (int)cdhw[2];
    int  width = (int)cdhw[3];
    printf("c:%d  d:%d  h:%d  w:%d\n",channels,depth_num,height,width);

    float * trans = (float*)&cachespace[offset];offset += 3*sizeof(float); 
    float * rot   = (float*)&cachespace[offset];offset += 9*sizeof(float);
    float * wa = (float*)&cachespace[offset];offset += depth_num*height*width*sizeof(float);
    float * wb = (float*)&cachespace[offset];offset += depth_num*height*width*sizeof(float);
    float * wc = (float*)&cachespace[offset];offset += depth_num*height*width*sizeof(float);
    float * wd = (float*)&cachespace[offset];offset += depth_num*height*width*sizeof(float);
    int   * x0 = (int  *)&cachespace[offset];offset += depth_num*height*width*sizeof(int);
    int   * x1 = (int  *)&cachespace[offset];offset += depth_num*height*width*sizeof(int);
    int   * y0 = (int  *)&cachespace[offset];offset += depth_num*height*width*sizeof(int);
    int   * y1 = (int  *)&cachespace[offset];offset += depth_num*height*width*sizeof(int);
    float * d_xyz = (float*)&cachespace[offset];offset += sizeof(float)*width*height*3;
    float * d_rot_xyz        = (float*)&cachespace[offset];offset += sizeof(float)*width*height*3;
    float * depth_xyz        = (float*)&cachespace[offset];offset += 3*depth_num*height*width*sizeof(float);
    float * d_src_proj       = (float*)&cachespace[offset];offset += sizeof(float)*3*4;
    float * d_ref_proj_      = (float*)&cachespace[offset];offset += sizeof(float)*3*4;
    float * d_proj_          = (float*)&cachespace[offset];offset += sizeof(float)*3*3;
    float * d_regs_          = (float*)&cachespace[offset];offset += sizeof(float)*3;
    float * d_ref_proj_trans = (float*)&cachespace[offset];offset += sizeof(float)*3;
    float * src_proj_cache   = (float*)&cachespace[offset];offset += sizeof(float)*3*4;
    float * ref_proj_cache   = (float*)&cachespace[offset];offset += sizeof(float)*3*4*2;

    // float * h_wa = (float*)malloc(sizeof(float)*depth_num*height*width);
    // float * h_wb = (float*)malloc(sizeof(float)*depth_num*height*width);
    // float * h_wc = (float*)malloc(sizeof(float)*depth_num*height*width);

    // float * h_src_proj_;
    // float * h_regs_;
    // h_src_proj_ = (float*)malloc(sizeof(float)*3*4);
    // h_regs_     = (float*)malloc(sizeof(float)*3);
        std::cout << "Run float\n";
        reprojection(reinterpret_cast<const float *>(inputs[0]),reinterpret_cast<const float *>(inputs[2]),trans,rot,reinterpret_cast<const float *>(inputs[1]),
                    reinterpret_cast<const float *>(inputs[3]),reinterpret_cast<const float *>(inputs[4]),reinterpret_cast<const float *>(inputs[5]),width
                    ,height,channels,depth_num,(float*)outputs[0],wa,wb,wc,wd,x0,x1,y0,y1,
                    d_xyz,d_rot_xyz,depth_xyz,d_src_proj,d_ref_proj_,d_ref_proj_trans,d_proj_,d_regs_,
                    h_src_proj_,h_regs_,src_proj_cache,ref_proj_cache, stream);
    }
    else if (inputDesc[0].type == DataType::kHALF){
          // int32_t inputs_offset[7] = {1,1,1,1,1,1,1};
    for(int s=0;s<7;s++){
    printf("input[%d] nbDims:%d\n",s,inputDesc[s].dims.nbDims);
    printf("input[%d] format:%d\n",s,inputDesc[s].format);
    printf("input[%d] type:%d\n",s,inputDesc[s].type);
    for(int i = 0;i < inputDesc[s].dims.nbDims;i++)
    {
        
        printf("idex[%d] long:%d\n",i,inputDesc[s].dims.d[i]);
    }
    // printf("%d-------------\n",inputs_offset[s]);
    }
    printf("output nbDims:%d\n",outputDesc[0].dims.nbDims);
    printf("output format:%d\n",outputDesc[0].format);
    printf("output type:%d\n",outputDesc[0].type);
    for(int i = 0;i < outputDesc[0].dims.nbDims;i++)
    {
        
        printf("idex[%d] long:%d\n",i,outputDesc[0].dims.d[i]);
        
    }
    half cdhwH[4];
    
    cudaMemcpy((void*)cdhwH,reinterpret_cast<const void *>(inputs[6])  ,sizeof(half)*4,cudaMemcpyDeviceToHost);

    int offset = 0;

    char * cachespace = (char*)workspace;
    
    int  channels = (int)cdhwH[0];
    int depth_num = (int)cdhwH[1];
    int  height = (int)cdhwH[2];
    int  width = (int)cdhwH[3];
    printf("c:%d  d:%d  h:%d  w:%d \n",channels,depth_num,height,width);

    float * trans = (float*)&cachespace[offset];offset += 3*sizeof(float); 
    float * rot   = (float*)&cachespace[offset];offset += 9*sizeof(float);
    float * wa = (float*)&cachespace[offset];offset += depth_num*height*width*sizeof(float);
    float * wb = (float*)&cachespace[offset];offset += depth_num*height*width*sizeof(float);
    float * wc = (float*)&cachespace[offset];offset += depth_num*height*width*sizeof(float);
    float * wd = (float*)&cachespace[offset];offset += depth_num*height*width*sizeof(float);
    int   * x0 = (int  *)&cachespace[offset];offset += depth_num*height*width*sizeof(int);
    int   * x1 = (int  *)&cachespace[offset];offset += depth_num*height*width*sizeof(int);
    int   * y0 = (int  *)&cachespace[offset];offset += depth_num*height*width*sizeof(int);
    int   * y1 = (int  *)&cachespace[offset];offset += depth_num*height*width*sizeof(int);
    float * d_xyz = (float*)&cachespace[offset];offset += sizeof(float)*width*height*3;
    float * d_rot_xyz        = (float*)&cachespace[offset];offset += sizeof(float)*width*height*3;
    float * depth_xyz        = (float*)&cachespace[offset];offset += 3*depth_num*height*width*sizeof(float);
    float * d_src_proj       = (float*)&cachespace[offset];offset += sizeof(float)*3*4;
    float * d_ref_proj_      = (float*)&cachespace[offset];offset += sizeof(float)*3*4;
    float * d_proj_          = (float*)&cachespace[offset];offset += sizeof(float)*3*3;
    float * d_regs_          = (float*)&cachespace[offset];offset += sizeof(float)*3;
    float * d_ref_proj_trans = (float*)&cachespace[offset];offset += sizeof(float)*3;
    float * src_proj_cache   = (float*)&cachespace[offset];offset += sizeof(float)*3*4;
    float * ref_proj_cache   = (float*)&cachespace[offset];offset += sizeof(float)*3*4*2;

    // half * h_wa = (half*)malloc(sizeof(half)*depth_num*height*width);
    // half * h_wb = (half*)malloc(sizeof(half)*depth_num*height*width);
    // half * h_wc = (half*)malloc(sizeof(half)*depth_num*height*width);

    // half * h_src_proj_;
    // half * h_regs_;
    // h_src_proj_ = (half*)malloc(sizeof(half)*3*4);
    // h_regs_     = (half*)malloc(sizeof(short)*3);
        std::cout << "Run half\n";
        reprojection_half(reinterpret_cast<const half *>(inputs[0]),reinterpret_cast<const half *>(inputs[2]),trans,rot,reinterpret_cast<const half *>(inputs[1]),
                    reinterpret_cast<const half *>(inputs[3]),reinterpret_cast<const half *>(inputs[4]),reinterpret_cast<const half *>(inputs[5]),width
                    ,height,channels,depth_num,(half*)outputs[0],wa,wb,wc,wd,x0,x1,y0,y1,
                    d_xyz,d_rot_xyz,depth_xyz,d_src_proj,d_ref_proj_,d_ref_proj_trans,d_proj_,d_regs_,
                    src_proj_cache,ref_proj_cache, stream);
    }
    else if (inputDesc[0].type == DataType::kINT8){
        std::cout << "Run int8\n";
        // reprojection(reinterpret_cast<const float *>(inputs[0]),reinterpret_cast<const float *>(inputs[2]),trans,rot,reinterpret_cast<const float *>(inputs[1]),
        //             reinterpret_cast<const float *>(inputs[3]),reinterpret_cast<const float *>(inputs[4]),reinterpret_cast<const float *>(inputs[5]),width
        //             ,height,channels,depth_num,(float*)outputs[0],wa,wb,wc,wd,x0,x1,y0,y1,
        //             d_xyz,d_rot_xyz,depth_xyz,d_src_proj,d_ref_proj_,d_ref_proj_trans,d_proj_,d_regs_,
        //             h_src_proj_,h_regs_,src_proj_cache,ref_proj_cache, stream);
    }
    // cudaMemcpy((void*)h_wa,(void*)wa,sizeof(float)*depth_num*height*width,cudaMemcpyDeviceToHost);
    // cudaMemcpy((void*)h_wb,(void*)wb,sizeof(float)*depth_num*height*width,cudaMemcpyDeviceToHost);
    // cudaMemcpy((void*)h_wc,(void*)wc,sizeof(float)*depth_num*height*width,cudaMemcpyDeviceToHost);
    // fstream file1;
    // file1.open("./outputx.txt",ios::in|ios::out|ios::trunc);
    // fstream file2;
    // file2.open("./outputy.txt",ios::in|ios::out|ios::trunc);
    // fstream file3;
    // file3.open("./outputz.txt",ios::in|ios::out|ios::trunc);
    // printf("----------------y--------------\n");
    // for (size_t i = 0; i < depth_num; i++)
    // {
    //     for (size_t s = 0; s < height*width; s++)
    //     {
    //         file1<<" ["<< s <<"]"<<h_wa[height*width*i + s]<<" ";
    //         //printf("%.4f ",h_wa[height*width*i + s]);
    //     }
    //     file1<<"<<<>>>\n";
    //     //printf("\n");
        
    // }
    // printf("----------------x--------------\n");
    // for (size_t i = 0; i < depth_num; i++)
    // {
    //     for (size_t s = 0; s < height*width; s++)
    //     {
    //         file2<<" ["<< s <<"]"<<h_wb[height*width*i + s]<<" ";
    //         //printf("%.4f ",h_wb[height*width*i + s]);
    //     }
    //     file2<<"<<<>>>\n";
    //     //printf("\n");
        
    // }
    // printf("----------------z--------------\n");
    // for (size_t i = 0; i < depth_num; i++)
    // {
    //     for (size_t s = 0; s < height*width; s++)
    //     {
    //         file3<<" ["<< s <<"]"<<h_wc[height*width*i + s]<<" ";
    //         //printf("%.4f ",h_wb[height*width*i + s]);
    //     }
    //     file3<<"<<<>>>\n";
    //     //printf("\n");
        
    // }
    // file1.close();
    // file2.close();
    // file3.close();

    // std::cout << "engine ret---\n";
    return 0;
}

void AddScalarPlugin::destroy() noexcept
{
    WHERE_AM_I();
    free(h_src_proj_);
    free(h_regs_);
    free(cdhw);
    cudaFree(intr_mat_inv_);
    cudaFree(intr_mat_);
    cublasDestroy(handle);
    return;
}

int32_t AddScalarPlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void AddScalarPlugin::terminate() noexcept
{
    WHERE_AM_I();
    return;
}

size_t AddScalarPlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return 0;//sizeof(m_);
}

void AddScalarPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I();
    //memcpy(buffer, &m_, sizeof(m_));
    return;
}

void AddScalarPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *AddScalarPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *AddScalarPlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *AddScalarPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

void AddScalarPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
{
    WHERE_AM_I();
    return;
}

void AddScalarPlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
    return;
}

// class AddScalarPluginCreator
PluginFieldCollection    AddScalarPluginCreator::fc_ {};
std::vector<PluginField> AddScalarPluginCreator::attr_;

AddScalarPluginCreator::AddScalarPluginCreator()
{
    WHERE_AM_I();
    attr_.clear();
    attr_.emplace_back(PluginField("scalar", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

AddScalarPluginCreator::~AddScalarPluginCreator()
{
    WHERE_AM_I();
}

// 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
IPluginV2 *AddScalarPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    WHERE_AM_I();
    // float                          scalar = 0;
    // std::map<std::string, float *> parameterMap {{"scalar", &scalar}};

    // for (int i = 0; i < fc->nbFields; ++i)
    // {
    //     if (parameterMap.find(fc->fields[i].name) != parameterMap.end())
    //     {
    //         *parameterMap[fc->fields[i].name] = *reinterpret_cast<const float *>(fc->fields[i].data);
    //     }
    // }
    AddScalarPlugin *pObj = new AddScalarPlugin(name);//, scalar);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

IPluginV2 *AddScalarPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    AddScalarPlugin *pObj = new AddScalarPlugin(name);//, serialData, serialLength);
    pObj->setPluginNamespace(namespace_.c_str());
    return pObj;
}

void AddScalarPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = pluginNamespace;
    return;
}

const char *AddScalarPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char *AddScalarPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_NAME;
}

const char *AddScalarPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return PLUGIN_VERSION;
}

const PluginFieldCollection *AddScalarPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(AddScalarPluginCreator);

} // namespace nvinfer1