struct Vec3 {
    float x;
    float y;
    float z;
};

extern "C" __global__ void add(const float *x, const float *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = x[i] + y[i];
    }
}

extern "C" __global__ void sub(const float *x, const float *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = x[i] - y[i];
    }
}

extern "C" __global__ void mul(const float *x, const float *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = x[i] * y[i];
    }
}

extern "C" __global__ void divide(const float *x, const float *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = x[i] / y[i];
    }
}

extern "C" __global__ void vec3_add(const Vec3 *x, const Vec3 *y, Vec3 *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i].x = x[i].x + y[i].x;
        out[i].y = x[i].y + y[i].y;
        out[i].z = x[i].z + y[i].z;
    }
}

extern "C" __global__ void vec3_sub(const Vec3 *x, const Vec3 *y, Vec3 *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i].x = x[i].x - y[i].x;
        out[i].y = x[i].y - y[i].y;
        out[i].z = x[i].z - y[i].z;
    }
}

extern "C" __global__ void vec3_mul_scalar(const Vec3 *v, const float *scalar, Vec3 *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i].x = v[i].x * scalar[i];
        out[i].y = v[i].y * scalar[i];
        out[i].z = v[i].z * scalar[i];
    }
}

extern "C" __global__ void vec3_div_scalar(const Vec3 *v, const float *scalar, Vec3 *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i].x = v[i].x / scalar[i];
        out[i].y = v[i].y / scalar[i];
        out[i].z = v[i].z / scalar[i];
    }
}

extern "C" __global__ void vec3_len(const Vec3 *v, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = sqrtf(v[i].x*v[i].x + v[i].y*v[i].y + v[i].z*v[i].z);
    }
}

extern "C" __global__ void vec3_normalize(const Vec3 *v, Vec3 *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float len = 0;
    for(int i=start_idx; i<count; i += stride) {
        len = sqrtf(v[i].x*v[i].x + v[i].y*v[i].y + v[i].z*v[i].z);
        out[i] = {v[i].x / len, v[i].y / len, v[i].z / len};
    }
}

extern "C" __global__ void vec3_get_x(const Vec3 *v, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = v[i].x;
    }
}

extern "C" __global__ void vec3_get_y(const Vec3 *v, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = v[i].y;
    }
}

extern "C" __global__ void vec3_get_z(const Vec3 *v, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = v[i].z;
    }
}