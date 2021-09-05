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

extern "C" __global__ void float_max(const float *x, const float *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = max(x[i], y[i]);
    }
}

extern "C" __global__ void float_min(const float *x, const float *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = min(x[i], y[i]);
    }
}

extern "C" __global__ void inv(const float *x, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = -x[i];
    }
}

extern "C" __global__ void float_sqrt(const float *x, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = sqrtf(x[i]);
    }
}

extern "C" __global__ void float_pow(const float *x, const float *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = powf(x[i], y[i]);
    }
}

extern "C" __global__ void is_positive(const float *x, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = max(x[i], 0.0f) / x[i];
    }
}

extern "C" __global__ void is_negative(const float *x, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = 1.0f - (max(x[i], 0.0f) / x[i]);
    }
}

extern "C" __global__ void bool_and(const float *x, const float *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = x[i] * y[i];
    }
}

extern "C" __global__ void bool_or(const float *x, const float *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = max(x[i], y[i]);       // 0 iff x[i] and x[i] both 0
    }
}

extern "C" __global__ void bool_not(const float *x, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = 1.0f - x[i];
    }
}

extern "C" __global__ void float_select(const float *flag, const float *when_true, const float *when_false, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = flag[i]*when_true[i] + (1.0f - flag[i])*when_false[i];
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

extern "C" __global__ void vec3_dot(const Vec3 *x, const Vec3 *y, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = x[i].x*y[i].x + x[i].y*y[i].y + x[i].z*y[i].z;
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

extern "C" __global__ void vec3_len_squared(const Vec3 *v, float *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = v[i].x*v[i].x + v[i].y*v[i].y + v[i].z*v[i].z;
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

extern "C" __global__ void vec3_inv(const Vec3 *v, Vec3 *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = {-v[i].x, -v[i].y, -v[i].z};
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

extern "C" __global__ void vec3_select(const float *flag, const Vec3 *when_true, const Vec3 *when_false, Vec3 *out, int count) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=start_idx; i<count; i += stride) {
        out[i] = {
            flag[i]*when_true[i].x + (1.0f - flag[i])*when_false[i].x,
            flag[i]*when_true[i].y + (1.0f - flag[i])*when_false[i].y,
            flag[i]*when_true[i].z + (1.0f - flag[i])*when_false[i].z
        };
    }
}
