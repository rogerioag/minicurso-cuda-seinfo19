#include <stdio.h>
#include "debug.h"

/* Tipo para o ponteiro de função. */
typedef int (*op_func) (void);

//---------------------------------------
// 1D_ (grid) -> Only Block executions.
//---------------------------------------
// gx > 1 -> (32,1,1)(1,1,1)
__device__ int getGlobalIdx_grid_1D_x() {
    PRINT_FUNC_NAME;
    return blockIdx.x;
}

// gy > 1 -> (1,32,1)(1,1,1)
__device__ int getGlobalIdx_grid_1D_y() {
    PRINT_FUNC_NAME;
    return blockIdx.y;
}

// gz > 1 -> (1,1,32)(1,1,1)
__device__ int getGlobalIdx_grid_1D_z() {
    PRINT_FUNC_NAME;
    return blockIdx.z;
}

//---------------------------------------
// _1D (block) -> Only Threads execution.
//---------------------------------------
// bx > 1 -> (1,1,1)(32,1,1)
__device__ int getGlobalIdx_block_1D_x() {
    PRINT_FUNC_NAME;
    return threadIdx.x;
}

// by > 1 -> (1,1,1)(1,32,1)
__device__ int getGlobalIdx_block_1D_y() {
    PRINT_FUNC_NAME;
    return threadIdx.y;
}

// bz > 1 -> (1,1,1)(1,1,32)
__device__ int getGlobalIdx_block_1D_z() {
    PRINT_FUNC_NAME;
    return threadIdx.z;
}

//---------------------------------------
// 2D_ (grid) -> Only Block execution.
//---------------------------------------
// gx,gy > 1 -> (32,32,1)(1,1,1)
__device__ int getGlobalIdx_grid_2D_xy() {
    PRINT_FUNC_NAME;
    return blockIdx.x + blockIdx.y * gridDim.x;
}

// gx,gz > 1 -> (32,1,32)(1,1,1)
__device__ int getGlobalIdx_grid_2D_xz() {
    PRINT_FUNC_NAME;
    return blockIdx.x + blockIdx.z * gridDim.x;
}

// gy,gz > 1 -> (1,32,32)(1,1,1)
__device__ int getGlobalIdx_grid_2D_yz() {
    PRINT_FUNC_NAME;
    return blockIdx.y + blockIdx.z * gridDim.y;
}

//---------------------------------------
// _2D (block) -> Only Threads Execution.
//---------------------------------------
// bx,by > 1 -> (1,1,1)(32,32,1)
__device__ int getGlobalIdx_block_2D_xy() {
    PRINT_FUNC_NAME;
    return threadIdx.x + threadIdx.y * blockDim.x;
}

// bx,bz > 1 -> (1,1,1)(32,1,32)
__device__ int getGlobalIdx_block_2D_xz() {
    PRINT_FUNC_NAME;
    return threadIdx.x + threadIdx.z * blockDim.x;
}

// by,bz > 1 -> (1,1,1)(1,32,32)
__device__ int getGlobalIdx_block_2D_yz() {
    PRINT_FUNC_NAME;
    return threadIdx.y + threadIdx.z * blockDim.y;
}

//---------------------------------------
// 3D_ (grid) -> Only Block Execution.
//---------------------------------------
// gx,gy,gz > 1 -> (32,32,32)(1,1,1)
__device__ int getGlobalIdx_grid_3D_xyz() {
    PRINT_FUNC_NAME;
    return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

//---------------------------------------
// _3D (block) -> Only Threads Execution.
//---------------------------------------
// bx,by,bz > 1 -> (1,1,1)(32,32,32)
__device__ int getGlobalIdx_block_3D_xyz() {
    PRINT_FUNC_NAME;
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

//---------------------------------------
// 1D_1D (grid and block) -> Grid and Block combination.
//---------------------------------------
// gx e bx > 1 -> (32,1,1)(32,1,1)
__device__ int getGlobalIdx_grid_1D_x_block_1D_x() {
    PRINT_FUNC_NAME;
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// gx e by > 1 -> (32,1,1)(1,32,1)
__device__ int getGlobalIdx_grid_1D_x_block_1D_y() {
    PRINT_FUNC_NAME;
    return blockIdx.x * blockDim.y + threadIdx.y;
}

// gx e bz > 1 -> (32,1,1)(1,1,32)
__device__ int getGlobalIdx_grid_1D_x_block_1D_z() {
    PRINT_FUNC_NAME;
    return blockIdx.x * blockDim.z + threadIdx.z;
}

// gy e bx > 1 -> (1,32,1)(32,1,1)
__device__ int getGlobalIdx_grid_1D_y_block_1D_x() {
    PRINT_FUNC_NAME;
    return blockIdx.y * blockDim.x + threadIdx.x;
}

// gy e by > 1 -> (1,32,1)(1,32,1)
__device__ int getGlobalIdx_grid_1D_y_block_1D_y() {
    PRINT_FUNC_NAME;
    return blockIdx.y * blockDim.y + threadIdx.y;
}

// gy e bz > 1 -> (1,32,1)(1,1,32)
__device__ int getGlobalIdx_grid_1D_y_block_1D_z() {
    PRINT_FUNC_NAME;
    return blockIdx.y * blockDim.z + threadIdx.z;
}

// gz e bx > 1 -> (1,1,32)(32,1,1)
__device__ int getGlobalIdx_grid_1D_z_block_1D_x() {
    PRINT_FUNC_NAME;
    return blockIdx.z * blockDim.x + threadIdx.x;
}

// gz e by > 1 -> (1,1,32)(1,32,1)
__device__ int getGlobalIdx_grid_1D_z_block_1D_y() {
    PRINT_FUNC_NAME;
    return blockIdx.z * blockDim.y + threadIdx.y;
}

// gz e bz > 1 -> (1,1,32)(1,1,32)
__device__ int getGlobalIdx_grid_1D_z_block_1D_z() {
    PRINT_FUNC_NAME;
    return blockIdx.z * blockDim.z + threadIdx.z;
}

//---------------------------------------
// 1D_2D
//---------------------------------------
// gx e bx,by > 1 -> (32,1,1)(32,32,1)
//                    Dz      Dx Dy
//                     z       x  y
//   id = x + y . Dx + z . Dx . Dy 
__device__ int getGlobalIdx_grid_1D_x_block_2D_xy() {
    PRINT_FUNC_NAME;
    // return blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y;
    return threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
}

// gx e bx,bz > 1 -> (32,1,1)(32,1,32)
//                    Dz      Dx   Dy
//                     z       x    y
//   id = x + y . Dx + z . Dx . Dy 
__device__ int getGlobalIdx_grid_1D_x_block_2D_xz() {
    PRINT_FUNC_NAME;
    // return blockDim.x * blockDim.z * blockIdx.x + threadIdx.x + threadIdx.z;
    return threadIdx.x + threadIdx.z * blockDim.x + blockIdx.x * blockDim.x * blockDim.z;
}
                       
// gx e by,bz > 1 -> (32,1,1)(1,32,32)
//                    Dz        Dx Dy
//                     z         x  y
//   id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_1D_x_block_2D_yz() {
    PRINT_FUNC_NAME;
    // return blockDim.y * blockDim.z * blockIdx.x + threadIdx.y + threadIdx.z;
    return threadIdx.y + threadIdx.z * blockDim.y + blockIdx.x * blockDim.y * blockDim.z;
}

// gy e bx,by > 1 -> (1,32,1)(32,32,1)
__device__ int getGlobalIdx_grid_1D_y_block_2D_xy() {
    PRINT_FUNC_NAME;
    // return blockDim.x * blockDim.y * blockIdx.y + threadIdx.x + threadIdx.y;
    return threadIdx.x + threadIdx.y * blockDim.x + blockIdx.y * blockDim.x * blockDim.y;
}

// gy e bx,bz > 1 -> (1,32,1)(32,1,32)
__device__ int getGlobalIdx_grid_1D_y_block_2D_xz() {
    PRINT_FUNC_NAME;
    // return blockDim.x * blockDim.z * blockIdx.y + threadIdx.x + threadIdx.z;
    return threadIdx.x + threadIdx.z * blockDim.x + blockIdx.y * blockDim.x * blockDim.z;
}

// gy e by,bz > 1 -> (1,32,1)(1,32,32)
__device__ int getGlobalIdx_grid_1D_y_block_2D_yz() {
    PRINT_FUNC_NAME;
    //return blockDim.y * blockDim.z * blockIdx.y + threadIdx.y + threadIdx.z;
    return threadIdx.y + threadIdx.z * blockDim.y + blockIdx.y * blockDim.y * blockDim.z;
}

// gz e bx,by > 1 -> (1,1,32)(32,32,1)
__device__ int getGlobalIdx_grid_1D_z_block_2D_xy() {
    PRINT_FUNC_NAME;
    // return blockDim.x * blockDim.y * blockIdx.z + threadIdx.x + threadIdx.y;
    return threadIdx.x + threadIdx.y * blockDim.x + blockIdx.z * blockDim.x * blockDim.y;
}

// gz e bx,bz > 1 -> (1,1,32)(32,1,32)
__device__ int getGlobalIdx_grid_1D_z_block_2D_xz() {
    // TODO .
    // return blockDim.x * blockDim.z * blockIdx.z + threadIdx.x + threadIdx.z;
    return threadIdx.x + threadIdx.z * blockDim.x + blockIdx.z * blockDim.x * blockDim.z;
}

// gz e by,bz > 1 -> (1,1,32)(1,32,32)
__device__ int getGlobalIdx_grid_1D_z_block_2D_yz() {
    PRINT_FUNC_NAME;
    // return blockDim.y * blockDim.z * blockIdx.z + threadIdx.y + threadIdx.z;
    return threadIdx.y + threadIdx.z * blockDim.y + blockIdx.z * blockDim.y * blockDim.z;
}

//---------------------------------------
// 1D_3D
//---------------------------------------
// gx e bx,by,bz > 1 -> (32,1,1)(16,32,2)
//                       Dw      Dx Dy Dz
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_1D_x_block_3D_xyz() {
    PRINT_FUNC_NAME;
    return threadIdx.x 
           + threadIdx.y * blockDim.x 
           + threadIdx.z * blockDim.x * blockDim.y 
           + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
}

// gy e bx,by,bz > 1 -> (1,32,1)(16,32,2)
//                         Dw    Dx Dy Dz
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_1D_y_block_3D_xyz() {
    PRINT_FUNC_NAME;
    return threadIdx.x 
           + threadIdx.y * blockDim.x 
           + threadIdx.z * blockDim.x * blockDim.y 
           + blockIdx.y * blockDim.x * blockDim.y * blockDim.z;
}

// gz e bx,by,bz > 1 -> (1,1,32)(16,32,2)
//                           Dw  Dx Dy Dz
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_1D_z_block_3D_xyz() {
    PRINT_FUNC_NAME;
    return threadIdx.x 
           + threadIdx.y * blockDim.x 
           + threadIdx.z * blockDim.x * blockDim.y 
           + blockIdx.z * blockDim.x * blockDim.y * blockDim.z;
}

//---------------------------------------
// 2D_1D
//---------------------------------------
// gx,gy e bx > 1 -> (32,32,1)(32,1,1)
//                    Dy Dz    Dx
//  id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_2D_xy_block_1D_x() {
    PRINT_FUNC_NAME;
    // int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    // int threadId = blockId * blockDim.x + threadIdx.x;
    // return threadId;
    // Expandindo chega nisso.

    return threadIdx.x
           + blockIdx.x * blockDim.x
           + blockIdx.y * gridDim.x * blockDim.x;
}

// gx,gy e by > 1 -> (32,32,1)(1,32,1)
//                    Dy Dz      Dx
//  id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_2D_xy_block_1D_y() {
    PRINT_FUNC_NAME;
    // int blockId = blockIdx.y * gridDim.y + blockIdx.x;
    // int threadId = blockId * blockDim.y + threadIdx.y;
    // return threadId;

    return threadIdx.y
           + blockIdx.x * blockDim.y
           + blockIdx.y * gridDim.x * blockDim.y;
}

// gx,gy e bz > 1 -> (32,32,1)(1,1,32)
//                    Dy Dz        Dx
//  id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_2D_xy_block_1D_z() {
    PRINT_FUNC_NAME;
    return threadIdx.z
           + blockIdx.x * blockDim.z
           + blockIdx.y * gridDim.x * blockDim.z;
}

// gx,gz e bx > 1 -> (32,1,32)(32,1,1)
//                    Dy   Dz  Dx
//  id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_2D_xz_block_1D_x() {
    PRINT_FUNC_NAME;
    return threadIdx.x
           + blockIdx.x * blockDim.x
           + blockIdx.z * gridDim.x * blockDim.x;
}

// gx,gz e by > 1 -> (32,1,32)(1,32,1)
//                    Dy   Dz    Dx
//  id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_2D_xz_block_1D_y() {
    PRINT_FUNC_NAME;
    return threadIdx.y
           + blockIdx.x * blockDim.y
           + blockIdx.z * gridDim.x * blockDim.y;
}

// gx,gz e bz > 1 -> (32,1,32)(1,1,32)
//                    Dy   Dz      Dx
//  id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_2D_xz_block_1D_z() {
    PRINT_FUNC_NAME;
    return threadIdx.z
           + blockIdx.x * blockDim.z
           + blockIdx.z * gridDim.x * blockDim.z;
}

// gy,gz e bx > 1 -> (1,32,32)(32,1,1)
//                      Dy Dz  Dx
//  id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_2D_yz_block_1D_x() {
    PRINT_FUNC_NAME;
    return threadIdx.x
           + blockIdx.y * blockDim.x
           + blockIdx.z * gridDim.y * blockDim.x;
}

// gy,gz e by > 1 -> (1,32,32)(1,32,1)
//                      Dy Dz    Dx
//  id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_2D_yz_block_1D_y() {
    PRINT_FUNC_NAME;
    return threadIdx.y
           + blockIdx.y * blockDim.y
           + blockIdx.z * gridDim.y * blockDim.y;
}

// gy,gz e bz > 1 -> (1,32,32)(1,1,32)
//                      Dy Dz      Dx
//  id = x + y . Dx + z . Dx . Dy
__device__ int getGlobalIdx_grid_2D_yz_block_1D_z() {
    PRINT_FUNC_NAME;
    return threadIdx.z
           + blockIdx.y * blockDim.z
           + blockIdx.z * gridDim.y * blockDim.z;
}

//---------------------------------------
// 2D_2D
//---------------------------------------
// gx,gy e bx,by > 1 -> (32,32,1)(32,32,1)
//                       Dz Dw    Dx Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_2D_xy_block_2D_xy() {
    PRINT_FUNC_NAME;
    // 9 operações.
    /*return threadIdx.x 
           + threadIdx.y * blockDim.x
           + blockIdx.x + blockDim.x * blockDim.y
           + blockIdx.y * blockDim.x * blockDim.y * gridDim.x;*/

    // 7 operações com o valor intermediário para o blockId.
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x + (threadIdx.y * blockDim.x)
                   + blockId * (blockDim.x * blockDim.y);
    return threadId;
    
}

// gx,gy e bx,bz > 1 -> (32,32,1)(32,1,32)
//                       Dz Dw    Dx   Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_2D_xy_block_2D_xz() {
    PRINT_FUNC_NAME;

    /*return threadIdx.x 
           + threadIdx.z * blockDim.x
           + blockIdx.x + blockDim.x * blockDim.z
           + blockIdx.y * blockDim.x * blockDim.z * gridDim.x;*/

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x + (threadIdx.z * blockDim.x)
                   + blockId * (blockDim.x * blockDim.z);
    return threadId;  
}

// gx,gy e by,bz > 1 -> (32,32,1)(1,32,32)
//                       Dz Dw      Dx Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_2D_xy_block_2D_yz() {
    PRINT_FUNC_NAME;
    /*return threadIdx.y 
           + threadIdx.z * blockDim.y
           + blockIdx.x + blockDim.y * blockDim.z
           + blockIdx.y * blockDim.y * blockDim.z * gridDim.x;*/

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.y + (threadIdx.z * blockDim.x)
                   + blockId * (blockDim.y * blockDim.z);
    return threadId;
}

// gx,gz e bx,by > 1 -> (32,1,32)(32,32,1)
//                       Dz   Dw  Dx Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_2D_xz_block_2D_xy() {
    PRINT_FUNC_NAME;
    /*return threadIdx.x 
           + threadIdx.y * blockDim.x
           + blockIdx.x + blockDim.x * blockDim.y
           + blockIdx.z * blockDim.x * blockDim.y * gridDim.x;*/

    int blockId = blockIdx.x + blockIdx.z * gridDim.x;

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x)
                   + blockId * (blockDim.x * blockDim.y);
    return threadId;
}

// gx,gz e bx,bz > 1 -> (32,1,32)(32,1,32)
//                       Dz   Dw  Dx   Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_2D_xz_block_2D_xz() {
    PRINT_FUNC_NAME;
    /*return threadIdx.x 
           + threadIdx.z * blockDim.x
           + blockIdx.x + blockDim.x * blockDim.z
           + blockIdx.z * blockDim.x * blockDim.z * gridDim.x;*/

    int blockId = blockIdx.x + blockIdx.z * gridDim.x;

    int threadId = threadIdx.x + (threadIdx.z * blockDim.x)
                   + blockId * (blockDim.x * blockDim.z);
    return threadId;
}

// gx,gz e by,bz > 1 -> (32,1,32)(1,32,32)
//                       Dz   Dw    Dx Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_2D_xz_block_2D_yz() {
    PRINT_FUNC_NAME;
    /*return threadIdx.y 
           + threadIdx.z * blockDim.y
           + blockIdx.x + blockDim.y * blockDim.z
           + blockIdx.z * blockDim.y * blockDim.z * gridDim.x;*/

    int blockId = blockIdx.x + blockIdx.z * gridDim.x;

    int threadId = threadIdx.y + (threadIdx.z * blockDim.y)
                   + blockId * (blockDim.y * blockDim.z);
    return threadId;
}

// gy,gz e bx,by > 1 -> (1,32,32)(32,32,1)
//                         Dz Dw  Dx Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_2D_yz_block_2D_xy() {
    PRINT_FUNC_NAME;
    /*return threadIdx.x 
           + threadIdx.y * blockDim.x
           + blockIdx.y + blockDim.x * blockDim.y
           + blockIdx.z * blockDim.x * blockDim.y * gridDim.y;*/

    int blockId = blockIdx.y + blockIdx.z * gridDim.y;

    int threadId = threadIdx.x + (threadIdx.y * blockDim.x)
                   + blockId * (blockDim.x * blockDim.y);
    return threadId;
}

// gy,gz e bx,bz > 1 -> (1,32,32)(32,1,32)
//                         Dz Dw  Dx   Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_2D_yz_block_2D_xz() {
    PRINT_FUNC_NAME;
    /*return threadIdx.x 
           + threadIdx.z * blockDim.x
           + blockIdx.y + blockDim.x * blockDim.z
           + blockIdx.z * blockDim.x * blockDim.z * gridDim.y;*/

    int blockId = blockIdx.y + blockIdx.z * gridDim.y;

    int threadId = threadIdx.x + (threadIdx.z * blockDim.x)
                   + blockId * (blockDim.x * blockDim.z);
    return threadId;
}

// gy,gz e by,bz > 1 -> (1,32,32)(1,32,32)
//                         Dz Dw    Dx Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_2D_yz_block_2D_yz() {
    PRINT_FUNC_NAME;
    /*return threadIdx.y 
           + threadIdx.z * blockDim.y
           + blockIdx.y + blockDim.y * blockDim.z
           + blockIdx.z * blockDim.y * blockDim.z * gridDim.y;*/

    int blockId = blockIdx.y + blockIdx.z * gridDim.y;

    int threadId = threadIdx.y + (threadIdx.z * blockDim.y)
                   + blockId * (blockDim.y * blockDim.z);
    return threadId;
}

//---------------------------------------
// 2D_3D -> Mapping 5D
//---------------------------------------
// gx,gy e bx,by,bz > 1 -> (32,32,1)(32,32,32)
//                          Dw Dt    Dx Dy Dz
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz + t . Dx . Dy . Dz . Dw
__device__ int getGlobalIdx_grid_2D_xy_block_3D_xyz() {
    PRINT_FUNC_NAME;
    // 14 operations.
    /*int threadId = threadIdx.x 
                   + threadIdx.y * blockDim.x
                   + threadIdx.z * (blockDim.x * blockDim.y)
                   + blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
                   + blockIdx.y * (blockDim.x * blockDim.y * blockDim.z) * gridDim.x
    */

    // 11 Operations.
    /*int blockDims_xy = (blockDim.x * blockDim.y);
    int blockDims_xyz = blockDims_xy * blockDim.z;

    int threadId = threadIdx.x 
                   + threadIdx.y * blockDim.x
                   + threadIdx.z * blockDims_xy
                   + blockIdx.x * blockDims_xyz
                   + blockIdx.y * blockDims_xyz * gridDim.x;
    return threadId;*/


    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    
    int threadId = threadIdx.x 
                   + (threadIdx.y * blockDim.x)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + blockId * (blockDim.x * blockDim.y * blockDim.z);
    return threadId;
}

// gx,gz e bx,by,bz > 1 -> (32,1,32)(32,32,32)
//                          Dw   Dt  Dx Dy Dz
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz + t . Dx . Dy . Dz . Dw
__device__ int getGlobalIdx_grid_2D_xz_block_3D_xyz() {
    PRINT_FUNC_NAME;
    int blockId = blockIdx.x + blockIdx.z * gridDim.x;
    
    int threadId = threadIdx.x 
                   + (threadIdx.y * blockDim.x)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + blockId * (blockDim.x * blockDim.y * blockDim.z);
    return threadId;
}

// gy,gz e bx,by,bz > 1 -> (1,32,32)(32,32,32)
//                            Dw Dt  Dx Dy Dz
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz + t . Dx . Dy . Dz . Dw
__device__ int getGlobalIdx_grid_2D_yz_block_3D_xyz() {
    PRINT_FUNC_NAME;
    int blockId = blockIdx.y + blockIdx.z * gridDim.y;
    
    int threadId = threadIdx.x 
                   + (threadIdx.y * blockDim.x)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + blockId * (blockDim.x * blockDim.y * blockDim.z);
    return threadId;
}

//---------------------------------------
// 3D_1D -> Mapping 4D
//---------------------------------------
// gx,gy,gz e bx > 1 -> (32,32,32)(32,1,1)
//                       Dy Dz Dw  Dx
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_3D_xyz_block_1D_x() {
    PRINT_FUNC_NAME;

    /*return threadIdx.x
           + blockIdx.x * blockDim.x
           + blockIdx.y * blockDim.x * gridDim.x
           + blockIdx.z * blockDim.x * gridDim.x * gridDim.y;*/

    int blockId = blockIdx.x 
                  + blockIdx.y * gridDim.x
                  + blockIdx.z * gridDim.x * gridDim.y;
    
    int threadId = threadIdx.x + blockId * blockDim.x;
    
    return threadId;
}

// gx,gy,gz e by > 1 -> (32,32,32)(1,32,1)
//                       Dy Dz Dw    Dx
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_3D_xyz_block_1D_y() {
    PRINT_FUNC_NAME;
    int blockId = blockIdx.x 
                  + blockIdx.y * gridDim.x
                  + blockIdx.z * gridDim.x * gridDim.y;
    
    int threadId = threadIdx.y + blockId * blockDim.y;
    
    return threadId;
}

// gx,gy,gz e bz > 1 -> (32,32,32)(1,1,32)
//                       Dy Dz Dw      Dx
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz
__device__ int getGlobalIdx_grid_3D_xyz_block_1D_z() {
    PRINT_FUNC_NAME;
    int blockId = blockIdx.x 
                  + blockIdx.y * gridDim.x
                  + blockIdx.z * gridDim.x * gridDim.y;
    
    int threadId = threadIdx.z + blockId * blockDim.z;
    
    return threadId;
}

//---------------------------------------
// 3D_2D -> Mapping 5D.
//---------------------------------------
// gx,gy,gz e bx,by > 1 -> (32,32,32)(16,32,1)
//                          Dz Dw Dt  Dx Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz + t . Dx . Dy . Dz . Dw
__device__ int getGlobalIdx_grid_3D_xyz_block_2D_xy() {
    PRINT_FUNC_NAME;
    int blockId = blockIdx.x 
                  + blockIdx.y * gridDim.x
                  + blockIdx.z * gridDim.x * gridDim.y;
    
    int threadId = threadIdx.x 
                   + (threadIdx.y * blockDim.x)
                   + blockId * (blockDim.x * blockDim.y);
    
    return threadId;
}

// gx,gy,gz e bx,bz > 1 -> (32,32,32)(16,1,32)
//                          Dz Dw Dt  Dx   Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz + t . Dx . Dy . Dz . Dw
__device__ int getGlobalIdx_grid_3D_xyz_block_2D_xz() {
    PRINT_FUNC_NAME;
    int blockId = blockIdx.x 
                  + blockIdx.y * gridDim.x
                  + blockIdx.z * gridDim.x * gridDim.y;
    
    int threadId = threadIdx.x 
                   + (threadIdx.z * blockDim.x)
                   + blockId * (blockDim.x * blockDim.z);
    
    return threadId;
}

// gx,gy,gz e by,bz > 1 -> (32,32,32)(1,16,32)
//                          Dz Dw Dt    Dx Dy
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz + t . Dx . Dy . Dz . Dw
__device__ int getGlobalIdx_grid_3D_xyz_block_2D_yz() {
    PRINT_FUNC_NAME;
    int blockId = blockIdx.x 
                  + blockIdx.y * gridDim.x
                  + blockIdx.z * gridDim.x * gridDim.y;
    
    int threadId = threadIdx.y 
                   + (threadIdx.z * blockDim.y)
                   + blockId * (blockDim.y * blockDim.z);
    
    return threadId;
}

//---------------------------------------
// 3D_3D -> Mapping 6D.
//---------------------------------------
// gx,gy,gz e bx,by,bz > 1 -> (32,32,32)(2,16,32)
//                          Dw Dt Du Dx Dy Dz
//  id = x + y . Dx + z . Dx . Dy + w . Dx . Dy . Dz + t . Dx . Dy . Dz . Dw + u . Dx . Dy . Dz . Dw . Dt
__device__ int getGlobalIdx_grid_3D_xyz_block_3D_xyz() {
    PRINT_FUNC_NAME;
    // Operações -> multiply: 9 add: 5 (14 FLOPs).
    // printf("getGlobalIdx_3D_3D.\n");
    int blockId = blockIdx.x 
                  + blockIdx.y * gridDim.x
                  + blockIdx.z * gridDim.x * gridDim.y;
    
    int threadId = threadIdx.x 
                   + threadIdx.y * blockDim.x
                   + threadIdx.z * blockDim.x * blockDim.y
                   + blockId * blockDim.x * blockDim.y * blockDim.z;

    return threadId;
}

/*
     grid(gx,gy,gx) block(bx,by,bz)
     funcId é escolhida com base nos valores de [gx,gy,gx,bx,by,bz]
     Cada valor irá contribuir com uma parcela para o cálculo do índice da função:
     [gx > 1, gy > 1, gx > 1, bx > 1, by > 1, bz > 1]
     Exemplo: grid(32,1,1) block(32,1,1)
              [1,0,0,1,0,0] -> [32,16,8,4,2,1] = [32 + 4] = 36
              A função getGlobalIdFunc(36) será:
              // 36: 100 100 getGlobalIdx_grid_1D_x_block_1D_x 
*/
int calculateFunctionId(dim3 grid, dim3 block){
  int funcId = 0;

  funcId += (grid.x > 1) ? 32 : 0;
  funcId += (grid.y > 1) ? 16 : 0;
  funcId += (grid.z > 1) ? 8 : 0;
  funcId += (block.x > 1) ? 4 : 0;
  funcId += (block.y > 1) ? 2 : 0;
  funcId += (block.z > 1) ? 1 : 0;

  return funcId;
}


/* Tabela de funções para chamada parametrizada. */
/*__device__ op_func getGlobalIdFunc[9] = { getGlobalIdx_1D_1D, getGlobalIdx_1D_2D, getGlobalIdx_1D_3D, 
                      getGlobalIdx_2D_1D, getGlobalIdx_2D_2D, getGlobalIdx_2D_3D,
                      getGlobalIdx_3D_1D, getGlobalIdx_3D_2D, getGlobalIdx_3D_3D};
*/
__device__ op_func getGlobalIdFunc[64] = {
    /* 0: 000 000*/ NULL,
    /* 1: 000 001*/ getGlobalIdx_block_1D_z,
    /* 2: 000 010*/ getGlobalIdx_block_1D_y,
    /* 3: 000 011*/ getGlobalIdx_block_2D_yz,
    /* 4: 000 100*/ getGlobalIdx_block_1D_x,
    /* 5: 000 101*/ getGlobalIdx_block_2D_xz,
    /* 6: 000 110*/ getGlobalIdx_block_2D_xy,
    /* 7: 000 111*/ getGlobalIdx_block_3D_xyz,
    /* 8: 001 000*/ getGlobalIdx_grid_1D_z,
    /* 9: 001 001*/ getGlobalIdx_grid_1D_z_block_1D_z,
    /*10: 001 010*/ getGlobalIdx_grid_1D_z_block_1D_y,
    /*11: 001 011*/ getGlobalIdx_grid_1D_z_block_2D_yz,
    /*12: 001 100*/ getGlobalIdx_grid_1D_z_block_1D_x,
    /*13: 001 101*/ getGlobalIdx_grid_1D_z_block_2D_xz,
    /*14: 001 110*/ getGlobalIdx_grid_1D_z_block_2D_xy,
    /*15: 001 111*/ getGlobalIdx_grid_1D_z_block_3D_xyz,
    /*16: 010 000*/ getGlobalIdx_grid_1D_y,
    /*17: 010 001*/ getGlobalIdx_grid_1D_y_block_1D_z,
    /*18: 010 010*/ getGlobalIdx_grid_1D_y_block_1D_y,
    /*19: 010 011*/ getGlobalIdx_grid_1D_y_block_2D_yz,
    /*20: 010 100*/ getGlobalIdx_grid_1D_y_block_1D_x,
    /*21: 010 101*/ getGlobalIdx_grid_1D_y_block_2D_xz,
    /*22: 010 110*/ getGlobalIdx_grid_1D_y_block_2D_xy,
    /*23: 010 111*/ getGlobalIdx_grid_1D_y_block_3D_xyz,
    /*24: 011 000*/ getGlobalIdx_grid_2D_yz,
    /*25: 011 001*/ getGlobalIdx_grid_2D_yz_block_1D_z,
    /*26: 011 010*/ getGlobalIdx_grid_2D_yz_block_1D_y,
    /*27: 011 011*/ getGlobalIdx_grid_2D_yz_block_2D_yz,
    /*28: 011 100*/ getGlobalIdx_grid_2D_yz_block_1D_x,
    /*29: 011 101*/ getGlobalIdx_grid_2D_yz_block_2D_xz,
    /*30: 011 110*/ getGlobalIdx_grid_2D_yz_block_2D_xy,
    /*31: 011 111*/ getGlobalIdx_grid_2D_yz_block_3D_xyz,
    /*32: 100 000*/ getGlobalIdx_grid_1D_x,
    /*33: 100 001*/ getGlobalIdx_grid_1D_x_block_1D_z,
    /*34: 100 010*/ getGlobalIdx_grid_1D_x_block_1D_y,
    /*35: 100 011*/ getGlobalIdx_grid_1D_x_block_2D_yz,
    /*36: 100 100*/ getGlobalIdx_grid_1D_x_block_1D_x,
    /*37: 100 101*/ getGlobalIdx_grid_1D_x_block_2D_xz,
    /*38: 100 110*/ getGlobalIdx_grid_1D_x_block_2D_xy,
    /*39: 100 111*/ getGlobalIdx_grid_1D_x_block_3D_xyz,
    /*40: 101 000*/ getGlobalIdx_grid_2D_xz,
    /*41: 101 001*/ getGlobalIdx_grid_2D_xz_block_1D_z,
    /*42: 101 010*/ getGlobalIdx_grid_2D_xz_block_1D_y,
    /*43: 101 011*/ getGlobalIdx_grid_2D_xz_block_2D_yz,
    /*44: 101 100*/ getGlobalIdx_grid_2D_xz_block_1D_x,
    /*45: 101 101*/ getGlobalIdx_grid_2D_xz_block_2D_xz,
    /*46: 101 110*/ getGlobalIdx_grid_2D_xz_block_2D_xy,
    /*47: 101 111*/ getGlobalIdx_grid_2D_xz_block_3D_xyz,
    /*48: 110 000*/ getGlobalIdx_grid_2D_xy,
    /*49: 110 001*/ getGlobalIdx_grid_2D_xy_block_1D_z,
    /*50: 110 010*/ getGlobalIdx_grid_2D_xy_block_1D_y,
    /*51: 110 011*/ getGlobalIdx_grid_2D_xy_block_2D_yz,
    /*52: 110 100*/ getGlobalIdx_grid_2D_xy_block_1D_x,
    /*53: 110 101*/ getGlobalIdx_grid_2D_xy_block_2D_xz,
    /*54: 110 110*/ getGlobalIdx_grid_2D_xy_block_2D_xy,
    /*55: 110 111*/ getGlobalIdx_grid_2D_xy_block_3D_xyz, 
    /*56: 111 000*/ getGlobalIdx_grid_3D_xyz,
    /*57: 111 001*/ getGlobalIdx_grid_3D_xyz_block_1D_z,
    /*58: 111 010*/ getGlobalIdx_grid_3D_xyz_block_1D_y,
    /*59: 111 011*/ getGlobalIdx_grid_3D_xyz_block_2D_yz,
    /*60: 111 100*/ getGlobalIdx_grid_3D_xyz_block_1D_x,
    /*61: 111 101*/ getGlobalIdx_grid_3D_xyz_block_2D_xz,
    /*62: 111 110*/ getGlobalIdx_grid_3D_xyz_block_2D_xy,
    /*62: 111 111*/ getGlobalIdx_grid_3D_xyz_block_3D_xyz
};
