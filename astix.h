//////////////////////////////////////////////////////////////////////
//
//  Scalar  - A number with only 1 quantity.
//
//  Vector  - Measures a value, by having to quantities that
//      are treated as one. Has direction and magnitude.
//
//  Length/Magnitude - Length of vector from origin.
//
//  Unit/Normal -
//      Vector where elements are divided by the length
//      of the vector. Fits within a single unit. Represents
//      the direction of the vector with a length of 1.
//
//  Haddamard Product -
//      Multiply each element of the vector by the 
//      vector's cooresponding element.
//     
//////////////////////////////////////////////////////////////////////
//
//  Translation -   Move left right up or down. Vector is placed
//                  along the right side.
//
//  Reflection  - Invert vector component signs.
//  Rotation    - Rotate vector around origin.
//  Dialation   - Control scale of vector.
//      
//////////////////////////////////////////////////////////////////////

/*************************************************
 * 
 * Identity 
 * [1, 0, 0, 0]
 * [0, 1, 0, 0]
 * [0, 0, 1, 0]
 * [0, 0, 0, 1]
 * 
 * Translate 
 * [1, 0, 0, X]
 * [1, 1, 0, Y]
 * [0, 0, 1, Z]
 * [0, 0, 0, W] 
 * 
 * Scale 
 * [W, 0, 0, 0] Width 
 * [0, H, 0, 0] Height 
 * [0, 0, L, 0] Length
 * [0, 0, 0, 1] 
 * 
*************************************************/

#ifndef __ASTIX_H__
#define __ASTIX_H__

#ifdef __cplusplus
extern "C" {
#endif // C++

#define astix_pi     (3.1415926535897932384626433833)
#define astix_tau    (6.2831853071795864769252867666) // pi * 2
#define astix_euler  (2.7182818284590452353602874714)
#define astix_golden (1.6180339887498948482045868343)

#define astix_min(a, b) ((a) < (b) ? (a) : (b))
#define astix_max(a, b) ((a) > (b) ? (a) : (b))
#define astix_abs(a) ((a) > 0 ? (a) : -(a))
#define astix_mod(a, m) (((a) % (m)) >= 0 ? ((a) % (m)) : (((a) % (m)) + (m)))
#define astix_sqr(x) ((x) * (x))
#define astix_isnan(x) ((x) != (x))
#define astix_rad2deg(x) ((x) * (57.295779513082320876798154813)) // 180/pi
#define astix_deg2rad(x) ((x) * (0.0174532925199432957692369076)) // pi/180

typedef float scal_t;

typedef union {
    struct { scal_t x,y; };
    struct { scal_t u,v; };
    struct { scal_t s,t; };
    scal_t  elm[2];
} vec2_t;

typedef union {
    struct { scal_t x,y,z; };
    struct { scal_t r,g,b; };
    scal_t  elm[3];
} vec3_t;

typedef union {
    struct { scal_t x,y,z,w; };
    struct { scal_t r,g,b,a; };
    scal_t  elm[4];
} vec4_t;

typedef union {
    struct { scal_t x,y,z,w; };
    struct { scal_t r,g,b,a; };
    struct { scal_t i,j,k; };
    scal_t  elm[4];
} quat_t;

typedef union {
    vec2_t vec[2];
    scal_t mat[2][2];
    scal_t elm[4];
} mat2_t;

typedef union {
    vec3_t vec[3];
    scal_t mat[3][3];
    scal_t elm[9];
} mat3_t;

typedef union {
    vec4_t vec[4];
    quat_t qts[4];
    scal_t mat[4][4];
    scal_t elm[16];
} mat4_t;

//  It just works...
#define vec2_init(...) (vec2_t){__VA_ARGS__}
#define vec3_init(...) (vec3_t){__VA_ARGS__}
#define vec4_init(...) (vec4_t){__VA_ARGS__}
#define mat2_init(...) (mat2_t){__VA_ARGS__}
#define mat3_init(...) (mat3_t){__VA_ARGS__}
#define mat4_init(...) (mat4_t){__VA_ARGS__}
#define quat_init(...) (quat_t){__VA_ARGS__}

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	2D Vector
//
//===============================================================
/////////////////////////////////////////////////////////////////

extern vec2_t vec2_add      (vec2_t U, vec2_t V);
extern vec2_t vec2_sub      (vec2_t U, vec2_t V);
extern vec2_t vec2_mul      (vec2_t U, vec2_t V);
extern vec2_t vec2_adds     (vec2_t U, scal_t S);
extern vec2_t vec2_subs     (vec2_t U, scal_t S);
extern vec2_t vec2_muls     (vec2_t U, scal_t S);
extern scal_t vec2_dot      (vec2_t U, vec2_t V);
extern scal_t vec2_lensqr   (vec2_t U);
extern scal_t vec2_length   (vec2_t U);
extern vec2_t vec2_normal   (vec2_t U);
extern vec2_t vec2_mul_mat2 (vec2_t V, mat2_t M);
extern vec2_t vec2_lerp     (vec2_t start, vec2_t end, scal_t movement);
extern vec2_t vec2_polar    ( float dist, float ang );
/////////////////////////////////////////////////////////////////
//===============================================================
//
//	3D Vector
//
//===============================================================
/////////////////////////////////////////////////////////////////

extern vec3_t vec3_add      (vec3_t U, vec3_t V);
extern vec3_t vec3_sub      (vec3_t U, vec3_t V);
extern vec3_t vec3_mul      (vec3_t U, vec3_t V);
extern vec3_t vec3_adds     (vec3_t U, scal_t S);
extern vec3_t vec3_subs     (vec3_t U, scal_t S);
extern vec3_t vec3_muls     (vec3_t U, scal_t S);
extern scal_t vec3_dot      (vec3_t U, vec3_t V);
extern scal_t vec3_lensqr   (vec3_t U);
extern scal_t vec3_length   (vec3_t U);
extern vec3_t vec3_normal   (vec3_t U);
extern vec3_t vec3_cross    (vec3_t U, vec3_t V);
extern vec3_t vec3_mul_mat3 (vec3_t V, mat3_t M);
extern vec3_t vec3_mul_mat4 (vec3_t V, mat4_t M);
extern vec3_t vec3_lerp     (vec3_t start, vec3_t end, scal_t movement);

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	4D Vector
//
//===============================================================
/////////////////////////////////////////////////////////////////

extern vec4_t vec4_add      (vec4_t U, vec4_t V);
extern vec4_t vec4_sub      (vec4_t U, vec4_t V);
extern vec4_t vec4_mul      (vec4_t U, vec4_t V);
extern vec4_t vec4_adds     (vec4_t U, scal_t S);
extern vec4_t vec4_subs     (vec4_t U, scal_t S);
extern vec4_t vec4_muls     (vec4_t U, scal_t S);
extern scal_t vec4_dot      (vec4_t U, vec4_t V);
extern scal_t vec4_lensqr   (vec4_t U);
extern scal_t vec4_length   (vec4_t U);
extern vec4_t vec4_normal   (vec4_t U);
extern vec4_t vec4_mul_mat4 (vec4_t V, mat4_t M);
extern vec4_t vec4_lerp     (vec4_t start, vec4_t end, scal_t movement);

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	Quaternion
//
//===============================================================
/////////////////////////////////////////////////////////////////

extern quat_t quat_add      (quat_t U, quat_t V);
extern quat_t quat_sub      (quat_t U, quat_t V);
extern quat_t quat_mul      (quat_t X, quat_t Y);
extern quat_t quat_adds     (quat_t U, scal_t S);
extern quat_t quat_subs     (quat_t U, scal_t S);
extern quat_t quat_muls     (quat_t U, scal_t S);
extern scal_t quat_dot      (quat_t U, quat_t V);
extern scal_t quat_lensqr   (quat_t U);
extern scal_t quat_length   (quat_t U);
extern quat_t quat_normal   (quat_t U);
extern quat_t quat_inverse  (quat_t Q);
extern mat4_t quat_to_mat4  (quat_t Q);
extern scal_t quat_find_w   (quat_t q);

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	2x2 Matrix
//
//===============================================================
/////////////////////////////////////////////////////////////////

extern mat2_t mat2_diagonal  (scal_t d);
extern mat2_t mat2_add       (mat2_t M, mat2_t N);
extern mat2_t mat2_sub       (mat2_t M, mat2_t N);
extern mat2_t mat2_mul       (mat2_t M, mat2_t N);
extern mat2_t mat2_adds      (mat2_t M, scal_t N);
extern mat2_t mat2_subs      (mat2_t M, scal_t N);
extern mat2_t mat2_muls      (mat2_t M, scal_t N);
extern mat2_t mat2_transpose (mat2_t M);
extern mat2_t mat2_inverse   (mat2_t M);

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	3x3 Matrix
//
//===============================================================
/////////////////////////////////////////////////////////////////

extern mat3_t mat3_diagonal  (scal_t d);
extern mat3_t mat3_add       (mat3_t M, mat3_t N);
extern mat3_t mat3_sub       (mat3_t M, mat3_t N);
extern mat3_t mat3_mul       (mat3_t M, mat3_t N);
extern mat3_t mat3_adds      (mat3_t M, scal_t N);
extern mat3_t mat3_subs      (mat3_t M, scal_t N);
extern mat3_t mat3_muls      (mat3_t M, scal_t N);
extern mat3_t mat3_transpose (mat3_t M);
extern mat3_t mat3_inverse   (mat3_t M);

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	4x4 Matrix
//
//===============================================================
/////////////////////////////////////////////////////////////////

extern mat4_t mat4_diagonal       (scal_t d);
extern mat4_t mat4_add            (mat4_t M, mat4_t N);
extern mat4_t mat4_sub            (mat4_t M, mat4_t N);
extern mat4_t mat4_mul            (mat4_t M, mat4_t N);
extern mat4_t mat4_adds           (mat4_t M, scal_t N);
extern mat4_t mat4_subs           (mat4_t M, scal_t N);
extern mat4_t mat4_muls           (mat4_t M, scal_t N);
extern mat4_t mat4_transpose      (mat4_t M);
extern mat4_t mat4_inverse        (mat4_t M);
extern mat4_t mat4_scale          (vec3_t V);
extern mat4_t mat4_translation    (vec3_t V);
extern mat4_t mat4_ortho          (scal_t L, scal_t R, scal_t B, scal_t T, scal_t N, scal_t F);
extern mat4_t mat4_perspective    (scal_t V, scal_t A, scal_t N, scal_t F);
extern mat4_t mat4_lookat         (vec3_t eye, vec3_t center, vec3_t up);
extern mat4_t mat4_rotate         (vec3_t V);

#ifdef __cplusplus
}
#endif // C++

#endif // __ASTIX_H__