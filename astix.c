/*************************************************
 * Astix math library. 
 * 
 * A linear algebra library designed with the goal
 * of simplifying concepts like vectors, matrices
 * and quaternions. Work in progress..
*************************************************/
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

#ifdef __cplusplus
extern "C" {
#endif // C++

#include <math.h>
#include "astix.h"

float invsqrt ( float x ) {

	int i;
	float y, r;

	y = x * 0.5f;
	i = *(int*)( &x );
	i = 0x5f3759df - ( i >> 1 );
	r = *(float*)( &i );
	r = r * ( 1.5f - r * r * y );
    r = r * ( 1.5f - r * r * y );
    r = r * ( 1.5f - r * r * y );
	return r;
}


/////////////////////////////////////////////////////////////////
//===============================================================
//
//	2D Vector
//
//===============================================================
/////////////////////////////////////////////////////////////////

vec2_t vec2_add (vec2_t U, vec2_t V) {
    return vec2_init(
        U.x + V.x,
        U.y + V.y
    );
}

vec2_t vec2_sub (vec2_t U, vec2_t V) {
    return vec2_init(
        U.x - V.x,
        U.y - V.y
    );
}

vec2_t vec2_mul (vec2_t U, vec2_t V) {
    return vec2_init(
        U.x * V.x,
        U.y * V.y
    );
}

vec2_t vec2_adds (vec2_t U, scal_t S) {
    return vec2_init(
        U.x + S,
        U.y + S
    );
}

vec2_t vec2_subs (vec2_t U, scal_t S) {
    return vec2_init(
        U.x - S,
        U.y - S
    );
}

vec2_t vec2_muls (vec2_t U, scal_t S) {
    return vec2_init(
        U.x * S,
        U.y * S);
}

scal_t vec2_dot ( vec2_t u, vec2_t v ) {
    return  ( u.x * v.x ) +
            ( u.y * v.y);
}

scal_t vec2_lensqr ( vec2_t v ) {
    return ( v.x * v.x ) +
           ( v.y * v.y);
}

scal_t vec2_invlength ( vec2_t v ) {
    return invsqrt ( vec2_lensqr (v) );
}

scal_t vec2_length ( vec2_t v ) {
    return (sqrtf(vec2_lensqr(v)));
}

vec2_t vec2_normal (vec2_t v) {
    return (vec2_muls(v, vec2_invlength(v)));
}

vec2_t vec2_mul_mat2(vec2_t v, mat2_t m)
{
	return vec2_init(
        m.vec[0].x * v.x + m.vec[0].y * v.y,
	    m.vec[1].x * v.x + m.vec[1].y * v.y);
}

vec2_t vec2_polar ( float dist, float ang )
{
    return vec2_init(
        dist * cosf(ang),
        dist * sinf(ang),    
    );
}


/////////////////////////////////////////////////////////////////
//===============================================================
//
//	3D Vector
//
//===============================================================
/////////////////////////////////////////////////////////////////

vec3_t vec3_add ( vec3_t U, vec3_t V ) {
    return vec3_init(
        U.x + V.x,
        U.y + V.y,
        U.z + V.z
    );
}

vec3_t vec3_sub ( vec3_t U, vec3_t V ) {
    return vec3_init(
        U.x - V.x,
        U.y - V.y,
        U.z - V.z
    );
}

vec3_t vec3_mul ( vec3_t U, vec3_t V ) {
    return vec3_init(
        U.x * V.x,
        U.y * V.y,
        U.z * V.z
    );
}

vec3_t vec3_adds ( vec3_t U, scal_t S ) {
    return vec3_init(
        U.x + S,
        U.y + S,
        U.z + S
    );
}

vec3_t vec3_subs ( vec3_t U, scal_t S ) {
    return vec3_init(
        U.x - S,
        U.y - S,
        U.z - S
    );
}

vec3_t vec3_muls ( vec3_t U, scal_t S ) {
    return vec3_init(
        U.x * S,
        U.y * S,
        U.z * S
    );
}

scal_t vec3_dot ( vec3_t U, vec3_t V ) {
    return  ( U.x * V.x ) +
            ( U.y * V.y ) + 
            ( U.z * V.z );
}

scal_t vec3_lensqr ( vec3_t U ) {
    return (U.x * U.x) +
           (U.y * U.y) +
           (U.z * U.z);
}

scal_t vec3_invlength ( vec3_t U ) {
    return invsqrt( vec3_lensqr(U) );
}

scal_t vec3_length ( vec3_t U ) {
    return sqrtf(vec3_lensqr(U));
}

vec3_t vec3_normal ( vec3_t U ) {
    return (vec3_muls(U, vec3_invlength(U)));
}

vec3_t vec3_cross ( vec3_t U, vec3_t V ) {
    return vec3_init(
        (U.y*V.z)-(U.z*V.y),
        (U.x*V.z)-(U.z*V.x),
        (U.x*V.y)-(U.y*V.x)
    );
}


vec3_t vec3_mul_mat3 ( vec3_t v, mat3_t m ) {
	return vec3_init(
        m.vec[0].x * v.x + m.vec[0].y * v.y + m.vec[0].z * v.z,
	    m.vec[1].x * v.x + m.vec[1].y * v.y + m.vec[1].z * v.z,
	    m.vec[2].x * v.x + m.vec[2].y * v.y + m.vec[2].z * v.z
    );
}

vec3_t vec3_mul_mat4 ( vec3_t V, mat4_t M )
{
    scal_t s =  1.0f/ (M.vec[3].x * V.x + M.vec[3].w  * V.y + M.vec[3].z * V.z + M.vec[3].w);
    
    vec3_t U = vec3_init(
            (M.vec[0].x * V.x + M.vec[0].y * V.y + M.vec[0].z * V.z + M.vec[0].w) * s,
            (M.vec[1].x * V.x + M.vec[1].y * V.y + M.vec[1].z * V.z + M.vec[1].w) * s,
            (M.vec[2].x * V.x + M.vec[2].y * V.y + M.vec[2].z * V.z + M.vec[2].w) * s
    );
    
    
    return U;
}

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	4D Vector
//
//===============================================================
/////////////////////////////////////////////////////////////////

vec4_t vec4_add ( vec4_t U, vec4_t V ) {
    return vec4_init(
        U.x + V.x,
        U.y + V.y,
        U.z + V.z,
        U.w + V.w
    );
}

vec4_t vec4_sub ( vec4_t U, vec4_t V ) {
    return vec4_init(
        U.x - V.x,
        U.y - V.y,
        U.z - V.z,
        U.w - V.w
    );
}

vec4_t vec4_mul ( vec4_t U, vec4_t V ) {
    return vec4_init(
        U.x * V.x,
        U.y * V.y,
        U.z * V.z,
        U.w * V.w
    );
}

vec4_t vec4_adds ( vec4_t U, scal_t S ) {
    return vec4_init(
        U.x + S,
        U.y + S,
        U.z + S,
        U.w + S
    );
}

vec4_t vec4_subs ( vec4_t U, scal_t S ) {
    return vec4_init(
        U.x - S,
        U.y - S,
        U.z - S,
        U.w - S
    );
}

vec4_t vec4_muls ( vec4_t U, scal_t S) {
    return vec4_init(
        U.x * S,
        U.y * S,
        U.z * S,
        U.w * S
    );
}

scal_t vec4_dot ( vec4_t U, vec4_t V ) {
    return  ( U.x * V.x ) + 
            ( U.y * V.y ) +
            ( U.z * V.z ) + 
            ( U.w * V.w );
}
scal_t vec4_lensqr ( vec4_t U ) {
    return (U.x * U.x) +
           (U.y * U.y) +
           (U.z * U.z) +
           (U.w * U.w);
}

scal_t vec4_invlength (vec4_t U) {
    return invsqrt( vec4_lensqr(U) );
}

scal_t vec4_length ( vec4_t U ) {
    return sqrtf(vec4_lensqr(U));
}

vec4_t vec4_normal ( vec4_t U ) {
    return vec4_muls(U, vec4_invlength(U));
}

vec4_t vec4_mul_mat4 ( vec4_t v, mat4_t m ) {
	return vec4_init(
        (m.vec[0].x*v.x)+(m.vec[0].y*v.y)+(m.vec[0].z*v.z)+(m.vec[0].w*v.w),
	    (m.vec[1].x*v.x)+(m.vec[1].y*v.y)+(m.vec[1].z*v.z)+(m.vec[1].w*v.w),
	    (m.vec[2].x*v.x)+(m.vec[2].y*v.y)+(m.vec[2].z*v.z)+(m.vec[2].w*v.w),
	    (m.vec[3].x*v.x)+(m.vec[3].y*v.y)+(m.vec[3].z*v.z)+(m.vec[3].w*v.w));
}

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	Quaternion
//
//===============================================================
/////////////////////////////////////////////////////////////////


quat_t quat_add  ( quat_t U, quat_t V ) {
    return quat_init(
        U.x + V.x,
        U.y + V.y,
        U.z + V.z,
        U.w + V.w
    );
}

quat_t quat_sub  ( quat_t U, quat_t V ) {
    return quat_init(
        U.x - V.x,
        U.y - V.y,
        U.z - V.z,
        U.w - V.w
    );
}

quat_t quat_mul  ( quat_t U, quat_t V ) {
    return quat_init(
        U.w * V.x + U.x * V.w + U.y * V.z - U.z * V.y,
        U.w * V.y + U.y * V.w + U.z * V.x - U.x * V.z,
        U.w * V.z + U.z * V.w + U.x * V.y - U.y * V.x,
        U.w * V.w - U.x * V.x - U.y * V.y - U.z * V.z
    );
}

quat_t quat_adds ( quat_t U, scal_t S ) {
    return quat_init(
        U.x + S,
        U.y + S,
        U.z + S,
        U.w + S
    );
}

quat_t quat_subs ( quat_t U, scal_t S ) {
    return quat_init(
        U.x - S,
        U.y - S,
        U.z - S,
        U.w - S
    );
}

quat_t quat_muls ( quat_t U, scal_t S ) {
    return quat_init(
        U.x * S,
        U.y * S,
        U.z * S,
        U.w * S
    );
}

scal_t quat_dot  ( quat_t q, quat_t r ) {
    return ( q.x * r.x ) + 
           ( q.y * r.y ) + 
           ( q.z * r.z ) + 
           ( q.w * r.w );
}

scal_t quat_lensqr ( quat_t q ) {
    return (q.x * q.x) +
           (q.y * q.y) +
           (q.z * q.z) +
           (q.w * q.w);
}

scal_t quat_invlength ( quat_t q ) {
    return invsqrt ( quat_lensqr ( q ) );
}

scal_t quat_length ( quat_t q ) {
    return sqrtf ( quat_lensqr(q) );
}

quat_t quat_normal ( quat_t q ) {
    return quat_muls ( q, quat_invlength(q) );
}

quat_t quat_conjugate ( quat_t q ) {
    return quat_init ( -q.x, -q.y, -q.z, q.w );
}

quat_t quat_inverse ( quat_t q ) {
    return quat_normal ( quat_conjugate(q) );
}

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	2x2 Matrix
//
//===============================================================
/////////////////////////////////////////////////////////////////

static mat2_t _mat2_id_ = {
    1.0f, 0.0f,
    0.0f, 1.0f
};

mat2_t mat2_identity ( void ) {
    return _mat2_id_;
}

mat2_t mat2_diagonal (float d) {
    return (mat2_t){d, 0.0f, 0.0f, d};
}

mat2_t mat2_add (mat2_t M, mat2_t N) {
    return mat2_init(
        vec2_add(M.vec[0], N.vec[0]),
        vec2_add(M.vec[1], N.vec[1])
    );
}

mat2_t mat2_sub (mat2_t M, mat2_t N) {
    return mat2_init(
        vec2_sub(M.vec[0], N.vec[0]),
        vec2_sub(M.vec[1], N.vec[1])
    );
}

mat2_t mat2_mul(mat2_t M, mat2_t N)
{
    mat2_t O = mat2_identity();    
    O.mat[0][0] = M.mat[0][0] * N.mat[0][0] + M.mat[1][0] * N.mat[0][1];
    O.mat[0][1] = M.mat[0][1] * N.mat[0][0] + M.mat[1][1] * N.mat[0][1];
    O.mat[1][0] = M.mat[0][0] * N.mat[1][0] + M.mat[1][0] * N.mat[1][1];
    O.mat[1][1] = M.mat[0][1] * N.mat[1][0] + M.mat[1][1] * N.mat[1][1];
    return O;
}

mat2_t mat2_adds (mat2_t M, scal_t S) {
    return mat2_init(
        vec2_adds(M.vec[0], S),
        vec2_adds(M.vec[1], S)
    );
}

mat2_t mat2_subs (mat2_t M, scal_t S) {
    return mat2_init(
        vec2_subs(M.vec[0], S),
        vec2_subs(M.vec[1], S)
    );
}

mat2_t mat2_muls (mat2_t M, scal_t S) {
    return mat2_init(
        vec2_muls(M.vec[0], S),
        vec2_muls(M.vec[1], S)
    );
}

mat2_t mat2_transpose (mat2_t M)
{
    mat2_t R = M; 
    R.elm[1] = M.elm[2];
    R.elm[2] = M.elm[1];
    return R;
}

mat2_t mat2_inverse(mat2_t M)
{
    mat2_t I = mat2_identity();
    scal_t det = M.mat[0][0] * M.mat[1][1] - M.mat[0][1] * M.mat[1][0];
    
    if( fabs(det) < 1e-10 )
    {
        return I;
    }
    
    det = 1.0f/det;
    I.elm[0] = det*+M.mat[1][1];
    I.elm[1] = det*-M.mat[0][1];
    I.elm[2] = det*-M.mat[1][0];
    I.elm[3] = det*+M.mat[0][0];
    return I;
}
/////////////////////////////////////////////////////////////////
//===============================================================
//
//	3x3 Matrix
//
//===============================================================
/////////////////////////////////////////////////////////////////

static mat3_t _mat3_id_ = {
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f
};

mat3_t mat3_identity ( void ) {
    return _mat3_id_;
}

mat3_t mat3_add (mat3_t M, mat3_t N)
{
    return mat3_init(
        vec3_add(M.vec[0], N.vec[0]),
        vec3_add(M.vec[1], N.vec[1]),
        vec3_add(M.vec[2], N.vec[2])
    );
}

mat3_t mat3_sub (mat3_t M, mat3_t N)
{
    return mat3_init(
        vec3_sub(M.vec[0], N.vec[0]), 
        vec3_sub(M.vec[1], N.vec[1]), 
        vec3_sub(M.vec[2], N.vec[2])
    );
}

mat3_t mat3_mul (mat3_t M, mat3_t N)
{
    mat3_t O = mat3_identity();
    O.mat[0][0] = M.mat[0][0] * N.mat[0][0] + M.mat[1][0] * N.mat[0][1] + M.mat[2][0] * N.mat[0][2];
    O.mat[0][1] = M.mat[0][1] * N.mat[0][0] + M.mat[1][1] * N.mat[0][1] + M.mat[2][1] * N.mat[0][2];
    O.mat[0][2] = M.mat[0][2] * N.mat[0][0] + M.mat[1][2] * N.mat[0][1] + M.mat[2][2] * N.mat[0][2];
    O.mat[1][0] = M.mat[0][0] * N.mat[1][0] + M.mat[1][0] * N.mat[1][1] + M.mat[2][0] * N.mat[1][2];
    O.mat[1][1] = M.mat[0][1] * N.mat[1][0] + M.mat[1][1] * N.mat[1][1] + M.mat[2][1] * N.mat[1][2];
    O.mat[1][2] = M.mat[0][2] * N.mat[1][0] + M.mat[1][2] * N.mat[1][1] + M.mat[2][2] * N.mat[1][2];
    O.mat[2][0] = M.mat[0][0] * N.mat[2][0] + M.mat[1][0] * N.mat[2][1] + M.mat[2][0] * N.mat[2][2];
    O.mat[2][1] = M.mat[0][1] * N.mat[2][0] + M.mat[1][1] * N.mat[2][1] + M.mat[2][1] * N.mat[2][2];
    O.mat[2][2] = M.mat[0][2] * N.mat[2][0] + M.mat[1][2] * N.mat[2][1] + M.mat[2][2] * N.mat[2][2];
    return O;    
}

mat3_t mat3_adds (mat3_t M, scal_t S)
{
    return mat3_init(
        vec3_adds(M.vec[0], S), 
        vec3_adds(M.vec[1], S), 
        vec3_adds(M.vec[2], S)
    );
}

mat3_t mat3_subs (mat3_t M, scal_t S)
{
    return mat3_init(
        vec3_subs(M.vec[0], S), 
        vec3_subs(M.vec[1], S), 
        vec3_subs(M.vec[2], S)
    );
}

mat3_t mat3_muls (mat3_t M, scal_t S)
{
    return mat3_init(
        vec3_muls(M.vec[0], S), 
        vec3_muls(M.vec[1], S), 
        vec3_muls(M.vec[2], S)
    );
}

mat3_t mat3_transpose(mat3_t M)
{
    mat3_t T = M;
    T.mat[0][1] = M.mat[1][0];
    T.mat[0][2] = M.mat[2][0];
    T.mat[1][0] = M.mat[0][1];
    T.mat[1][2] = M.mat[2][1];
    T.mat[2][0] = M.mat[0][2];
    T.mat[2][1] = M.mat[1][2];
    return T;
}

mat3_t mat3_inverse(mat3_t M)
{
	mat3_t I = mat3_identity();
	double det = 0.0f;

	I.mat[0][0] = M.mat[1][1] * M.mat[2][2] - M.mat[1][2] * M.mat[2][1];
	I.mat[1][0] = M.mat[1][2] * M.mat[2][0] - M.mat[1][0] * M.mat[2][2];
	I.mat[2][0] = M.mat[1][0] * M.mat[2][1] - M.mat[1][1] * M.mat[2][0];

	det = M.mat[0][0] * I.mat[0][0] + M.mat[0][1] * I.mat[1][0] + M.mat[0][2] * I.mat[2][0];

	if ( fabs(det) < 1e-10 ) {
		return I;
	}

	det = 1.0f / det;

	I.mat[0][1] = M.mat[0][2] * M.mat[2][1] - M.mat[0][1] * M.mat[2][2];
	I.mat[0][2] = M.mat[0][1] * M.mat[1][2] - M.mat[0][2] * M.mat[1][1];
	I.mat[1][1] = M.mat[0][0] * M.mat[2][2] - M.mat[0][2] * M.mat[2][0];
	I.mat[1][2] = M.mat[0][2] * M.mat[1][0] - M.mat[0][0] * M.mat[1][2];
	I.mat[2][1] = M.mat[0][1] * M.mat[2][0] - M.mat[0][0] * M.mat[2][1];
	I.mat[2][2] = M.mat[0][0] * M.mat[1][1] - M.mat[0][1] * M.mat[1][0];
	M.mat[0][0] = I.mat[0][0] * det;
	M.mat[0][1] = I.mat[0][1] * det;
	M.mat[0][2] = I.mat[0][2] * det;
	M.mat[1][0] = I.mat[1][0] * det;
	M.mat[1][1] = I.mat[1][1] * det;
	M.mat[1][2] = I.mat[1][2] * det;
	M.mat[2][0] = I.mat[2][0] * det;
	M.mat[2][1] = I.mat[2][1] * det;
	M.mat[2][2] = I.mat[2][2] * det;
	return M;
}

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	4x4 Matrix
//
//===============================================================
/////////////////////////////////////////////////////////////////

static mat4_t _mat4_id_ = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

mat4_t mat4_identity ( void ) {
    return _mat4_id_;
}

mat4_t mat4_add ( mat4_t M, mat4_t N )
{
    return mat4_init(
        vec4_add(M.vec[0], N.vec[0]), 
        vec4_add(M.vec[1], N.vec[1]), 
        vec4_add(M.vec[2], N.vec[2]), 
        vec4_add(M.vec[3], N.vec[3])
    );
}

mat4_t mat4_sub ( mat4_t M, mat4_t N )
{
    return mat4_init(
        vec4_sub(M.vec[0], N.vec[0]), 
        vec4_sub(M.vec[1], N.vec[1]), 
        vec4_sub(M.vec[2], N.vec[2]), 
        vec4_sub(M.vec[3], N.vec[3])
    );
}

mat4_t mat4_mul( mat4_t M, mat4_t N )
{
    mat4_t O = mat4_identity();
    O.mat[0][0] = M.mat[0][0] * N.mat[0][0] + M.mat[1][0] * N.mat[0][1] + M.mat[2][0] * N.mat[0][2] + M.mat[3][0] * N.mat[0][3];
    O.mat[0][1] = M.mat[0][1] * N.mat[0][0] + M.mat[1][1] * N.mat[0][1] + M.mat[2][1] * N.mat[0][2] + M.mat[3][1] * N.mat[0][3];
    O.mat[0][2] = M.mat[0][2] * N.mat[0][0] + M.mat[1][2] * N.mat[0][1] + M.mat[2][2] * N.mat[0][2] + M.mat[3][2] * N.mat[0][3];
    O.mat[0][3] = M.mat[0][3] * N.mat[0][0] + M.mat[1][3] * N.mat[0][1] + M.mat[2][3] * N.mat[0][2] + M.mat[3][3] * N.mat[0][3];
    O.mat[1][0] = M.mat[0][0] * N.mat[1][0] + M.mat[1][0] * N.mat[1][1] + M.mat[2][0] * N.mat[1][2] + M.mat[3][0] * N.mat[1][3];
    O.mat[1][1] = M.mat[0][1] * N.mat[1][0] + M.mat[1][1] * N.mat[1][1] + M.mat[2][1] * N.mat[1][2] + M.mat[3][1] * N.mat[1][3];
    O.mat[1][2] = M.mat[0][2] * N.mat[1][0] + M.mat[1][2] * N.mat[1][1] + M.mat[2][2] * N.mat[1][2] + M.mat[3][2] * N.mat[1][3];
    O.mat[1][3] = M.mat[0][3] * N.mat[1][0] + M.mat[1][3] * N.mat[1][1] + M.mat[2][3] * N.mat[1][2] + M.mat[3][3] * N.mat[1][3];
    O.mat[2][0] = M.mat[0][0] * N.mat[2][0] + M.mat[1][0] * N.mat[2][1] + M.mat[2][0] * N.mat[2][2] + M.mat[3][0] * N.mat[2][3];
    O.mat[2][1] = M.mat[0][1] * N.mat[2][0] + M.mat[1][1] * N.mat[2][1] + M.mat[2][1] * N.mat[2][2] + M.mat[3][1] * N.mat[2][3];
    O.mat[2][2] = M.mat[0][2] * N.mat[2][0] + M.mat[1][2] * N.mat[2][1] + M.mat[2][2] * N.mat[2][2] + M.mat[3][2] * N.mat[2][3];
    O.mat[2][3] = M.mat[0][3] * N.mat[2][0] + M.mat[1][3] * N.mat[2][1] + M.mat[2][3] * N.mat[2][2] + M.mat[3][3] * N.mat[2][3];
    O.mat[3][0] = M.mat[0][0] * N.mat[3][0] + M.mat[1][0] * N.mat[3][1] + M.mat[2][0] * N.mat[3][2] + M.mat[3][0] * N.mat[3][3];
    O.mat[3][1] = M.mat[0][1] * N.mat[3][0] + M.mat[1][1] * N.mat[3][1] + M.mat[2][1] * N.mat[3][2] + M.mat[3][1] * N.mat[3][3];
    O.mat[3][2] = M.mat[0][2] * N.mat[3][0] + M.mat[1][2] * N.mat[3][1] + M.mat[2][2] * N.mat[3][2] + M.mat[3][2] * N.mat[3][3];
    O.mat[3][3] = M.mat[0][3] * N.mat[3][0] + M.mat[1][3] * N.mat[3][1] + M.mat[2][3] * N.mat[3][2] + M.mat[3][3] * N.mat[3][3];
    return O;    
}

mat4_t mat4_adds( mat4_t M, scal_t S )
{
    return mat4_init(
        vec4_adds(M.vec[0], S), 
        vec4_adds(M.vec[1], S), 
        vec4_adds(M.vec[2], S), 
        vec4_adds(M.vec[3], S)
    );
}

mat4_t mat4_subs( mat4_t M, scal_t S )
{
    return mat4_init(
        vec4_subs(M.vec[0], S), 
        vec4_subs(M.vec[1], S), 
        vec4_subs(M.vec[2], S), 
        vec4_subs(M.vec[3], S)
    );
}

mat4_t mat4_muls( mat4_t M, scal_t S )
{
    return mat4_init(
        vec4_muls(M.vec[0], S), 
        vec4_muls(M.vec[1], S), 
        vec4_muls(M.vec[2], S), 
        vec4_muls(M.vec[3], S)
    );
}

mat4_t mat4_transpose( mat4_t M )
{
    mat4_t I = M;
    I.mat[0][1] = M.mat[1][0];
    I.mat[0][2] = M.mat[2][0];
    I.mat[0][3] = M.mat[3][0];
    I.mat[1][0] = M.mat[0][1];
    I.mat[1][2] = M.mat[2][1];
    I.mat[1][3] = M.mat[3][1];
    I.mat[2][0] = M.mat[0][2];
    I.mat[2][1] = M.mat[1][2];
    I.mat[2][3] = M.mat[3][2];
    I.mat[3][0] = M.mat[0][3];
    I.mat[3][1] = M.mat[1][3];
    I.mat[3][2] = M.mat[2][3];
    return I;
}

mat4_t mat4_inverse( mat4_t M )
{
    mat4_t I = mat4_identity();  
    I.elm[0]  =  M.elm[5] * M.elm[10] * M.elm[15] - M.elm[5] * M.elm[11] * M.elm[14] - M.elm[9] * M.elm[6] * M.elm[15] + M.elm[9] * M.elm[7] * M.elm[14] + M.elm[13] * M.elm[6] * M.elm[11] - M.elm[13] * M.elm[7] * M.elm[10];
    I.elm[1]  = -M.elm[1] * M.elm[10] * M.elm[15] + M.elm[1] * M.elm[11] * M.elm[14] + M.elm[9] * M.elm[2] * M.elm[15] - M.elm[9] * M.elm[3] * M.elm[14] - M.elm[13] * M.elm[2] * M.elm[11] + M.elm[13] * M.elm[3] * M.elm[10];
    I.elm[2]  =  M.elm[1] * M.elm[ 6] * M.elm[15] - M.elm[1] * M.elm[ 7] * M.elm[14] - M.elm[5] * M.elm[2] * M.elm[15] + M.elm[5] * M.elm[3] * M.elm[14] + M.elm[13] * M.elm[2] * M.elm[ 7] - M.elm[13] * M.elm[3] * M.elm[ 6];
    I.elm[3]  = -M.elm[1] * M.elm[ 6] * M.elm[11] + M.elm[1] * M.elm[ 7] * M.elm[10] + M.elm[5] * M.elm[2] * M.elm[11] - M.elm[5] * M.elm[3] * M.elm[10] - M.elm[ 9] * M.elm[2] * M.elm[ 7] + M.elm[ 9] * M.elm[3] * M.elm[ 6];
    I.elm[4]  = -M.elm[4] * M.elm[10] * M.elm[15] + M.elm[4] * M.elm[11] * M.elm[14] + M.elm[8] * M.elm[6] * M.elm[15] - M.elm[8] * M.elm[7] * M.elm[14] - M.elm[12] * M.elm[6] * M.elm[11] + M.elm[12] * M.elm[7] * M.elm[10];
    I.elm[5]  =  M.elm[0] * M.elm[10] * M.elm[15] - M.elm[0] * M.elm[11] * M.elm[14] - M.elm[8] * M.elm[2] * M.elm[15] + M.elm[8] * M.elm[3] * M.elm[14] + M.elm[12] * M.elm[2] * M.elm[11] - M.elm[12] * M.elm[3] * M.elm[10];
    I.elm[6]  = -M.elm[0] * M.elm[ 6] * M.elm[15] + M.elm[0] * M.elm[ 7] * M.elm[14] + M.elm[4] * M.elm[2] * M.elm[15] - M.elm[4] * M.elm[3] * M.elm[14] - M.elm[12] * M.elm[2] * M.elm[ 7] + M.elm[12] * M.elm[3] * M.elm[ 6];
    I.elm[7]  =  M.elm[0] * M.elm[ 6] * M.elm[11] - M.elm[0] * M.elm[ 7] * M.elm[10] - M.elm[4] * M.elm[2] * M.elm[11] + M.elm[4] * M.elm[3] * M.elm[10] + M.elm[ 8] * M.elm[2] * M.elm[ 7] - M.elm[ 8] * M.elm[3] * M.elm[ 6];
    I.elm[8]  =  M.elm[4] * M.elm[ 9] * M.elm[15] - M.elm[4] * M.elm[11] * M.elm[13] - M.elm[8] * M.elm[5] * M.elm[15] + M.elm[8] * M.elm[7] * M.elm[13] + M.elm[12] * M.elm[5] * M.elm[11] - M.elm[12] * M.elm[7] * M.elm[ 9];
    I.elm[9]  = -M.elm[0] * M.elm[ 9] * M.elm[15] + M.elm[0] * M.elm[11] * M.elm[13] + M.elm[8] * M.elm[1] * M.elm[15] - M.elm[8] * M.elm[3] * M.elm[13] - M.elm[12] * M.elm[1] * M.elm[11] + M.elm[12] * M.elm[3] * M.elm[ 9];
    I.elm[10] =  M.elm[0] * M.elm[ 5] * M.elm[15] - M.elm[0] * M.elm[ 7] * M.elm[13] - M.elm[4] * M.elm[1] * M.elm[15] + M.elm[4] * M.elm[3] * M.elm[13] + M.elm[12] * M.elm[1] * M.elm[ 7] - M.elm[12] * M.elm[3] * M.elm[ 5];
    I.elm[11] = -M.elm[0] * M.elm[ 5] * M.elm[11] + M.elm[0] * M.elm[ 7] * M.elm[ 9] + M.elm[4] * M.elm[1] * M.elm[11] - M.elm[4] * M.elm[3] * M.elm[ 9] - M.elm[ 8] * M.elm[1] * M.elm[ 7] + M.elm[ 8] * M.elm[3] * M.elm[ 5];
    I.elm[12] = -M.elm[4] * M.elm[ 9] * M.elm[14] + M.elm[4] * M.elm[10] * M.elm[13] + M.elm[8] * M.elm[5] * M.elm[14] - M.elm[8] * M.elm[6] * M.elm[13] - M.elm[12] * M.elm[5] * M.elm[10] + M.elm[12] * M.elm[6] * M.elm[ 9];
    I.elm[13] =  M.elm[0] * M.elm[ 9] * M.elm[14] - M.elm[0] * M.elm[10] * M.elm[13] - M.elm[8] * M.elm[1] * M.elm[14] + M.elm[8] * M.elm[2] * M.elm[13] + M.elm[12] * M.elm[1] * M.elm[10] - M.elm[12] * M.elm[2] * M.elm[ 9];
    I.elm[14] = -M.elm[0] * M.elm[ 5] * M.elm[14] + M.elm[0] * M.elm[ 6] * M.elm[13] + M.elm[4] * M.elm[1] * M.elm[14] - M.elm[4] * M.elm[2] * M.elm[13] - M.elm[12] * M.elm[1] * M.elm[ 6] + M.elm[12] * M.elm[2] * M.elm[ 5];
    I.elm[15] =  M.elm[0] * M.elm[ 5] * M.elm[10] - M.elm[0] * M.elm[ 6] * M.elm[ 9] - M.elm[4] * M.elm[1] * M.elm[10] + M.elm[4] * M.elm[2] * M.elm[ 9] + M.elm[ 8] * M.elm[1] * M.elm[ 6] - M.elm[ 8] * M.elm[2] * M.elm[ 5];
    scal_t det = M.elm[0] * I.elm[0] + M.elm[1] * I.elm[4] + M.elm[2] * I.elm[8] + M.elm[3] * I.elm[12];

    if ( fabs(det) < 1e-10 ) {
        return mat4_identity();
    }

    det = 1.0 / det;
    return mat4_muls(I, det);
}


mat4_t mat4_translate ( mat4_t M, vec3_t V )
{
    mat4_t T = mat4_identity();
    T.mat[0][3] = V.x;
    T.mat[1][3] = V.y;
    T.mat[2][3] = V.z;
    return mat4_mul ( M, T );
}

mat4_t mat4_scale ( mat4_t M, vec3_t V )
{
    mat4_t S = mat4_identity();
    S.mat[0][0] = V.x;
    S.mat[1][1] = V.y;
    S.mat[2][2] = V.z;
    return mat4_mul ( M, S );
}


mat4_t mat4_rotate( vec3_t v )
{
    mat4_t m = mat4_identity();
    scal_t cosx = cosf(v.x);
    scal_t cosy = cosf(v.y);
    scal_t cosz = cosf(v.z);
    scal_t sinx = sinf(v.x);
    scal_t siny = sinf(v.y);
    scal_t sinz = sinf(v.z);
    m.mat[0][0] =  cosy * cosz;
    m.mat[1][0] = -cosx * sinz + sinx * siny * cosz;
    m.mat[2][0] =  sinx * sinz + cosx * siny * cosz;
    m.mat[0][1] =  cosy * sinz;
    m.mat[1][1] =  cosx * cosz + sinx * siny * sinz;
    m.mat[2][1] = -sinx * cosz + cosx * siny * sinz;
    m.mat[0][2] = -siny;
    m.mat[1][2] =  sinx * cosy;
    m.mat[2][2] =  cosx * cosy;
    m.mat[3][3] = 1;  
    return m;
}

mat4_t mat4_ortho(scal_t left, scal_t right, scal_t bot, scal_t top, scal_t near, scal_t far)
{
    mat4_t O = mat4_identity();
    scal_t lr = 1.0f / (left - right);
    scal_t bt = 1.0f / (bot  - top);
    scal_t nf = 1.0f / (near - far);
    O.vec[0].x = (-2.0f * lr);
    O.vec[1].y = (-2.0f * bt);
    O.vec[2].z = ( 2.0f * nf);
    O.vec[3].x = (left + right) * lr;
    O.vec[3].y = (top  + bot  ) * bt;
    O.vec[3].z = (far  + near ) * nf;
    return O;
}

mat4_t mat4_perspective(scal_t fov, scal_t aspect_ratio, scal_t near, scal_t far)
{
    scal_t hfov = tan(fov * 0.5f);
    mat4_t P = mat4_identity();   
    P.vec[0].x = (1.0f / (aspect_ratio * hfov));
    P.vec[1].y = (1.0f / hfov);
    P.vec[2].z = (-((far + near) / (far - near)));
    P.vec[2].w = (-1.0f);
    P.vec[3].z = (-((2.0f * far * near) / (far - near)));
    return P;
}

mat4_t mat4_lookat (vec3_t eye, vec3_t center, vec3_t up)
{
    mat4_t result = mat4_identity();   
    vec3_t f = vec3_normal(vec3_sub(center, eye));
    vec3_t s = vec3_normal(vec3_cross(f, up));
    vec3_t u = vec3_cross(s, f);    
    result.mat[0][0] =  s.x;
    result.mat[0][1] =  u.x;
    result.mat[0][2] = -f.x;
    result.mat[0][3] =  0.0f;    
    result.mat[1][0] =  s.y;
    result.mat[1][1] =  u.y;
    result.mat[1][2] = -f.y;
    result.mat[1][3] =  0.0f;
    result.mat[2][0] =  s.z;
    result.mat[2][1] =  u.z;
    result.mat[2][2] = -f.z;
    result.mat[2][3] =  0.0f;
    result.mat[3][0] = -vec3_dot(s, eye);
    result.mat[3][1] =  vec3_dot(u, eye);
    result.mat[3][2] =  vec3_dot(f, eye);
    result.mat[3][3] =  1.0f;
    return result;
}





#ifdef __cplusplus
}
#endif // C++


