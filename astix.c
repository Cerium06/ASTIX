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

vec2_t vec2_add  ( vec2_t u, vec2_t v ) {
    return vec2_init ( u.x + v.x, u.y + v.y );
}

vec2_t vec2_sub  ( vec2_t u, vec2_t v ) {
    return vec2_init ( u.x - v.x, u.y - v.y );
}

vec2_t vec2_mul  ( vec2_t u, vec2_t v ) {
    return vec2_init ( u.x * v.x, u.y * v.y );
}

vec2_t vec2_adds ( vec2_t u, scal_t s ) {
    return vec2_init ( u.x + s, u.y + s );
}

vec2_t vec2_subs ( vec2_t u, scal_t s ) {
    return vec2_init ( u.x - s, u.y - s );
}

vec2_t vec2_muls ( vec2_t u, scal_t s ) {
    return vec2_init ( u.x * s, u.y * s );
}

scal_t vec2_dot ( vec2_t u, vec2_t v ) {
    return  ( u.x * v.x ) +
            ( u.y * v.y );
}

scal_t vec2_lensqr ( vec2_t u ) {
    return ( u.x * u.x ) +
           ( u.y * u.y );
}

scal_t vec2_invlength ( vec2_t u ) {
    return invsqrt ( vec2_lensqr (u) );
}

scal_t vec2_length ( vec2_t u ) {
    return sqrtf( vec2_lensqr(u) );
}

vec2_t vec2_normal ( vec2_t u ) {
    return vec2_muls(u, vec2_invlength(u) );
}

vec2_t vec2_mul_mat2( vec2_t u, mat2_t m )
{
	return vec2_init (
        m.vec[0].x * u.x + m.vec[0].y * u.y,
	    m.vec[1].x * u.x + m.vec[1].y * u.y);
}

vec2_t vec2_polar ( float dist, float ang )
{
    return vec2_init (
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

vec3_t vec3_add  ( vec3_t u, vec3_t v ) {
    return vec3_init ( u.x + v.x, u.y + v.y, u.z + v.z );
}

vec3_t vec3_sub  ( vec3_t u, vec3_t v ) {
    return vec3_init ( u.x - v.x, u.y - v.y, u.z - v.z );
}

vec3_t vec3_mul  ( vec3_t u, vec3_t v ) {
    return vec3_init ( u.x * v.x, u.y * v.y, u.z * v.z );
}

vec3_t vec3_adds ( vec3_t u, scal_t s ) {
    return vec3_init ( u.x + s, u.y + s, u.z + s );
}

vec3_t vec3_subs ( vec3_t u, scal_t s ) {
    return vec3_init ( u.x - s, u.y - s, u.z - s );
}

vec3_t vec3_muls ( vec3_t u, scal_t s ) {
    return vec3_init ( u.x * s, u.y * s, u.z * s );
}


scal_t vec3_dot ( vec3_t u, vec3_t v ) {
    return  ( u.x * v.x ) +
            ( u.y * v.y ) + 
            ( u.z * v.z );
}

scal_t vec3_lensqr ( vec3_t u ) {
    return (u.x * u.x) +
           (u.y * u.y) +
           (u.z * u.z);
}

scal_t vec3_invlength ( vec3_t u ) {
    return invsqrt( vec3_lensqr(u) );
}

scal_t vec3_length ( vec3_t u ) {
    return sqrtf( vec3_lensqr(u) );
}

vec3_t vec3_normal ( vec3_t u ) {
    return vec3_muls(u, vec3_invlength(u) );
}

vec3_t vec3_cross ( vec3_t u, vec3_t v ) {
    return vec3_init (
        (u.y * v.z) - (u.z * v.y),
        (u.x * v.z) - (u.z * v.x),
        (u.x * v.y) - (u.y * v.x)
    );
}


vec3_t vec3_mul_mat3 ( vec3_t u, mat3_t m ) {
	return vec3_init (
        m.vec[0].x * u.x + m.vec[0].y * u.y + m.vec[0].z * u.z,
	    m.vec[1].x * u.x + m.vec[1].y * u.y + m.vec[1].z * u.z,
	    m.vec[2].x * u.x + m.vec[2].y * u.y + m.vec[2].z * u.z
    );
}

vec3_t vec3_mul_mat4 ( vec3_t V, mat4_t M )
{
    scal_t s =  1.0f/ (M.vec[3].x * V.x + M.vec[3].w  * V.y + M.vec[3].z * V.z + M.vec[3].w);
    
    vec3_t U = vec3_init (
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

vec4_t vec4_add  ( vec4_t u, vec4_t v ) {
    return vec4_init ( u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w );
}

vec4_t vec4_sub  ( vec4_t u, vec4_t v ) {
    return vec4_init ( u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w );
}

vec4_t vec4_mul  ( vec4_t u, vec4_t v ) {
    return vec4_init ( u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w );
}

vec4_t vec4_adds ( vec4_t u, scal_t s ) {
    return vec4_init ( u.x + s, u.y + s, u.z + s, u.w + s );
}

vec4_t vec4_subs ( vec4_t u, scal_t s ) {
    return vec4_init ( u.x - s, u.y - s, u.z - s, u.w - s );
}

vec4_t vec4_muls ( vec4_t u, scal_t s ) {
    return vec4_init ( u.x * s, u.y * s, u.z * s, u.w * s );
}

scal_t vec4_dot ( vec4_t u, vec4_t v ) {
    return  ( u.x * v.x ) + 
            ( u.y * v.y ) +
            ( u.z * v.z ) + 
            ( u.w * v.w );
}

scal_t vec4_lensqr ( vec4_t u ) {
    return (u.x * u.x) +
           (u.y * u.y) +
           (u.z * u.z) +
           (u.w * u.w);
}

scal_t vec4_invlength ( vec4_t u ) {
    return invsqrt( vec4_lensqr(u) );
}

scal_t vec4_length ( vec4_t u ) {
    return sqrtf( vec4_lensqr(u) );
}

vec4_t vec4_normal ( vec4_t u ) {
    return vec4_muls(u, vec4_invlength(u));
}

vec4_t vec4_mul_mat4 ( vec4_t u, mat4_t m ) {
	return vec4_init (
        (m.vec[0].x * u.x) + (m.vec[0].y * u.y) + (m.vec[0].z * u.z) + (m.vec[0].w * u.w),
	    (m.vec[1].x * u.x) + (m.vec[1].y * u.y) + (m.vec[1].z * u.z) + (m.vec[1].w * u.w),
	    (m.vec[2].x * u.x) + (m.vec[2].y * u.y) + (m.vec[2].z * u.z) + (m.vec[2].w * u.w),
	    (m.vec[3].x * u.x) + (m.vec[3].y * u.y) + (m.vec[3].z * u.z) + (m.vec[3].w * u.w));
}

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	Quaternion
//
//===============================================================
/////////////////////////////////////////////////////////////////

quat_t quat_add  ( quat_t u, quat_t v ) {
    return quat_init ( u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w );
}

quat_t quat_sub  ( quat_t u, quat_t v ) {
    return quat_init ( u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w );
}

quat_t quat_mul  ( quat_t q, quat_t r ) {
    return quat_init (
        q.w * r.x + q.x * r.w + q.y * r.z - q.z * r.y,
        q.w * r.y + q.y * r.w + q.z * r.x - q.x * r.z,
        q.w * r.z + q.z * r.w + q.x * r.y - q.y * r.x,
        q.w * r.w - q.x * r.x - q.y * r.y - q.z * r.z
    );
}

quat_t quat_adds ( quat_t u, scal_t s ) {
    return quat_init ( u.x + s, u.y + s, u.z + s, u.w + s );
}

quat_t quat_subs ( quat_t u, scal_t s ) {
    return quat_init ( u.x - s, u.y - s, u.z - s, u.w - s );
}

quat_t quat_muls ( quat_t u, scal_t s ) {
    return quat_init ( u.x * s, u.y * s, u.z * s, u.w * s );
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

mat2_t mat2_add ( mat2_t m, mat2_t n ) {
    return mat2_init(
        vec2_add(m.vec[0], n.vec[0]),
        vec2_add(m.vec[1], n.vec[1])
    );
}

mat2_t mat2_sub ( mat2_t m, mat2_t n ) {
    return mat2_init(
        vec2_sub(m.vec[0], n.vec[0]),
        vec2_sub(m.vec[1], n.vec[1])
    );
}

mat2_t mat2_mul( mat2_t m, mat2_t n ) {
    mat2_t o = mat2_identity();    
    o.mat[0][0] = m.mat[0][0] * n.mat[0][0] + m.mat[1][0] * n.mat[0][1];
    o.mat[0][1] = m.mat[0][1] * n.mat[0][0] + m.mat[1][1] * n.mat[0][1];
    o.mat[1][0] = m.mat[0][0] * n.mat[1][0] + m.mat[1][0] * n.mat[1][1];
    o.mat[1][1] = m.mat[0][1] * n.mat[1][0] + m.mat[1][1] * n.mat[1][1];
    return o;
}

mat2_t mat2_adds ( mat2_t m, scal_t s ) {
    return mat2_init(
        vec2_adds( m.vec[0], s ),
        vec2_adds( m.vec[1], s )
    );
}

mat2_t mat2_subs ( mat2_t m, scal_t s ) {
    return mat2_init(
        vec2_subs( m.vec[0], s ),
        vec2_subs( m.vec[1], s )
    );
}

mat2_t mat2_muls ( mat2_t m, scal_t s ) {
    return mat2_init(
        vec2_muls(m.vec[0], s),
        vec2_muls(m.vec[1], s)
    );
}

mat2_t mat2_transpose ( mat2_t m ) {
    mat2_t t = m; 
    t.elm[1] = m.elm[2];
    t.elm[2] = m.elm[1];
    return t;
}

mat2_t mat2_inverse( mat2_t m ) {
    mat2_t i = mat2_identity();
    scal_t det = m.mat[0][0] * m.mat[1][1] - m.mat[0][1] * m.mat[1][0];
    
    if( fabs(det) < 1e-10 )
    {
        return i;
    }
    
    det = 1.0f/det;
    i.elm[0] = det * +m.mat[1][1];
    i.elm[1] = det * -m.mat[0][1];
    i.elm[2] = det * -m.mat[1][0];
    i.elm[3] = det * +m.mat[0][0];
    return i;
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

mat3_t mat3_add ( mat3_t m, mat3_t n ) {
    return mat3_init(
        vec3_add(m.vec[0], n.vec[0]),
        vec3_add(m.vec[1], n.vec[1]),
        vec3_add(m.vec[2], n.vec[2])
    );
}

mat3_t mat3_sub ( mat3_t m, mat3_t n ) {
    return mat3_init (
        vec3_sub(m.vec[0], n.vec[0]), 
        vec3_sub(m.vec[1], n.vec[1]), 
        vec3_sub(m.vec[2], n.vec[2])
    );
}

mat3_t mat3_mul ( mat3_t m, mat3_t n ) {
    mat3_t o = mat3_identity();
    o.mat[0][0] = m.mat[0][0] * n.mat[0][0] + m.mat[1][0] * n.mat[0][1] + m.mat[2][0] * n.mat[0][2];
    o.mat[0][1] = m.mat[0][1] * n.mat[0][0] + m.mat[1][1] * n.mat[0][1] + m.mat[2][1] * n.mat[0][2];
    o.mat[0][2] = m.mat[0][2] * n.mat[0][0] + m.mat[1][2] * n.mat[0][1] + m.mat[2][2] * n.mat[0][2];
    o.mat[1][0] = m.mat[0][0] * n.mat[1][0] + m.mat[1][0] * n.mat[1][1] + m.mat[2][0] * n.mat[1][2];
    o.mat[1][1] = m.mat[0][1] * n.mat[1][0] + m.mat[1][1] * n.mat[1][1] + m.mat[2][1] * n.mat[1][2];
    o.mat[1][2] = m.mat[0][2] * n.mat[1][0] + m.mat[1][2] * n.mat[1][1] + m.mat[2][2] * n.mat[1][2];
    o.mat[2][0] = m.mat[0][0] * n.mat[2][0] + m.mat[1][0] * n.mat[2][1] + m.mat[2][0] * n.mat[2][2];
    o.mat[2][1] = m.mat[0][1] * n.mat[2][0] + m.mat[1][1] * n.mat[2][1] + m.mat[2][1] * n.mat[2][2];
    o.mat[2][2] = m.mat[0][2] * n.mat[2][0] + m.mat[1][2] * n.mat[2][1] + m.mat[2][2] * n.mat[2][2];
    return o;    
}

mat3_t mat3_adds ( mat3_t m, scal_t s )
{
    return mat3_init(
        vec3_adds(m.vec[0], s), 
        vec3_adds(m.vec[1], s), 
        vec3_adds(m.vec[2], s)
    );
}

mat3_t mat3_subs ( mat3_t m, scal_t s ) {
    return mat3_init(
        vec3_subs(m.vec[0], s), 
        vec3_subs(m.vec[1], s), 
        vec3_subs(m.vec[2], s)
    );
}

mat3_t mat3_muls ( mat3_t m, scal_t s ) {
    return mat3_init(
        vec3_muls(m.vec[0], s), 
        vec3_muls(m.vec[1], s), 
        vec3_muls(m.vec[2], s)
    );
}

mat3_t mat3_transpose ( mat3_t M ) {
    mat3_t T = M;
    T.mat[0][1] = M.mat[1][0];
    T.mat[0][2] = M.mat[2][0];
    T.mat[1][0] = M.mat[0][1];
    T.mat[1][2] = M.mat[2][1];
    T.mat[2][0] = M.mat[0][2];
    T.mat[2][1] = M.mat[1][2];
    return T;
}

mat3_t mat3_inverse ( mat3_t m ) {
	mat3_t i = mat3_identity();
	double det = 0.0f;

	i.mat[0][0] = m.mat[1][1] * m.mat[2][2] - m.mat[1][2] * m.mat[2][1];
	i.mat[1][0] = m.mat[1][2] * m.mat[2][0] - m.mat[1][0] * m.mat[2][2];
	i.mat[2][0] = m.mat[1][0] * m.mat[2][1] - m.mat[1][1] * m.mat[2][0];

	det = m.mat[0][0] * i.mat[0][0] + m.mat[0][1] * i.mat[1][0] + m.mat[0][2] * i.mat[2][0];

	if ( fabs(det) < 1e-10 ) {
		return i;
	}

	det = 1.0f / det;
	i.mat[0][1] = m.mat[0][2] * m.mat[2][1] - m.mat[0][1] * m.mat[2][2];
	i.mat[0][2] = m.mat[0][1] * m.mat[1][2] - m.mat[0][2] * m.mat[1][1];
	i.mat[1][1] = m.mat[0][0] * m.mat[2][2] - m.mat[0][2] * m.mat[2][0];
	i.mat[1][2] = m.mat[0][2] * m.mat[1][0] - m.mat[0][0] * m.mat[1][2];
	i.mat[2][1] = m.mat[0][1] * m.mat[2][0] - m.mat[0][0] * m.mat[2][1];
	i.mat[2][2] = m.mat[0][0] * m.mat[1][1] - m.mat[0][1] * m.mat[1][0];
	m.mat[0][0] = i.mat[0][0] * det;
	m.mat[0][1] = i.mat[0][1] * det;
	m.mat[0][2] = i.mat[0][2] * det;
	m.mat[1][0] = i.mat[1][0] * det;
	m.mat[1][1] = i.mat[1][1] * det;
	m.mat[1][2] = i.mat[1][2] * det;
	m.mat[2][0] = i.mat[2][0] * det;
	m.mat[2][1] = i.mat[2][1] * det;
	m.mat[2][2] = i.mat[2][2] * det;
	return m;
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

mat4_t mat4_add ( mat4_t m, mat4_t n ) {
    return mat4_init(
        vec4_add(m.vec[0], n.vec[0]), 
        vec4_add(m.vec[1], n.vec[1]), 
        vec4_add(m.vec[2], n.vec[2]), 
        vec4_add(m.vec[3], n.vec[3])
    );
}

mat4_t mat4_sub ( mat4_t m, mat4_t n ) {
    return mat4_init(
        vec4_sub(m.vec[0], n.vec[0]), 
        vec4_sub(m.vec[1], n.vec[1]), 
        vec4_sub(m.vec[2], n.vec[2]), 
        vec4_sub(m.vec[3], n.vec[3])
    );
}

mat4_t mat4_mul ( mat4_t m, mat4_t n ) {
    mat4_t o = mat4_identity();
    o.mat[0][0] = m.mat[0][0] * n.mat[0][0] + m.mat[1][0] * n.mat[0][1] + m.mat[2][0] * n.mat[0][2] + m.mat[3][0] * n.mat[0][3];
    o.mat[0][1] = m.mat[0][1] * n.mat[0][0] + m.mat[1][1] * n.mat[0][1] + m.mat[2][1] * n.mat[0][2] + m.mat[3][1] * n.mat[0][3];
    o.mat[0][2] = m.mat[0][2] * n.mat[0][0] + m.mat[1][2] * n.mat[0][1] + m.mat[2][2] * n.mat[0][2] + m.mat[3][2] * n.mat[0][3];
    o.mat[0][3] = m.mat[0][3] * n.mat[0][0] + m.mat[1][3] * n.mat[0][1] + m.mat[2][3] * n.mat[0][2] + m.mat[3][3] * n.mat[0][3];
    o.mat[1][0] = m.mat[0][0] * n.mat[1][0] + m.mat[1][0] * n.mat[1][1] + m.mat[2][0] * n.mat[1][2] + m.mat[3][0] * n.mat[1][3];
    o.mat[1][1] = m.mat[0][1] * n.mat[1][0] + m.mat[1][1] * n.mat[1][1] + m.mat[2][1] * n.mat[1][2] + m.mat[3][1] * n.mat[1][3];
    o.mat[1][2] = m.mat[0][2] * n.mat[1][0] + m.mat[1][2] * n.mat[1][1] + m.mat[2][2] * n.mat[1][2] + m.mat[3][2] * n.mat[1][3];
    o.mat[1][3] = m.mat[0][3] * n.mat[1][0] + m.mat[1][3] * n.mat[1][1] + m.mat[2][3] * n.mat[1][2] + m.mat[3][3] * n.mat[1][3];
    o.mat[2][0] = m.mat[0][0] * n.mat[2][0] + m.mat[1][0] * n.mat[2][1] + m.mat[2][0] * n.mat[2][2] + m.mat[3][0] * n.mat[2][3];
    o.mat[2][1] = m.mat[0][1] * n.mat[2][0] + m.mat[1][1] * n.mat[2][1] + m.mat[2][1] * n.mat[2][2] + m.mat[3][1] * n.mat[2][3];
    o.mat[2][2] = m.mat[0][2] * n.mat[2][0] + m.mat[1][2] * n.mat[2][1] + m.mat[2][2] * n.mat[2][2] + m.mat[3][2] * n.mat[2][3];
    o.mat[2][3] = m.mat[0][3] * n.mat[2][0] + m.mat[1][3] * n.mat[2][1] + m.mat[2][3] * n.mat[2][2] + m.mat[3][3] * n.mat[2][3];
    o.mat[3][0] = m.mat[0][0] * n.mat[3][0] + m.mat[1][0] * n.mat[3][1] + m.mat[2][0] * n.mat[3][2] + m.mat[3][0] * n.mat[3][3];
    o.mat[3][1] = m.mat[0][1] * n.mat[3][0] + m.mat[1][1] * n.mat[3][1] + m.mat[2][1] * n.mat[3][2] + m.mat[3][1] * n.mat[3][3];
    o.mat[3][2] = m.mat[0][2] * n.mat[3][0] + m.mat[1][2] * n.mat[3][1] + m.mat[2][2] * n.mat[3][2] + m.mat[3][2] * n.mat[3][3];
    o.mat[3][3] = m.mat[0][3] * n.mat[3][0] + m.mat[1][3] * n.mat[3][1] + m.mat[2][3] * n.mat[3][2] + m.mat[3][3] * n.mat[3][3];
    return o;    
}

mat4_t mat4_adds ( mat4_t m, scal_t s ) {
    return mat4_init(
        vec4_adds(m.vec[0], s), 
        vec4_adds(m.vec[1], s), 
        vec4_adds(m.vec[2], s), 
        vec4_adds(m.vec[3], s)
    );
}

mat4_t mat4_subs ( mat4_t m, scal_t s ) {
    return mat4_init(
        vec4_subs(m.vec[0], s), 
        vec4_subs(m.vec[1], s), 
        vec4_subs(m.vec[2], s), 
        vec4_subs(m.vec[3], s)
    );
}

mat4_t mat4_muls ( mat4_t m, scal_t s ) {
    return mat4_init(
        vec4_muls(m.vec[0], s), 
        vec4_muls(m.vec[1], s), 
        vec4_muls(m.vec[2], s), 
        vec4_muls(m.vec[3], s)
    );
}

mat4_t mat4_transpose ( mat4_t m ) {
    mat4_t i = m;
    i.mat[0][1] = m.mat[1][0];
    i.mat[0][2] = m.mat[2][0];
    i.mat[0][3] = m.mat[3][0];
    i.mat[1][0] = m.mat[0][1];
    i.mat[1][2] = m.mat[2][1];
    i.mat[1][3] = m.mat[3][1];
    i.mat[2][0] = m.mat[0][2];
    i.mat[2][1] = m.mat[1][2];
    i.mat[2][3] = m.mat[3][2];
    i.mat[3][0] = m.mat[0][3];
    i.mat[3][1] = m.mat[1][3];
    i.mat[3][2] = m.mat[2][3];
    return i;
}

mat4_t mat4_inverse ( mat4_t M )
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
        return mat4_identity ();
    }

    det = 1.0 / det;
    return mat4_muls(I, det);
}


mat4_t mat4_translate ( mat4_t m, vec3_t v )
{
    mat4_t t = mat4_identity();
    t.mat[0][3] = v.x;
    t.mat[1][3] = v.y;
    t.mat[2][3] = v.z;
    return mat4_mul (m, t);
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

mat4_t mat4_ortho ( scal_t left, scal_t right, scal_t bot, scal_t top, scal_t near, scal_t far )
{
    mat4_t o = mat4_identity();
    scal_t lr = 1.0f / (left - right);
    scal_t bt = 1.0f / (bot  - top);
    scal_t nf = 1.0f / (near - far);
    o.vec[0].x = (-2.0f * lr);
    o.vec[1].y = (-2.0f * bt);
    o.vec[2].z = ( 2.0f * nf);
    o.vec[3].x = (left + right) * lr;
    o.vec[3].y = (top  + bot  ) * bt;
    o.vec[3].z = (far  + near ) * nf;
    return o;
}

mat4_t mat4_perspective ( scal_t fov, scal_t aspect_ratio, scal_t near, scal_t far )
{
    mat4_t p = mat4_identity();   
    scal_t hfov = tan(fov * 0.5f);
    p.vec[0].x = (1.0f / (aspect_ratio * hfov));
    p.vec[1].y = (1.0f / hfov);
    p.vec[2].z = (-((far + near) / (far - near)));
    p.vec[2].w = (-1.0f);
    p.vec[3].z = (-((2.0f * far * near) / (far - near)));
    return p;
}

mat4_t mat4_lookat (vec3_t eye, vec3_t center, vec3_t up )
{
    mat4_t l = mat4_identity();   
    vec3_t f = vec3_normal(vec3_sub(center, eye));
    vec3_t s = vec3_normal(vec3_cross(f, up));
    vec3_t u = vec3_cross(s, f);    
    l.mat[0][0] =  s.x;
    l.mat[0][1] =  u.x;
    l.mat[0][2] = -f.x;
    l.mat[0][3] =  0.0f;    
    l.mat[1][0] =  s.y;
    l.mat[1][1] =  u.y;
    l.mat[1][2] = -f.y;
    l.mat[1][3] =  0.0f;
    l.mat[2][0] =  s.z;
    l.mat[2][1] =  u.z;
    l.mat[2][2] = -f.z;
    l.mat[2][3] =  0.0f;
    l.mat[3][0] = -vec3_dot(s, eye);
    l.mat[3][1] =  vec3_dot(u, eye);
    l.mat[3][2] =  vec3_dot(f, eye);
    l.mat[3][3] =  1.0f;
    return l;
}





#ifdef __cplusplus
}
#endif // C++


