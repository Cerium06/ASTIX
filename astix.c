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

vec2_t vec2_add         ( vec2_t a, vec2_t b ) { return vec2_init ( a.x + b.x, a.y + b.y ); }
vec2_t vec2_sub         ( vec2_t a, vec2_t b ) { return vec2_init ( a.x - b.x, a.y - b.y ); }
vec2_t vec2_mul         ( vec2_t a, vec2_t b ) { return vec2_init ( a.x * b.x, a.y * b.y ); }
vec2_t vec2_adds        ( vec2_t a, scal_t b ) { return vec2_init ( a.x + b,   a.y + b ); }
vec2_t vec2_subs        ( vec2_t a, scal_t b ) { return vec2_init ( a.x - b,   a.y - b ); }
vec2_t vec2_muls        ( vec2_t a, scal_t b ) { return vec2_init ( a.x * b,   a.y * b ); }
scal_t vec2_dot         ( vec2_t a, vec2_t b ) { return  ( a.x * b.x ) + ( a.y * b.y ); }
scal_t vec2_lensqr      ( vec2_t a ) { return ( a.x * a.x ) + ( a.y * a.y ); }
scal_t vec2_invlength   ( vec2_t a ) { return invsqrt ( vec2_lensqr (a) ); }
scal_t vec2_length      ( vec2_t a ) { return sqrtf( vec2_lensqr(a) ); }
vec2_t vec2_normal      ( vec2_t a ) { return vec2_muls(a, vec2_invlength(a) ); }

vec2_t vec2_mul_mat2( vec2_t u, mat2_t m )
{
	return vec2_init (
        m.vec[0].x * u.x + m.vec[0].y * u.y,
	    m.vec[1].x * u.x + m.vec[1].y * u.y);
}

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	3D Vector
//
//===============================================================
/////////////////////////////////////////////////////////////////

vec3_t vec3_add       ( vec3_t a, vec3_t b ) { return vec3_init ( a.x + b.x, a.y + b.y, a.z + b.z );  }
vec3_t vec3_sub       ( vec3_t a, vec3_t b ) { return vec3_init ( a.x - b.x, a.y - b.y, a.z - b.z );  }
vec3_t vec3_mul       ( vec3_t a, vec3_t b ) { return vec3_init ( a.x * b.x, a.y * b.y, a.z * b.z );  }
vec3_t vec3_adds      ( vec3_t a, scal_t b ) { return vec3_init ( a.x + b,   a.y + b,   a.z + b );    }
vec3_t vec3_subs      ( vec3_t a, scal_t b ) { return vec3_init ( a.x - b,   a.y - b,   a.z - b );    }
vec3_t vec3_muls      ( vec3_t a, scal_t b ) { return vec3_init ( a.x * b,   a.y * b,   a.z * b );    }
scal_t vec3_dot       ( vec3_t a, vec3_t b ) { return  ( a.x * b.x ) + ( a.y * b.y ) + ( a.z * b.z ); }
scal_t vec3_lensqr    ( vec3_t a )           { return (a.x * a.x) + (a.y * a.y) + (a.z * a.z); }
scal_t vec3_invlength ( vec3_t a )           { return invsqrt( vec3_lensqr(a) ); }
scal_t vec3_length    ( vec3_t a )           { return sqrtf( vec3_lensqr(a) ); }
vec3_t vec3_normal    ( vec3_t a )           { return vec3_muls(a, vec3_invlength(a) ); }

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

vec4_t vec4_add         ( vec4_t a, vec4_t b ) { return vec4_init ( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }
vec4_t vec4_sub         ( vec4_t a, vec4_t b ) { return vec4_init ( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); }
vec4_t vec4_mul         ( vec4_t a, vec4_t b ) { return vec4_init ( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w ); }
vec4_t vec4_adds        ( vec4_t a, scal_t b ) { return vec4_init ( a.x + b, a.y + b, a.z + b, a.w + b ); }
vec4_t vec4_subs        ( vec4_t a, scal_t b ) { return vec4_init ( a.x - b, a.y - b, a.z - b, a.w - b ); }
vec4_t vec4_muls        ( vec4_t a, scal_t b ) { return vec4_init ( a.x * b, a.y * b, a.z * b, a.w * b ); }
scal_t vec4_dot         ( vec4_t a, vec4_t b ) { return  ( a.x * b.x ) +  ( a.y * b.y ) + ( a.z * b.z ) +  ( a.w * b.w ); }
scal_t vec4_lensqr      ( vec4_t a ) { return (a.x * a.x) + (a.y * a.y) + (a.z * a.z) + (a.w * a.w); } 
scal_t vec4_invlength   ( vec4_t a ) { return invsqrt( vec4_lensqr(a) ); }
scal_t vec4_length      ( vec4_t a ) { return sqrtf( vec4_lensqr(a) ); }
vec4_t vec4_normal      ( vec4_t a ) { return vec4_muls(a, vec4_invlength(a)); }

vec4_t vec4_mul_mat4 ( vec4_t u, mat4_t m ) {
    vec4_t p = vec4_init(0.0f, 0.0f, 0.0f, 0.0f);
    p.x = (m.vec[0].x * u.x) + (m.vec[0].y * u.y) + (m.vec[0].z * u.z) + (m.vec[0].w * u.w);
    p.y = (m.vec[1].x * u.x) + (m.vec[1].y * u.y) + (m.vec[1].z * u.z) + (m.vec[1].w * u.w);
	p.z = (m.vec[2].x * u.x) + (m.vec[2].y * u.y) + (m.vec[2].z * u.z) + (m.vec[2].w * u.w);
    p.w = (m.vec[3].x * u.x) + (m.vec[3].y * u.y) + (m.vec[3].z * u.z) + (m.vec[3].w * u.w);
    return p;
}

/////////////////////////////////////////////////////////////////
//===============================================================
//
//	Quaternion
//
//===============================================================
/////////////////////////////////////////////////////////////////

quat_t quat_add  ( quat_t a, quat_t b ) { return quat_init ( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }
quat_t quat_sub  ( quat_t a, quat_t b ) { return quat_init ( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); }

quat_t quat_mul  ( quat_t a, quat_t b ) {
        return quat_init (
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
        a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

quat_t quat_adds        ( quat_t a, scal_t b ) { return quat_init ( a.x + b, a.y + b, a.z + b, a.w + b ); }
quat_t quat_subs        ( quat_t a, scal_t b ) { return quat_init ( a.x - b, a.y - b, a.z - b, a.w - b ); }
quat_t quat_muls        ( quat_t a, scal_t b ) { return quat_init ( a.x * b, a.y * b, a.z * b, a.w * b ); }
scal_t quat_dot         ( quat_t a, quat_t b ) { return ( a.x * b.x ) + ( a.y * b.y ) + ( a.z * b.z ) + ( a.w * b.w ); }
scal_t quat_lensqr      ( quat_t a ) { return (a.x * a.x) + (a.y * a.y) + (a.z * a.z) + (a.w * a.w); }
scal_t quat_invlength   ( quat_t a ) { return invsqrt ( quat_lensqr ( a ) ); }
scal_t quat_length      ( quat_t a ) { return sqrtf ( quat_lensqr(a) ); }
quat_t quat_normal      ( quat_t a ) { return quat_muls ( a, quat_invlength(a) ); }
quat_t quat_conjugate   ( quat_t a ) { return quat_init ( -a.x, -a.y, -a.z, a.w ); }
quat_t quat_inverse     ( quat_t a ) { return quat_normal ( quat_conjugate(a) ); }

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

mat2_t mat2_add ( mat2_t a, mat2_t b ) {
    return mat2_init(
        vec2_add(a.vec[0], b.vec[0]),
        vec2_add(a.vec[1], b.vec[1])
    );
}

mat2_t mat2_sub ( mat2_t a, mat2_t b ) {
    return mat2_init(
        vec2_sub(a.vec[0], b.vec[0]),
        vec2_sub(a.vec[1], b.vec[1])
    );
}

mat2_t mat2_mul( mat2_t a, mat2_t b ) {
    mat2_t p = mat2_identity();    
    p.mat[0][0] = a.mat[0][0] * b.mat[0][0] + a.mat[1][0] * b.mat[0][1];
    p.mat[0][1] = a.mat[0][1] * b.mat[0][0] + a.mat[1][1] * b.mat[0][1];
    p.mat[1][0] = a.mat[0][0] * b.mat[1][0] + a.mat[1][0] * b.mat[1][1];
    p.mat[1][1] = a.mat[0][1] * b.mat[1][0] + a.mat[1][1] * b.mat[1][1];
    return p;
}

mat2_t mat2_adds ( mat2_t a, scal_t b ) {
    return mat2_init(
        vec2_adds( a.vec[0], b ),
        vec2_adds( a.vec[1], b )
    );
}

mat2_t mat2_subs ( mat2_t a, scal_t b ) {
    return mat2_init(
        vec2_subs( a.vec[0], b ),
        vec2_subs( a.vec[1], b )
    );
}

mat2_t mat2_muls ( mat2_t a, scal_t b ) {
    return mat2_init(
        vec2_muls(a.vec[0], b),
        vec2_muls(a.vec[1], b)
    );
}

mat2_t mat2_transpose ( mat2_t a ) {
    mat2_t t = a; 
    t.elm[1] = a.elm[2];
    t.elm[2] = a.elm[1];
    return t;
}

mat2_t mat2_inverse( mat2_t m )
{
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

mat3_t mat3_add ( mat3_t a, mat3_t b ) {
    return mat3_init(
        vec3_add(a.vec[0], b.vec[0]),
        vec3_add(a.vec[1], b.vec[1]),
        vec3_add(a.vec[2], b.vec[2])
    );
}

mat3_t mat3_sub ( mat3_t a, mat3_t b ) {
    return mat3_init (
        vec3_sub(a.vec[0], b.vec[0]), 
        vec3_sub(a.vec[1], b.vec[1]), 
        vec3_sub(a.vec[2], b.vec[2])
    );
}

mat3_t mat3_mul ( mat3_t a, mat3_t b ) {
    mat3_t p = mat3_identity();
    p.mat[0][0] = a.mat[0][0] * b.mat[0][0] + a.mat[1][0] * b.mat[0][1] + a.mat[2][0] * b.mat[0][2];
    p.mat[0][1] = a.mat[0][1] * b.mat[0][0] + a.mat[1][1] * b.mat[0][1] + a.mat[2][1] * b.mat[0][2];
    p.mat[0][2] = a.mat[0][2] * b.mat[0][0] + a.mat[1][2] * b.mat[0][1] + a.mat[2][2] * b.mat[0][2];
    p.mat[1][0] = a.mat[0][0] * b.mat[1][0] + a.mat[1][0] * b.mat[1][1] + a.mat[2][0] * b.mat[1][2];
    p.mat[1][1] = a.mat[0][1] * b.mat[1][0] + a.mat[1][1] * b.mat[1][1] + a.mat[2][1] * b.mat[1][2];
    p.mat[1][2] = a.mat[0][2] * b.mat[1][0] + a.mat[1][2] * b.mat[1][1] + a.mat[2][2] * b.mat[1][2];
    p.mat[2][0] = a.mat[0][0] * b.mat[2][0] + a.mat[1][0] * b.mat[2][1] + a.mat[2][0] * b.mat[2][2];
    p.mat[2][1] = a.mat[0][1] * b.mat[2][0] + a.mat[1][1] * b.mat[2][1] + a.mat[2][1] * b.mat[2][2];
    p.mat[2][2] = a.mat[0][2] * b.mat[2][0] + a.mat[1][2] * b.mat[2][1] + a.mat[2][2] * b.mat[2][2];
    return p;    
}

mat3_t mat3_adds ( mat3_t a, scal_t b )
{
    return mat3_init(
        vec3_adds(a.vec[0], b), 
        vec3_adds(a.vec[1], b), 
        vec3_adds(a.vec[2], b)
    );
}

mat3_t mat3_subs ( mat3_t a, scal_t b ) {
    return mat3_init(
        vec3_subs(a.vec[0], b), 
        vec3_subs(a.vec[1], b), 
        vec3_subs(a.vec[2], b)
    );
}

mat3_t mat3_muls ( mat3_t a, scal_t b ) {
    return mat3_init(
        vec3_muls(a.vec[0], b), 
        vec3_muls(a.vec[1], b), 
        vec3_muls(a.vec[2], b)
    );
}

mat3_t mat3_transpose ( mat3_t a ) {
    mat3_t t = a;
    t.mat[0][1] = a.mat[1][0];
    t.mat[0][2] = a.mat[2][0];
    t.mat[1][0] = a.mat[0][1];
    t.mat[1][2] = a.mat[2][1];
    t.mat[2][0] = a.mat[0][2];
    t.mat[2][1] = a.mat[1][2];
    return t;
}

mat3_t mat3_inverse ( mat3_t m )
{
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

mat4_t mat4_add ( mat4_t a, mat4_t b ) {
    return mat4_init(
        vec4_add(a.vec[0], b.vec[0]), 
        vec4_add(a.vec[1], b.vec[1]), 
        vec4_add(a.vec[2], b.vec[2]), 
        vec4_add(a.vec[3], b.vec[3])
    );
}

mat4_t mat4_sub ( mat4_t a, mat4_t b ) {
    return mat4_init(
        vec4_sub(a.vec[0], b.vec[0]), 
        vec4_sub(a.vec[1], b.vec[1]), 
        vec4_sub(a.vec[2], b.vec[2]), 
        vec4_sub(a.vec[3], b.vec[3])
    );
}

mat4_t mat4_mul ( mat4_t a, mat4_t b ) {
    mat4_t p = mat4_identity();
    p.mat[0][0] = a.mat[0][0] * b.mat[0][0] + a.mat[1][0] * b.mat[0][1] + a.mat[2][0] * b.mat[0][2] + a.mat[3][0] * b.mat[0][3];
    p.mat[0][1] = a.mat[0][1] * b.mat[0][0] + a.mat[1][1] * b.mat[0][1] + a.mat[2][1] * b.mat[0][2] + a.mat[3][1] * b.mat[0][3];
    p.mat[0][2] = a.mat[0][2] * b.mat[0][0] + a.mat[1][2] * b.mat[0][1] + a.mat[2][2] * b.mat[0][2] + a.mat[3][2] * b.mat[0][3];
    p.mat[0][3] = a.mat[0][3] * b.mat[0][0] + a.mat[1][3] * b.mat[0][1] + a.mat[2][3] * b.mat[0][2] + a.mat[3][3] * b.mat[0][3];
    p.mat[1][0] = a.mat[0][0] * b.mat[1][0] + a.mat[1][0] * b.mat[1][1] + a.mat[2][0] * b.mat[1][2] + a.mat[3][0] * b.mat[1][3];
    p.mat[1][1] = a.mat[0][1] * b.mat[1][0] + a.mat[1][1] * b.mat[1][1] + a.mat[2][1] * b.mat[1][2] + a.mat[3][1] * b.mat[1][3];
    p.mat[1][2] = a.mat[0][2] * b.mat[1][0] + a.mat[1][2] * b.mat[1][1] + a.mat[2][2] * b.mat[1][2] + a.mat[3][2] * b.mat[1][3];
    p.mat[1][3] = a.mat[0][3] * b.mat[1][0] + a.mat[1][3] * b.mat[1][1] + a.mat[2][3] * b.mat[1][2] + a.mat[3][3] * b.mat[1][3];
    p.mat[2][0] = a.mat[0][0] * b.mat[2][0] + a.mat[1][0] * b.mat[2][1] + a.mat[2][0] * b.mat[2][2] + a.mat[3][0] * b.mat[2][3];
    p.mat[2][1] = a.mat[0][1] * b.mat[2][0] + a.mat[1][1] * b.mat[2][1] + a.mat[2][1] * b.mat[2][2] + a.mat[3][1] * b.mat[2][3];
    p.mat[2][2] = a.mat[0][2] * b.mat[2][0] + a.mat[1][2] * b.mat[2][1] + a.mat[2][2] * b.mat[2][2] + a.mat[3][2] * b.mat[2][3];
    p.mat[2][3] = a.mat[0][3] * b.mat[2][0] + a.mat[1][3] * b.mat[2][1] + a.mat[2][3] * b.mat[2][2] + a.mat[3][3] * b.mat[2][3];
    p.mat[3][0] = a.mat[0][0] * b.mat[3][0] + a.mat[1][0] * b.mat[3][1] + a.mat[2][0] * b.mat[3][2] + a.mat[3][0] * b.mat[3][3];
    p.mat[3][1] = a.mat[0][1] * b.mat[3][0] + a.mat[1][1] * b.mat[3][1] + a.mat[2][1] * b.mat[3][2] + a.mat[3][1] * b.mat[3][3];
    p.mat[3][2] = a.mat[0][2] * b.mat[3][0] + a.mat[1][2] * b.mat[3][1] + a.mat[2][2] * b.mat[3][2] + a.mat[3][2] * b.mat[3][3];
    p.mat[3][3] = a.mat[0][3] * b.mat[3][0] + a.mat[1][3] * b.mat[3][1] + a.mat[2][3] * b.mat[3][2] + a.mat[3][3] * b.mat[3][3];
    return p;
}

mat4_t mat4_adds ( mat4_t a, scal_t b ) {
    return mat4_init(
        vec4_adds(a.vec[0], b), 
        vec4_adds(a.vec[1], b), 
        vec4_adds(a.vec[2], b), 
        vec4_adds(a.vec[3], b)
    );
}

mat4_t mat4_subs ( mat4_t a, scal_t b ) {
    return mat4_init(
        vec4_subs(a.vec[0], b), 
        vec4_subs(a.vec[1], b), 
        vec4_subs(a.vec[2], b), 
        vec4_subs(a.vec[3], b)
    );
}

mat4_t mat4_muls ( mat4_t a, scal_t b ) {
    return mat4_init(
        vec4_muls(a.vec[0], b), 
        vec4_muls(a.vec[1], b), 
        vec4_muls(a.vec[2], b), 
        vec4_muls(a.vec[3], b)
    );
}

mat4_t mat4_transpose ( mat4_t a ) {
    mat4_t i = a;
    i.mat[0][1] = a.mat[1][0];
    i.mat[0][2] = a.mat[2][0];
    i.mat[0][3] = a.mat[3][0];
    i.mat[1][0] = a.mat[0][1];
    i.mat[1][2] = a.mat[2][1];
    i.mat[1][3] = a.mat[3][1];
    i.mat[2][0] = a.mat[0][2];
    i.mat[2][1] = a.mat[1][2];
    i.mat[2][3] = a.mat[3][2];
    i.mat[3][0] = a.mat[0][3];
    i.mat[3][1] = a.mat[1][3];
    i.mat[3][2] = a.mat[2][3];
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


mat4_t mat4_translate ( mat4_t a, vec3_t b )
{
    mat4_t c = mat4_identity();
    c.mat[0][3] = b.x;
    c.mat[1][3] = b.y;
    c.mat[2][3] = b.z;
    return mat4_mul (a, c);
}

mat4_t mat4_scale ( mat4_t a, vec3_t b )
{
    mat4_t c = mat4_identity();
    c.mat[0][0] = b.x;
    c.mat[1][1] = b.y;
    c.mat[2][2] = b.z;
    return mat4_mul ( a, c );
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


