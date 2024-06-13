/*************************************************
 * Astix math library - Test program.
 * 
 * Needs to a major rewite, but it works for now.
 * for now.. Plan to implement test functions for 
 * the perspective, ortho, lookat, rotate functs.
*************************************************/

#include <stdio.h>
#include <math.h>
#include "astix.h"



void print_vec2 ( vec2_t x ) {
    printf("%.4f, %.4f\n", x.x, x.y);
}

void print_vec3 ( vec3_t x ) {
    printf("%.4f, %.4f, %.4f\n", x.x, x.y, x.z);
}

void print_vec4 ( vec4_t x ) {
    printf("%.4f, %.4f, %.4f, %.4f\n", x.x, x.y, x.z, x.w);
}

void print_quat ( quat_t x ) {
    printf("%.4f, %.4f, %.4f, %.4f\n", x.x, x.y, x.z, x.w);
}

void print_mat2 ( mat2_t m ) {
    print_vec2(m.vec[0]);
    print_vec2(m.vec[1]);
}

void print_mat3 ( mat3_t m ) {
    print_vec3(m.vec[0]);
    print_vec3(m.vec[1]);
    print_vec3(m.vec[2]);
}

void print_mat4 ( mat4_t m ) {
    print_vec4(m.vec[0]);
    print_vec4(m.vec[1]);
    print_vec4(m.vec[2]);
    print_vec4(m.vec[3]);
}

void vec2_testfunctions(void)
{
    vec2_t v = {1.0f, 2.0f};
    vec2_t u = {2.0f, 1.0f};
    mat2_t m = mat2_init (
                0.0f, 1.0f,
                4.0f, 5.0f
            );
    printf("add  --------- "); print_vec2(vec2_add(v, u));
    printf("sub  --------- "); print_vec2(vec2_sub(v, u));
    printf("mul  --------- "); print_vec2(vec2_mul(v, u));
    printf("adds --------- "); print_vec2(vec2_adds(v, 4.0f));
    printf("subs --------- "); print_vec2(vec2_subs(v, 4.0f));
    printf("muls --------- "); print_vec2(vec2_muls(v, 4.0f));
    printf("dot ---------- "); printf("%.4f\n", vec2_dot(v, u));
    printf("length^2 ----- "); printf("%.4f\n", vec2_lensqr(v));
    printf("1/length ----- "); printf("%.4f\n", vec2_invlength(v));
    printf("length ------- "); printf("%.4f\n", vec2_length(v));
    printf("normal ------- "); print_vec2(vec2_normal(v));
    printf("mul mat2 ----- "); print_vec2(vec2_mul_mat2(v, m));
}

void vec3_testfunctions(void)
{
    vec3_t v = {1.0f, 2.0f, 3.0f};
    vec3_t u = {3.0f, 2.0f, 1.0f};
    mat3_t m = mat3_init (
        0.0f, 1.0f, 2.0f,
        4.0f, 5.0f, 6.0f,
        8.0f, 9.0f, 0.0f );

    mat4_t m4 = mat4_init (
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 0.0f, 1.0f,
        2.0f, 3.0f, 4.0f, 5.0f );

    print_vec3( vec3_mul_mat3 (v, m)  );
    printf("add  --------- "); print_vec3(vec3_add(v, u));
    printf("sub  --------- "); print_vec3(vec3_sub(v, u));
    printf("mul  --------- "); print_vec3(vec3_mul(v, u));
    printf("adds --------- "); print_vec3(vec3_adds(v, 4.0f));
    printf("subs --------- "); print_vec3(vec3_subs(v, 4.0f));
    printf("muls --------- "); print_vec3(vec3_muls(v, 4.0f));
    printf("dot ---------- "); printf("%.4f\n", vec3_dot(v, u));
    printf("length^2 ----- "); printf("%.4f\n", vec3_lensqr(v));
    printf("1/length ----- "); printf("%.4f\n", vec3_invlength(v));
    printf("length ------- "); printf("%.4f\n", vec3_length(v));
    printf("normal ------- "); print_vec3(vec3_normal(v));
    printf("mul mat3 ----- "); print_vec3(vec3_mul_mat3(v, m));
    printf("mul mat4 ----- "); print_vec3(vec3_mul_mat4(v, m4));
    
}

void vec4_testfunctions(void)
{
    vec4_t v = {1.0f, 2.0f, 3.0f, 4.0f};
    vec4_t u = {4.0f, 3.0f, 2.0f, 1.0f};
    mat4_t m = mat4_init (
                         0.0f, 1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f, 7.0f,
                         8.0f, 9.0f, 0.0f, 1.0f,
                         2.0f, 3.0f, 4.0f, 5.0f
                        );
    printf("add  --------- "); print_vec4(vec4_add(v, u));
    printf("sub  --------- "); print_vec4(vec4_sub(v, u));
    printf("mul  --------- "); print_vec4(vec4_mul(v, u));
    printf("adds --------- "); print_vec4(vec4_adds(v, 4.0f));
    printf("subs --------- "); print_vec4(vec4_subs(v, 4.0f));
    printf("muls --------- "); print_vec4(vec4_muls(v, 4.0f));
    printf("dot ---------- "); printf("%.4f\n", vec4_dot(v, u));
    printf("length^2 ----- "); printf("%.4f\n", vec4_lensqr(v));
    printf("1/length ----- "); printf("%.4f\n", vec4_invlength(v));
    printf("length ------- "); printf("%.4f\n", vec4_length(v));
    printf("normal ------- "); print_vec4(vec4_normal(v));
    printf("mul mat4 ----- "); print_vec4(vec4_mul_mat4(v, m));
}

void quat_testfunctions(void)
{
    quat_t v = {1.0f, 2.0f, 3.0f, 4.0f};
    quat_t u = {4.0f, 3.0f, 2.0f, 1.0f};
    scal_t s = 2.0f;

    printf("add  --------- "); print_quat(quat_add(v, u));
    printf("sub  --------- "); print_quat(quat_sub(v, u));
    printf("mul  --------- "); print_quat(quat_mul(v, u));
    printf("adds --------- "); print_quat(quat_adds(v, s));
    printf("subs --------- "); print_quat(quat_subs(v, s));
    printf("muls --------- "); print_quat(quat_muls(v, s));
    printf("dot ---------- "); printf("%.4f\n", quat_dot(v, u));
    printf("length^2 ----- "); printf("%.4f\n", quat_lensqr(v));
    printf("1/length ----- "); printf("%.4f\n", quat_invlength(v));
    printf("length ------- "); printf("%.4f\n", quat_length(v));
    printf("normal ------- "); print_quat(quat_normal(v));
    printf("conjugate ---- "); print_quat(quat_conjugate(v));
    printf("inverse ------ "); print_quat(quat_inverse(v));
}

void mat2_testfunctions(void)
{
    mat2_t m = {1,2,   3,4};
    mat2_t i = {1,2,   2,3};
    printf("\nid   ---------\n"); print_mat2(mat2_identity());     
    printf("\nadd  ---------\n"); print_mat2(mat2_add (m, m)); 
    printf("\nsub  ---------\n"); print_mat2(mat2_sub (m, m)); 
    printf("\nmul  ---------\n"); print_mat2(mat2_mul (m, m));
    printf("\nadds ---------\n"); print_mat2(mat2_adds(m, 4.0f));
    printf("\nsubs ---------\n"); print_mat2(mat2_subs(m, 4.0f));
    printf("\nmuls ---------\n"); print_mat2(mat2_muls(m, 4.0f));
    printf("\ntrans --------\n"); print_mat2(mat2_transpose(m));
    printf("\ninverse ------\n"); print_mat2(mat2_inverse(i));
}

void mat3_testfunctions(void)
{
    mat3_t m = {1,2,3,   4,5,6,   7,8,9};
    mat3_t i = {1,2,3,   0,1,4,   5,6,0};
    printf("\nid   ---------\n"); print_mat3(mat3_identity());     
    printf("\nadd  ---------\n"); print_mat3(mat3_add (m, m)); 
    printf("\nsub  ---------\n"); print_mat3(mat3_sub (m, m)); 
    printf("\nmul  ---------\n"); print_mat3(mat3_mul (m, m));
    printf("\nadds ---------\n"); print_mat3(mat3_adds(m, 4.0f));
    printf("\nsubs ---------\n"); print_mat3(mat3_subs(m, 4.0f));
    printf("\nmuls ---------\n"); print_mat3(mat3_muls(m, 4.0f));
    printf("\ntrans --------\n"); print_mat3(mat3_transpose(m));
    printf("\ninverse ------\n"); print_mat3(mat3_inverse(i));
    

}

void mat4_testfunctions(void)
{
    mat4_t m = {1,2,3,4,   5,6,7,8,   9,1,2,3,   4,5,6,7};
    mat4_t t = {5,6,6,8,   2,2,2,8,   6,6,2,8,   2,3,6,7};
    printf("\nid   ---------\n"); print_mat4(mat4_identity());     
    printf("\nadd  ---------\n"); print_mat4(mat4_add (m, m)); 
    printf("\nsub  ---------\n"); print_mat4(mat4_sub (m, m)); 
    printf("\nmul  ---------\n"); print_mat4(mat4_mul (m, m));
    printf("\nadds ---------\n"); print_mat4(mat4_adds(m, 4.0f));
    printf("\nsubs ---------\n"); print_mat4(mat4_subs(m, 4.0f));
    printf("\nmuls ---------\n"); print_mat4(mat4_muls(m, 4.0f));
    printf("\ntrans --------\n"); print_mat4(mat4_transpose(m));
    printf("\ninverse ------\n"); print_mat4(mat4_inverse(t));
}

int main()
{
    quat_testfunctions();
    return 0;
}