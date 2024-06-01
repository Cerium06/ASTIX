/*************************************************
 * Astix math library - Test program.
 * 
 * Needs to a major rewite, ut it works for now.
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

void v2_testfunctions(void)
{
    vec2_t v = {1.0f, 2.0f};
    vec2_t u = {2.0f, 1.0f};
    scal_t s = 2.0f;
    printf("\nadd  ---------\n"); print_vec2(vec2_add(v, u));
    printf("\nsub  ---------\n"); print_vec2(vec2_sub(v, u));
    printf("\nmul  ---------\n"); print_vec2(vec2_mul(v, u));
    printf("\nadds ---------\n"); print_vec2(vec2_adds(v, s));
    printf("\nsubs ---------\n"); print_vec2(vec2_subs(v, s));
    printf("\nmuls ---------\n"); print_vec2(vec2_muls(v, s));
    printf("\ndot ----------\n"); printf("%.4f\n", vec2_dot(v, u));
    printf("\nlength^2 -----\n"); printf("%.4f\n", vec2_lensqr(v));
    printf("\nlength -------\n"); printf("%.4f\n", vec2_length(v));
    printf("\nnormal -------\n"); print_vec2(vec2_normal(v));
}

void v3_testfunctions(void)
{
    vec3_t v = {1.0f, 2.0f, 3.0f};
    vec3_t u = {3.0f, 2.0f, 1.0f};
    scal_t s = 2.0f;
    printf("\nadd  ---------\n"); print_vec3(vec3_add(v, u));
    printf("\nsub  ---------\n"); print_vec3(vec3_sub(v, u));
    printf("\nmul  ---------\n"); print_vec3(vec3_mul(v, u));
    printf("\nadds ---------\n"); print_vec3(vec3_adds(v, s));
    printf("\nsubs ---------\n"); print_vec3(vec3_subs(v, s));
    printf("\nmuls ---------\n"); print_vec3(vec3_muls(v, s));
    printf("\ndot ----------\n"); printf("%.4f\n", vec3_dot(v, u));
    printf("\nlength^2 -----\n"); printf("%.4f\n", vec3_lensqr(v));
    printf("\nlength -------\n"); printf("%.4f\n", vec3_length(v));
    printf("\nnormal -------\n"); print_vec3(vec3_normal(v));
}

void v4_testfunctions(void)
{
    vec4_t v = {1.0f, 2.0f, 3.0f, 4.0f};
    vec4_t u = {4.0f, 3.0f, 2.0f, 1.0f};
    scal_t s = 2.0f;

    printf("\nadd  ---------\n"); print_vec4(vec4_add(v, u));
    printf("\nsub  ---------\n"); print_vec4(vec4_sub(v, u));
    printf("\nmul  ---------\n"); print_vec4(vec4_mul(v, u));
    printf("\nadds ---------\n"); print_vec4(vec4_adds(v, s));
    printf("\nsubs ---------\n"); print_vec4(vec4_subs(v, s));
    printf("\nmuls ---------\n"); print_vec4(vec4_muls(v, s));
    printf("\ndot ----------\n"); printf("%.4f\n", vec4_dot(v, u));
    printf("\nlength^2 -----\n"); printf("%.4f\n", vec4_lensqr(v));
    printf("\nlength -------\n"); printf("%.4f\n", vec4_length(v));
    printf("\nnormal -------\n"); print_vec4(vec4_normal(v));
}

void quat_testfunctions(void)
{
    quat_t v = {1.0f, 2.0f, 3.0f, 4.0f};
    quat_t u = {4.0f, 3.0f, 2.0f, 1.0f};
    scal_t s = 2.0f;

    printf("\nadd  ---------\n"); print_quat(quat_add(v, u));
    printf("\nsub  ---------\n"); print_quat(quat_sub(v, u));
    printf("\nmul  ---------\n"); print_quat(quat_mul(v, u));
    printf("\nadds ---------\n"); print_quat(quat_adds(v, s));
    printf("\nsubs ---------\n"); print_quat(quat_subs(v, s));
    printf("\nmuls ---------\n"); print_quat(quat_muls(v, s));
    printf("\ndot ----------\n"); printf("%.4f\n", quat_dot(v, u));
    printf("\nlength^2 -----\n"); printf("%.4f\n", quat_lensqr(v));
    printf("\nlength -------\n"); printf("%.4f\n", quat_length(v));
    printf("\nnormal -------\n"); print_quat(quat_normal(v));
    printf("\ninverse ------\n"); print_quat(quat_inverse(v));
}

void mat2_testfunctions(void)
{
    mat2_t m = {1,2,   3,4};
    mat2_t i = {1,2,   2,3};
    printf("\nid   ---------\n"); print_mat2(mat2_identity());     
    printf("\nadd  ---------\n"); print_mat2(mat2_add (m, m)); 
    printf("\nsub  ---------\n"); print_mat2(mat2_sub (m, m)); 
    printf("\nmul  ---------\n"); print_mat2(mat2_mul (m, m));
    printf("\nadds ---------\n"); print_mat2(mat2_adds(m, 4));
    printf("\nsubs ---------\n"); print_mat2(mat2_subs(m, 4));
    printf("\nmuls ---------\n"); print_mat2(mat2_muls(m, 4));
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
    printf("\nadds ---------\n"); print_mat3(mat3_adds(m, 4));
    printf("\nsubs ---------\n"); print_mat3(mat3_subs(m, 4));
    printf("\nmuls ---------\n"); print_mat3(mat3_muls(m, 4));
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
    printf("\nadds ---------\n"); print_mat4(mat4_adds(m, 4));
    printf("\nsubs ---------\n"); print_mat4(mat4_subs(m, 4));
    printf("\nmuls ---------\n"); print_mat4(mat4_muls(m, 4));
    printf("\ntrans --------\n"); print_mat4(mat4_transpose(m));
    printf("\ninverse ------\n"); print_mat4(mat4_inverse(t));
}

int main()
{
    //mat2_testfunctions();
    v4_testfunctions();



    return 0;
}