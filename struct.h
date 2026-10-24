#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PI 3.141592654

#define EULER_STEP_RAD 2.09439510239
#define CONF_STEP_RAD 2.09439510239
#define EULER_STEP_CNT 3
#define CONF_STEP_CNT 3
#define FRAG_CNT 12 
#define M 12
#define __CONFORMATION_CANDIDATES_NUM__ CONF_STEP_CNT

#define H_VAN_ATOM_IDX  0
#define C_VAN_ATOM_IDX  1
#define N_VAN_ATOM_IDX  2
#define O_VAN_ATOM_IDX  3
#define F_VAN_ATOM_IDX  4
#define P_VAN_ATOM_IDX  5
#define S_VAN_ATOM_IDX  6
#define CL_VAN_ATOM_IDX 7

#define VANDER_WAALS_N 8

typedef struct RESIDUE
{
    int residueNum;
    char residueName[4] = { 0 };

    int atomNum[68];
    char atomName[68][5] = { 0 };
    int van_idx[68] = { 0 };

    int atoms = 0;
    float atomX[68];
    float atomY[68];
    float atomZ[68];

    char link[2];
    float occupyRatio[68];
    float temperature[68];
}RESIDUE;

typedef struct ATOM
{
    char num;
    char name[4] = { 0 };
    char van_table_idx;
    char residue_name[4] = {0};
    float X;
    float Y;
    float Z;
}ATOM;

typedef struct FRAGMENT
{
    ATOM atom[10];
    char atom_cnt;
    int index;
    unsigned int pose_cnt;
}FRAGMENT;
 
typedef struct GROUP
{
    char group_idx;
    char frag_cnt;
    char frag_idx[4];
    unsigned int* pcombidx;
    unsigned int comb_cnt;
}GROUP;

typedef struct BOND
{
    int frag_idx[2];
    int atom_idx[2];
}BOND;