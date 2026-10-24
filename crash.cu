#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include <math.h>
#include <time.h>
#include "struct.h"



/*----------------------------------------------------------The data structure------------------------------------------------------------------------*/
#if __CONFORMATION_CANDIDATES_NUM__ == 2
           unsigned int ConfStepCnt[12]    = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };  //CONF_STEP_CNT = 2
__device__ unsigned int ConfStepCntDev[12] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };  //CONF_STEP_CNT = 2
#elif __CONFORMATION_CANDIDATES_NUM__ == 3
		   unsigned int ConfStepCnt   [12] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147 };  //CONF_STEP_CNT = 3
__device__ unsigned int ConfStepCntDev[12] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147 };  //CONF_STEP_CNT = 3
#elif __CONFORMATION_CANDIDATES_NUM__ == 4
           unsigned int ConfStepCnt   [12] = { 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576 ,4194304};  //CONF_STEP_CNT = 4
__device__ unsigned int ConfStepCntDev[12] = { 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576 ,4194304};  //CONF_STEP_CNT = 4
#elif __CONFORMATION_CANDIDATES_NUM__ == 5
           unsigned int ConfStepCnt   [12]   = { 1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 48828125 };  //CONF_STEP_CNT = 5
__device__ unsigned int ConfStepCntDev[12]   = { 1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 9765625, 48828125 };  //CONF_STEP_CNT = 5
#endif

//0:H; 1:C; 2:N; 3:O; 4:F; 5:P; 6:S; 7:Cl;
float VanderWaalsHost[8][8] =
{
	// H      C     N    O      F     P     S     Cl
	/*H*/ {2.4f , 2.90f, 2.75f, 2.72f, 2.67f, 3.00f, 3.00f, 2.95f},
	/*C*/ {2.90f, 3.40f, 3.25f, 3.22f, 3.17f, 3.07f, 3.58f, 3.54f},
	/*N*/ {2.75f, 3.25f, 3.10f, 3.07f, 3.02f, 3.35f, 3.35f, 3.30f},
	/*O*/ {2.72f, 3.22f, 3.07f, 3.04f, 2.99f, 3.32f, 3.32f, 3.27f},
	/*F*/ {2.67f, 3.17f, 3.02f, 2.99f, 2.94f, 3.27f, 3.27f, 3.22f},
	/*P*/ {3.00f, 3.50f, 3.35f, 3.32f, 3.27f, 3.60f, 3.60f, 3.55f},
	/*S*/ {3.00f, 3.50f, 3.35f, 3.32f, 3.27f, 3.60f, 3.60f, 3.55f},
	/*Cl*/{2.95f, 3.54f, 3.30f, 3.27f, 3.22f, 3.55f, 3.55f, 3.50f}
};
__device__ float VanderWaalsDev[8][8] =
{
	// H      C     N    O      F     P     S     Cl
	/*H*/ {2.4f , 2.90, 2.75, 2.72, 2.67, 3.00, 3.00, 2.95},
	/*C*/ {2.90f, 3.40, 3.25, 3.22, 3.17, 3.07, 3.58, 3.54},
	/*N*/ {2.75f, 3.25, 3.10, 3.07, 3.02, 3.35, 3.35, 3.30},
	/*O*/ {2.72f, 3.22, 3.07, 3.04, 2.99, 3.32, 3.32, 3.27},
	/*F*/ {2.67f, 3.17, 3.02, 2.99, 2.94, 3.27, 3.27, 3.22},
	/*P*/ {3.00f, 3.50, 3.35, 3.32, 3.27, 3.60, 3.60, 3.55},
	/*S*/ {3.00f, 3.50, 3.35, 3.32, 3.27, 3.60, 3.60, 3.55},
	/*Cl*/{2.95f, 3.54, 3.30, 3.27, 3.22, 3.55, 3.55, 3.50}
};
__device__ int CrashTIdxMapDev[3][3] =
{ {-1, 0, 1},
  { 0,-1, 2},
  { 1, 2,-1},
};
int CrashTIdxMapHost[3][3] =
{ {-1, 0, 1},
  { 0,-1, 2},
  { 1, 2,-1},
};
/*--------------------------------------------------------------------------------------------------------------------------------------------------*/


/*---------------------------Identify a pair of atoms is on the bond-------------------------*/
//Function:Check if a pair of atoms is on the same bond or not
//Argument:
//  [in ]1.const BOND* bond:bond data
//  [in ]2.int bond_cnt: the number of global bonds
//  [in ]3.int frag1_idx:the index of fragment1 which atom1 is on
//  [in ]4.int frag2_idx:the index of fragment2 which atom2 is on
//  [in ]3.int atom1_idx:the index of atom1
//  [in ]4.int atom2_idx:the index of atom2
//Return value: 1--the pair of atoms is on the bond; 0--the pair of atoms isn't on the bond
__device__ inline bool IsOnBondDev(const BOND* bond, int bond_cnt, int frag1_idx, int frag2_idx, int atom1_idx, int atom2_idx)
{
	for (int i = 0; i < bond_cnt; i++)
	{
		if ((bond[i].frag_idx[0] == frag1_idx && bond[i].frag_idx[1] == frag2_idx && bond[i].atom_idx[0] == atom1_idx && bond[i].atom_idx[1] == atom2_idx)
		 || (bond[i].frag_idx[0] == frag2_idx && bond[i].frag_idx[1] == frag1_idx && bond[i].atom_idx[0] == atom2_idx && bond[i].atom_idx[1] == atom1_idx))

		{
			return true;
		}
	}			
	return false;
}
/*------------------------------------------------------------------------------------------------*/


/*----------------------------------extract legal groups ----------------------------------*/
//Function: Extract legal groups from group search space
//Argument: 
//  [out]1.unsigned int* pcombidx: table recording indexes of groups  
//  [in ]2.unsigned char* pconflictT: lookup table recording conflict flags of an intra-group
//  [in ]3.unsigned long long conflictT_byte_cnt: the size in bytes of the lookup table
//Reurn value: return the number of legal groups
unsigned int GetUnconflictedGroups(unsigned int* pcombidx, unsigned char* pconflictT, unsigned long long conflictT_byte_cnt)
{
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < conflictT_byte_cnt; i++)
	{
		if (0 == pconflictT[i])
		{
			pcombidx[cnt++] = i;
		}
	}
	return cnt;
}
/*-------------------------------------------------------------------------------------------*/


/*--------------------------------inra-group legality check------------------------------------*/
//Function: Check if some combinations of group0 conflict or not and fill the look up table
//Argument: 
//  [in ]1.FRAGMENT** pfragsr: the source fragments data 
//  [in ]2.BOND* bond: the bonds data
//  [in ]3.unsigned long long frag_poses_cnt: the number of combinations of different fragments poses in group0
//  [out]4.unsigned  char* pcrashTDev: look up table 0--unconflicted; 1--conflicted
//  [in ]5.double threshold_factor:conflict threshold coefficent
//Reurn value: 
__global__ void GPUCheckGroup0 (FRAGMENT** pfragsrc, BOND* bond, unsigned long long frag_poses_cnt, unsigned  char* pcrashTDev, double threshold_factor)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < frag_poses_cnt)
	{
		unsigned int pose[4];
		pose[0] = idx / ConfStepCntDev[3];//片段0姿态
		pose[1] = idx / ConfStepCntDev[2];//片段1姿态
		pose[2] = idx / CONF_STEP_CNT;//片段2姿态
		pose[3] = idx;//片段3姿态

		for (int i = 0; i < 3; i++)
		{
			for (int j = i + 1; j < 4; j++)
			{
				for (int m = 0; m < pfragsrc[i][0].atom_cnt; m++)
				{
					for (int n = 0; n < pfragsrc[j][0].atom_cnt; n++)
					{
						if (IsOnBondDev(bond, 11, i, j, m, n)) continue;
						double dx = pfragsrc[i][pose[i]].atom[m].X - pfragsrc[j][pose[j]].atom[n].X;
						double dy = pfragsrc[i][pose[i]].atom[m].Y - pfragsrc[j][pose[j]].atom[n].Y;
						double dz = pfragsrc[i][pose[i]].atom[m].Z - pfragsrc[j][pose[j]].atom[n].Z;

						int vanidx1 = pfragsrc[i][0].atom[m].van_table_idx;
						int vanidx2 = pfragsrc[j][0].atom[n].van_table_idx;
						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							pcrashTDev[idx] = 1;
							return;
						}
					}
				}
			}
		}
	}
}
//Function: Check if some combinations of group1 conflict or not and fill the look up table
//Argument: 
//  [in ]1.FRAGMENT** pfragsr: the source fragments data 
//  [in ]2.BOND* bond: the bonds data
//  [in ]3.unsigned long long frag_poses_cnt: the number of combinations of different fragments poses in group1
//  [out]4.look up table: 0--unconflicted; 1--conflicted
//  [in ]5.double threshold_factor:conflict threshold coefficent
//Reurn value: 
__global__ void GPUCheckGroup1(FRAGMENT** pfragsrc, BOND* bond, unsigned long long frag_poses_cnt, unsigned  char* pcrashTDev, double threshold_factor)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < frag_poses_cnt)
	{
		unsigned int pose[4];
		pose[0] = idx / ConfStepCntDev[3];//片段4姿态
		pose[1] = idx / ConfStepCntDev[2];//片段5姿态
		pose[2] = idx / CONF_STEP_CNT;//片段6姿态
		pose[3] = idx;//片段7姿态

		for (int i = 4; i < 7; i++)
		{
			for (int j = i + 1; j < 8; j++)
			{
				for (int m = 0; m < pfragsrc[i][0].atom_cnt; m++)
				{
					for (int n = 0; n < pfragsrc[j][0].atom_cnt; n++)
					{
						if (IsOnBondDev(bond, 11, i, j, m, n)) continue;
						double dx = pfragsrc[i][pose[i-4]].atom[m].X - pfragsrc[j][pose[j-4]].atom[n].X;
						double dy = pfragsrc[i][pose[i-4]].atom[m].Y - pfragsrc[j][pose[j-4]].atom[n].Y;
						double dz = pfragsrc[i][pose[i-4]].atom[m].Z - pfragsrc[j][pose[j-4]].atom[n].Z;

						int vanidx1 = pfragsrc[i][0].atom[m].van_table_idx;
						int vanidx2 = pfragsrc[j][0].atom[n].van_table_idx;
						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							pcrashTDev[idx] = 1;
							return;
						}
					}
				}
			}
		}
	}
}
//Function: Check if some combinations of group2 conflict or not and fill the look up table
//Argument: 
//  [in ]1.FRAGMENT** pfragsr: the source fragments data 
//  [in ]2.BOND* bond: the bonds data
//  [in ]3.unsigned long long frag_poses_cnt: the number of combinations of different fragments poses in group2
//  [out]4.look up table: 0--unconflicted; 1--conflicted
//  [in ]5.double threshold_factor:conflict threshold coefficent
//Reurn value: 
__global__ void GPUCheckGroup2(FRAGMENT** pfragsrc, BOND* bond, unsigned long long frag_poses_cnt, unsigned  char* pcrashTDev, double threshold_factor)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < frag_poses_cnt)
	{
		unsigned int pose[4];
		pose[0] = idx / ConfStepCntDev[3];//片段4姿态
		pose[1] = idx / ConfStepCntDev[2];//片段5姿态
		pose[2] = idx / CONF_STEP_CNT;//片段6姿态
		pose[3] = idx;//片段7姿态

		for (int i = 8; i < 11; i++)
		{
			for (int j = i + 1; j < 12; j++)
			{
				for (int m = 0; m < pfragsrc[i][0].atom_cnt; m++)
				{
					for (int n = 0; n < pfragsrc[j][0].atom_cnt; n++)
					{
						if (IsOnBondDev(bond, 11, i, j, m, n)) continue;
						double dx = pfragsrc[i][pose[i-8]].atom[m].X - pfragsrc[j][pose[j-8]].atom[n].X;
						double dy = pfragsrc[i][pose[i-8]].atom[m].Y - pfragsrc[j][pose[j-8]].atom[n].Y;
						double dz = pfragsrc[i][pose[i-8]].atom[m].Z - pfragsrc[j][pose[j-8]].atom[n].Z;

						int vanidx1 = pfragsrc[i][0].atom[m].van_table_idx;
						int vanidx2 = pfragsrc[j][0].atom[n].van_table_idx;
						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							pcrashTDev[idx] = 1;
							return;
						}
					}
				}
			}
		}
	}
}
/*-------------------------------------------------------------------------------------------*/


/*---------------------------------extract legal groups -------------------------------------*/
//Function: Check if some combinations of group0 conflict or not and fill the look up table
//Argument: 
//  [out]1.GROUP& group: the legal groups 
//  [in ]2.FRAGMENT** pfragsrc:the source fragments data 
//  [in ]3.unsigned int* frag_poses_cnt: the number of combinations of different fragments poses in group0
//  [out]4.unsigned int* pcombidx: table recording indexes of legal groups 
//  [in ]5.int frag_cnt:the number of fragments
//  [in ]6.int group_idx:current group index
//  [in ]7.BOND* bond_dev: the bonds data
//  [in ]8.double threshold_factor:conflict threshold coefficent
//Reurn value: the number of unconflicted groups
unsigned int  LoadGroup(GROUP& group, FRAGMENT** pfragsrc, unsigned int* frag_poses_cnt, unsigned int* pcombidx, int frag_cnt, int group_idx ,BOND* bond_dev, double threshold_factor)
{
	unsigned int unconflicted_comb_cnt = 0;
	unsigned char* pconflictTDev;
	unsigned char* pconflictTHost;
	switch (group_idx)
	{
	case 0:
		pconflictTHost = (unsigned char*)malloc(frag_poses_cnt[3]);
		cudaMalloc(&pconflictTDev, frag_poses_cnt[3]);
		cudaMemset(pconflictTDev, 0, frag_poses_cnt[3]);
		GPUCheckGroup0 << <(frag_poses_cnt[3] + 511) >> 9, 512 >> > (pfragsrc, bond_dev,  frag_poses_cnt[3], pconflictTDev, threshold_factor);
		cudaMemcpy(pconflictTHost, pconflictTDev, frag_poses_cnt[3], cudaMemcpyDeviceToHost);
		unconflicted_comb_cnt = GetUnconflictedGroups(pcombidx, pconflictTHost, frag_poses_cnt[3]);
		group.frag_idx[0] = 0;
		group.frag_idx[1] = 1;
		group.frag_idx[2] = 2;
		group.frag_idx[3] = 3;
		break;
	case 1:
		pconflictTHost = (unsigned char*)malloc(frag_poses_cnt[7]);
		cudaMalloc(&pconflictTDev, frag_poses_cnt[7]);
		cudaMemset(pconflictTDev, 0, frag_poses_cnt[7]);
		GPUCheckGroup1 << <(frag_poses_cnt[7] + 511) >> 9, 512 >> > (pfragsrc, bond_dev,  frag_poses_cnt[7], pconflictTDev, threshold_factor);
		cudaMemcpy(pconflictTHost, pconflictTDev, frag_poses_cnt[7], cudaMemcpyDeviceToHost);
		unconflicted_comb_cnt = GetUnconflictedGroups(pcombidx, pconflictTHost, frag_poses_cnt[7]);
		group.frag_idx[0] = 4;
		group.frag_idx[1] = 5;
		group.frag_idx[2] = 6;
		group.frag_idx[3] = 7;
		break;
	case 2:
		pconflictTHost = (unsigned char*)malloc(frag_poses_cnt[11]);
		cudaMalloc(&pconflictTDev, frag_poses_cnt[11]);
		cudaMemset(pconflictTDev, 0, frag_poses_cnt[11]);
		GPUCheckGroup2 << < (frag_poses_cnt[11] + 511) >> 9, 512 >> > (pfragsrc, bond_dev, frag_poses_cnt[11], pconflictTDev, threshold_factor);
		cudaMemcpy(pconflictTHost, pconflictTDev, frag_poses_cnt[11], cudaMemcpyDeviceToHost);
		unconflicted_comb_cnt = GetUnconflictedGroups(pcombidx, pconflictTHost, frag_poses_cnt[11]);
		group.frag_idx[0] = 8;
		group.frag_idx[1] = 9;
		group.frag_idx[2] = 10;
		group.frag_idx[3] = 11;
		break;
	}
	cudaFree(pconflictTDev);
	free(pconflictTHost);

	group.frag_cnt = frag_cnt;
	group.group_idx = group_idx;
	group.pcombidx = pcombidx;
	group.comb_cnt = unconflicted_comb_cnt;

	return unconflicted_comb_cnt;
}
/*--------------------------------------------------------------------------------------------*/


/*----------------------------group & pocket legality check---------------------------------*/
//Function: Check if some combinations of groups & the pocket atoms conflict or not and fill the look up table
//Argument: 
//  [out]1.unsigned char* pconflictTDev: lookup table recording conflict flags of an group & pocket atoms
//  [in ]2.ATOM* pocket_atom:data of pocket atoms
//  [in ]3.FRAGMENT** psrc_frag:the source fragments data 
//  [in ]4.GROUP& group: the legal groups 
//  [in ]5.unsigned int* pcombidx: table recording indexes of legal groups 
//  [in ]5.int pocket_atom_cnt: number of pocket atoms 
//  [in ]7.double threshold_factor:conflict threshold coefficent
//Reurn value: 
__global__ void CheckGroup0_Pocket(unsigned char* pconflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int * pcombidx, int  pocket_atom_cnt, float threshold_factor)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < group->comb_cnt )
	{
		long  long pose_idx[4];
		pose_idx[0] = pcombidx[idx] / ConfStepCntDev[3];
		pose_idx[1] = pcombidx[idx] / ConfStepCntDev[2];
		pose_idx[2] = pcombidx[idx] / CONF_STEP_CNT;
		pose_idx[3] = pcombidx[idx];
		for (int i = 0; i < 4; i++)
		{
			for (int k = 0; k < psrc_frag[i]->atom_cnt; k++)
			{
				for (int m = 0; m < pocket_atom_cnt; m++)
				{
					float dx = psrc_frag[i][pose_idx[i]].atom[k].X - pocket_atom[m].X;
					float dy = psrc_frag[i][pose_idx[i]].atom[k].Y - pocket_atom[m].Y;
					float dz = psrc_frag[i][pose_idx[i]].atom[k].Z - pocket_atom[m].Z;
					int vanidx1 = psrc_frag[i][0].atom[k].van_table_idx;
					int vanidx2 = pocket_atom[m].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						return;
					}
				}
			}
		}
		pconflictTDev[idx] = 0;
	}
}
__global__ void CheckGroup1_Pocket(unsigned char* pconflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int * pcombidx, int  pocket_atom_cnt, float threshold_factor)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < group->comb_cnt )
	{
		unsigned long  long pose_idx[4];
		pose_idx[0] = pcombidx[idx] / ConfStepCntDev[3];
		pose_idx[1] = pcombidx[idx] / ConfStepCntDev[2];
		pose_idx[2] = pcombidx[idx] / CONF_STEP_CNT;
		pose_idx[3] = pcombidx[idx];
		for (int i = 4; i < 8; i++)
		{
			for (int k = 0; k < psrc_frag[i]->atom_cnt; k++)
			{
				for (int m = 0; m < pocket_atom_cnt; m++)
				{
					float dx = psrc_frag[i][pose_idx[i-4]].atom[k].X - pocket_atom[m].X;
					float dy = psrc_frag[i][pose_idx[i-4]].atom[k].Y - pocket_atom[m].Y;
					float dz = psrc_frag[i][pose_idx[i-4]].atom[k].Z - pocket_atom[m].Z;
					int vanidx1 = psrc_frag[i][0].atom[k].van_table_idx;
					int vanidx2 = pocket_atom[m].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{
						return;
					}
				}
			}
		}
		pconflictTDev[idx] = 0;
	}
}
__global__ void CheckGroup2_Pocket(unsigned char* pconflictTDev, ATOM* pocket_atom, FRAGMENT** psrc_frag, GROUP* group, unsigned int * pcombidx, int  pocket_atom_cnt, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < (unsigned long long)group->comb_cnt)
	{
		unsigned long long  pose[4];
		
		pose[0] = pcombidx[idx] / ConfStepCntDev[3];// 片段8的姿态
		pose[1] = pcombidx[idx] / ConfStepCntDev[2]; // 片段9的姿态
		pose[2] = pcombidx[idx] / CONF_STEP_CNT; // 片段10的姿态
		pose[3] = pcombidx[idx]; // 片段11的姿态

		for (int i = 8; i < 12; i++)
		{
			for (int k = 0; k < psrc_frag[i][0].atom_cnt; k++)
			{
				for (int m = 0; m < pocket_atom_cnt; m++)
				{
					float dx     = psrc_frag[i][pose[i-8]].atom[k].X - pocket_atom[m].X;
					float dy     = psrc_frag[i][pose[i-8]].atom[k].Y - pocket_atom[m].Y;
					float dz     = psrc_frag[i][pose[i-8]].atom[k].Z - pocket_atom[m].Z;
					int vanidx1  = psrc_frag[i][0].atom[k].van_table_idx;
					int vanidx2  = pocket_atom[m].van_table_idx;
					if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
					{		
						return;
					}
				}
			}
		}
		pconflictTDev[idx] = 0;
	}
}
/*-------------------------------------------------------------------------------------------*/


/*---------------------compress the conflict table to a smaller size-------------------------*/
//Function: compress the conflict table
//Argument:
//  [in ]1.unsigned char* pconflictTDev:source conflict table
//  [out]2.unsigned char* pcrashCompressedTDev:compressed conflict table
//  [in ]3.unsigned long long bytes:the number of bytes of the source conflict table
//Return value:
__global__ void CompressCrashT(unsigned char* pconflictTDev, unsigned char* pcrashCompressedTDev, unsigned long long bytes)
{
	unsigned int idx = (unsigned int)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < bytes && ((idx & 7)== 0))
	{
		pcrashCompressedTDev[idx >> 3] |= (pconflictTDev[idx] | (pconflictTDev[idx + 1]<<1) | (pconflictTDev[idx + 2]<<2) | (pconflictTDev[idx + 3]<<3) | (pconflictTDev[idx+4]<<4) | (pconflictTDev[idx + 5]<<5) | (pconflictTDev[idx + 6]<<6) | (pconflictTDev[idx + 7]<<7));
	}
}
/*-------------------------------------------------------------------------------------------*/


/*------------------------------inter-group legality check-----------------------------------*/
//Function: Check if some combinations of group & groups conflict or not and fill the look up table
//Argument: 
//   [out]1.unsigned char* pconflictTDev: lookup table recording conflict flags of an group & group
//   [in ]2.FRAGMENT** psrc_frag:the source fragments data 
//   [in ]3.BOND* bond_dev: the bonds data
//   [in ]4.unsigned int conflictTlen: number of bytes of the conflict table
//   [in ]5.unsigned int* pcomidx_dev: table recording indexes of legal groups 
//   [in ]6.double threshold_factor:conflict threshold coefficent
//Reurn value: 
__global__ void CheckGroups_01(unsigned char* pconflictTDev, FRAGMENT** psrc_frag, BOND* bond, unsigned int conflictTlen, unsigned int* pcomidx_dev, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int pose[8];
	if (idx < (unsigned long long) conflictTlen)
	{
		pose[0] = pcomidx_dev[idx] / ConfStepCntDev[7];
		pose[1] = pcomidx_dev[idx] / ConfStepCntDev[6];
		pose[2] = pcomidx_dev[idx] / ConfStepCntDev[5];
		pose[3] = pcomidx_dev[idx] / ConfStepCntDev[4];
		pose[4] = pcomidx_dev[idx] / ConfStepCntDev[3];
		pose[5] = pcomidx_dev[idx] / ConfStepCntDev[2];
		pose[6] = pcomidx_dev[idx] / ConfStepCntDev[1];
		pose[7] = pcomidx_dev[idx];

		for (int m = 0; m < 4; m++)
		{
			for (int n = 4; n < 8; n++)
			{
				for (int i = 0; i < psrc_frag[m][0].atom_cnt; i++)
				{
					for (int j = 0; j < psrc_frag[n][0].atom_cnt; j++)
					{
						if (IsOnBondDev(bond, 11, m, n, i, j))	continue;
						float dx = psrc_frag[m][pose[m]].atom[i].X - psrc_frag[n][pose[n]].atom[j].X;
						float dy = psrc_frag[m][pose[m]].atom[i].Y - psrc_frag[n][pose[n]].atom[j].Y;
						float dz = psrc_frag[m][pose[m]].atom[i].Z - psrc_frag[n][pose[n]].atom[j].Z;
						int vanidx1 = psrc_frag[m][0].atom[i].van_table_idx;
						int vanidx2 = psrc_frag[n][0].atom[j].van_table_idx;
						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							return;
						}
					}
				}
			}
		}
		pconflictTDev[idx] = 0;
	}
}
__global__ void CheckGroups_02(unsigned char* pconflictTDev, FRAGMENT** psrc_frag, BOND* bond, unsigned int conflictTlen, unsigned int* pcomidx_dev,float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < (unsigned long long)conflictTlen)
	{
		unsigned long long pose[8];

		pose[0] = pcomidx_dev[idx] / ConfStepCntDev[11]; //片段0
		pose[1] = pcomidx_dev[idx] / ConfStepCntDev[10]; //片段1
		pose[2] = pcomidx_dev[idx] / ConfStepCntDev[9];//片段2
		pose[3] = pcomidx_dev[idx] / ConfStepCntDev[8];//片段3
		pose[4] = pcomidx_dev[idx] / ConfStepCntDev[3];//片段8
		pose[5] = pcomidx_dev[idx] / ConfStepCntDev[2];//片段9
		pose[6] = pcomidx_dev[idx] / ConfStepCntDev[1];//片段10
		pose[7] = pcomidx_dev[idx];//片段11

		for (int m = 0; m < 4; m++)
		{
			for (int n = 8; n < 12; n++)
			{
				for (int i = 0; i < psrc_frag[m]->atom_cnt; i++)
				{
					for (int j = 0; j < psrc_frag[n]->atom_cnt; j++)
					{
						if (IsOnBondDev(bond, 11, m, n, i, j))	continue;

						float dx = psrc_frag[m][pose[m]].atom[i].X - psrc_frag[n][pose[n-4]].atom[j].X;
						float dy = psrc_frag[m][pose[m]].atom[i].Y - psrc_frag[n][pose[n-4]].atom[j].Y;
						float dz = psrc_frag[m][pose[m]].atom[i].Z - psrc_frag[n][pose[n-4]].atom[j].Z;

						int vanidx1 = psrc_frag[m][pose[m]].atom[i].van_table_idx;
						int vanidx2 = psrc_frag[n][pose[n-4]].atom[j].van_table_idx;

						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							return;
						}
					}
				}
			}
		}
		pconflictTDev[idx] = 0;
	}
}
__global__ void CheckGroups_12(unsigned char* pconflictTDev, FRAGMENT** psrc_frag, BOND* bond, unsigned int conflictTlen, unsigned int* pcomidx_dev, float threshold_factor)
{
	unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < (unsigned long long)conflictTlen)
	{
		unsigned long long pose_idx[8];

		pose_idx[0]  = pcomidx_dev[idx] / ConfStepCntDev[7]; // 片段4的姿态
		pose_idx[1]  = pcomidx_dev[idx] / ConfStepCntDev[6]; // 片段5的姿态
		pose_idx[2]  = pcomidx_dev[idx] / ConfStepCntDev[5]; // 片段6的姿态
		pose_idx[3]  = pcomidx_dev[idx] / ConfStepCntDev[4]; // 片段7的姿态
		pose_idx[4]  = pcomidx_dev[idx] / ConfStepCntDev[3]; // 片段8的姿态
		pose_idx[5]  = pcomidx_dev[idx] / ConfStepCntDev[2]; // 片段9的姿态
		pose_idx[6]  = pcomidx_dev[idx] / ConfStepCntDev[1]; // 片段10的姿态
		pose_idx[7]  = pcomidx_dev[idx] ; // 片段11的姿态
		for (int m = 4; m < 8; m++)
		{
			for (int n = 8; n < 12; n++)
			{
				for (int i = 0; i < psrc_frag[m]->atom_cnt; i++)
				{
					for (int j = 0; j < psrc_frag[n]->atom_cnt; j++)
					{
						if (IsOnBondDev(bond, 11, m, n, i, j))	continue;
						float dx = psrc_frag[m][pose_idx[m - 4]].atom[i].X - psrc_frag[n][pose_idx[n - 4]].atom[j].X;
						float dy = psrc_frag[m][pose_idx[m - 4]].atom[i].Y - psrc_frag[n][pose_idx[n - 4]].atom[j].Y;
						float dz = psrc_frag[m][pose_idx[m - 4]].atom[i].Z - psrc_frag[n][pose_idx[n - 4]].atom[j].Z;

						int vanidx1 = psrc_frag[m][pose_idx[m - 4]].atom[i].van_table_idx;
						int vanidx2 = psrc_frag[n][pose_idx[n - 4]].atom[j].van_table_idx;
						if (sqrt(dx * dx + dy * dy + dz * dz) < threshold_factor * VanderWaalsDev[vanidx1][vanidx2])
						{
							return;
						}
					}
				}
			}
		}
		pconflictTDev[idx] = 0;
	}
}
/*-------------------------------------------------------------------------------------------*/


/*--------------------build inter-group conflict table available for DFS---------------------*/
//Function: build inter-group conflict table available for DFS
//Argument:
//   [out]1.unsigned char** pGroupsCrashTHost:conflict look up table
//   [in ]2.FRAGMENT** pfrag:the source fragments data 
//   [in ]3.BOND* bond_dev:the bonds data
//   [in ]4.GROUP* groups:groups data
//   [in ]5.unsigned long long len:maximum number of bytes of lookup table occupying
//   [in ]6.double threshold_factor:conflict threshold coefficent
//Return value:
void BuildGroups01ConflictT(unsigned char** pGroupsCrashTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long len, double threshold_fact)
{
	unsigned char* ptmp_crashTDev;
	unsigned int *pcomidx_dev;
	cudaMalloc(&ptmp_crashTDev, len);
	
	//把内存中的两组的组合号存入显存
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int) * len);
	cudaMemcpy(pcomidx_dev, groups[1].pcombidx, sizeof(unsigned int) * len, cudaMemcpyHostToDevice);

	//检测组0和组1是否冲突，并把冲突表转出入内存
	cudaMemset(ptmp_crashTDev, (unsigned char)1, len);
	CheckGroups_01<<<(len + 511) >> 9 ,512>>>(ptmp_crashTDev, pfrag, bond_dev, len, pcomidx_dev, threshold_fact);
	
	pGroupsCrashTHost[0] = new unsigned char[len];
	cudaMemcpy(pGroupsCrashTHost[0], ptmp_crashTDev, len, cudaMemcpyDeviceToHost);

	cudaFree(ptmp_crashTDev);
	cudaFree(pcomidx_dev);

	unsigned int cnt = 0;
	for (unsigned long long i = 0; i < len; i++)
	{
		cnt += (0 == (pGroupsCrashTHost[0][i]));
	}

	std::cout <<"Group(0) vs Group(1): " << ((float)cnt) / len << std::endl;
}
void BuildGroups02ConflictT(unsigned char** pGroupsCrashTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long crashTlen, double threshold_fact)
{
	unsigned  char* ptmp_crashTDev;
	cudaMalloc(&ptmp_crashTDev, crashTlen);

	unsigned int* pcomidx_dev;

	//把内存中的两组的组合号存入显存
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int) * crashTlen);
	cudaMemcpy(pcomidx_dev, groups[2].pcombidx, sizeof(unsigned int) * crashTlen, cudaMemcpyHostToDevice);

	//检测组0和组2是否冲突，并把冲突表转出入内存
	cudaMemset(ptmp_crashTDev, (unsigned char)1, crashTlen);
	CheckGroups_02<<<(crashTlen + 511) >> 9, 512>>>(ptmp_crashTDev, pfrag, bond_dev, crashTlen, pcomidx_dev, threshold_fact);

	pGroupsCrashTHost[1] = new unsigned char[crashTlen];
	cudaMemcpy((unsigned char*)pGroupsCrashTHost[1], (unsigned char*)ptmp_crashTDev, crashTlen, cudaMemcpyDeviceToHost);

	cudaFree(ptmp_crashTDev);
	cudaFree(pcomidx_dev);

	unsigned int cnt = 0;
	for (unsigned long long i = 0; i < crashTlen; i++)
	{
		cnt += (0 == (pGroupsCrashTHost[1][i]));
	}

	std::cout << "Group(0) vs Group(2): " << ((float)cnt) / crashTlen << std::endl;
}
void BuildGroups12ConflictT(unsigned char** pGroupsCrashTHost, FRAGMENT** pfrag, BOND* bond_dev, GROUP* groups, unsigned long long crashTlen, double threshold_fact)
{
	unsigned  char* ptmp_crashTDev;
	unsigned int* pcomidx_dev;
	cudaMalloc(&ptmp_crashTDev, crashTlen);
	//把内存中的两组的组合号存入显存
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int) * crashTlen);
	cudaMemcpy(pcomidx_dev, groups[2].pcombidx, sizeof(unsigned int) * crashTlen, cudaMemcpyHostToDevice);

	//检测组1和组2是否冲突，并把冲突表转出入内存
	cudaMemset(ptmp_crashTDev, (unsigned char)1, crashTlen);
	CheckGroups_12<<<(crashTlen + 511) >> 9,512>>>(ptmp_crashTDev, pfrag, bond_dev, crashTlen, pcomidx_dev, threshold_fact);

	pGroupsCrashTHost[2] = new unsigned char[crashTlen];
	cudaMemcpy(pGroupsCrashTHost[2], ptmp_crashTDev, crashTlen, cudaMemcpyDeviceToHost);


	cudaFree(ptmp_crashTDev);
	cudaFree(pcomidx_dev);

	unsigned int cnt = 0;
	for (unsigned long long i = 0; i < crashTlen; i++)
	{
		cnt += (0 == (pGroupsCrashTHost[2][i]));
	}
	std::cout << "Group(1) vs Group(2): " << ((float)cnt) / crashTlen << std::endl;
}
/*-------------------------------------------------------------------------------------------*/


/*-------------------build group & pocket conflict table available for DFS-------------------*/
//Function: build group & pocket conflict table available for DFS
//Argument:
//   [out]1.unsigned char** pGroupsCrashTHost:conflict look up table
//   [in ]2.ATOM* pocket_atom_dev:pocket atom data
//   [in ]3.FRAGMENT** psrc_frag:the source fragments data 
//   [in ]4.GROUP* groups:groups data
//   [in ]5.int pocket_atom_cnt:number of pocket atoms
//   [in ]6.double threshold_factor:conflict threshold coefficent
//Return value:
void BuildGroup0_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom_dev, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, double threshold_factor)
{
	unsigned char* pPocketConflictTDev;
	unsigned int* pcomidx_dev;	
	GROUP* pgroup_dev;
	
	//创建组与口袋的冲突表
	pPocketConflictTHost[0] = new unsigned char[groups[0].comb_cnt];
	cudaMalloc(&pPocketConflictTDev, groups[0].comb_cnt);

	//创建GPU上的group数据和组合号数据（group0）
	cudaMalloc(&pgroup_dev, sizeof(GROUP));
	cudaMemcpy(pgroup_dev, &groups[0], sizeof(GROUP), cudaMemcpyHostToDevice);
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int)* groups[0].comb_cnt);
	cudaMemcpy(pcomidx_dev, groups[0].pcombidx, sizeof(unsigned int)* groups[0].comb_cnt,cudaMemcpyHostToDevice);
	
	//检查组0(片段0-1)和口袋冲5
	cudaMemset(pPocketConflictTDev,(unsigned char)1,groups[0].comb_cnt);
	CheckGroup0_Pocket<<<(groups[0].comb_cnt>>9)+1, 512 >> >(pPocketConflictTDev, pocket_atom_dev, psrc_frag, pgroup_dev,pcomidx_dev, pocket_atom_cnt, threshold_factor);
	cudaMemcpy(pPocketConflictTHost[0], pPocketConflictTDev, groups[0].comb_cnt, cudaMemcpyDeviceToHost);
	//统计运算结果
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < groups[0].comb_cnt; i++)
	{
		cnt += (0 == pPocketConflictTHost[0][i]);
	}
	std::cout << "Groups(0) vs Pocket :" << ((float)cnt) / groups[0].comb_cnt << std::endl;
	
	//释放内存
	cudaFree(pcomidx_dev);
	cudaFree(pgroup_dev);
	cudaFree(pPocketConflictTDev);
}
void BuildGroup1_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom_dev, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, double threshold_factor)
{
	unsigned char* pPocketConflictTDev;
	unsigned int* pcomidx_dev;	
	GROUP* pgroup_dev;
	
	//创建组与口袋的冲突表
	pPocketConflictTHost[1] = new unsigned char[groups[1].comb_cnt];
	cudaMalloc(&pPocketConflictTDev, groups[1].comb_cnt);

	//创建GPU上的group数据和组合号数据（group0）
	cudaMalloc(&pgroup_dev, sizeof(GROUP));
	cudaMemcpy(pgroup_dev, &groups[1], sizeof(GROUP), cudaMemcpyHostToDevice);
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int)* groups[1].comb_cnt);
	cudaMemcpy(pcomidx_dev, groups[1].pcombidx, sizeof(unsigned int)* groups[1].comb_cnt,cudaMemcpyHostToDevice);
	
	//检查组1(片段2-3)和口袋冲5
	cudaMemset(pPocketConflictTDev, (unsigned char)1, groups[1].comb_cnt);
	CheckGroup1_Pocket<<<(groups[1].comb_cnt >> 9) + 1, 512 >> >(pPocketConflictTDev, pocket_atom_dev, psrc_frag, pgroup_dev,pcomidx_dev, pocket_atom_cnt, threshold_factor);
	cudaMemcpy(pPocketConflictTHost[1], pPocketConflictTDev, groups[1].comb_cnt, cudaMemcpyDeviceToHost);

	//统计运算结果
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < groups[1].comb_cnt; i++)
	{
		cnt += (0 == pPocketConflictTHost[1][i]);
	}
	std::cout << "Groups(1) vs Pocket :" << ((float)cnt) / groups[1].comb_cnt << std::endl;

	//释放内存
	cudaFree(pcomidx_dev);
	cudaFree(pgroup_dev);
	cudaFree(pPocketConflictTDev);
}
void BuildGroup2_PocketConflictT(unsigned char** pPocketConflictTHost, ATOM* pocket_atom_dev, FRAGMENT** psrc_frag, GROUP* groups, int pocket_atom_cnt, double threshold_factor)
{
	unsigned unsigned char* pPocketConflictTDev;
	unsigned int* pcomidx_dev;	
	GROUP* pgroup_dev;
	
	//创建组与口袋的冲突表
	pPocketConflictTHost[2] = new unsigned char[groups[2].comb_cnt];
	cudaMalloc(&pPocketConflictTDev, groups[2].comb_cnt);

	//创建GPU上的group数据和组合号数据（group0）
	cudaMalloc(&pgroup_dev, sizeof(GROUP));
	cudaMemcpy(pgroup_dev, &groups[2], sizeof(GROUP), cudaMemcpyHostToDevice);
	cudaMalloc(&pcomidx_dev, sizeof(unsigned int)* groups[2].comb_cnt);
	cudaMemcpy(pcomidx_dev, groups[2].pcombidx, sizeof(unsigned int)* groups[2].comb_cnt,cudaMemcpyHostToDevice);
	
	//检查组1(片段4-5)和口袋冲5
	cudaMemset(pPocketConflictTDev, (unsigned char)1, groups[2].comb_cnt);
	CheckGroup2_Pocket<<<(groups[2].comb_cnt >> 9) + 1, 512 >> >(pPocketConflictTDev, pocket_atom_dev, psrc_frag, pgroup_dev,pcomidx_dev, pocket_atom_cnt, threshold_factor);
	cudaMemcpy(pPocketConflictTHost[2], pPocketConflictTDev, groups[2].comb_cnt, cudaMemcpyDeviceToHost);

	//统计运算结果
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < groups[2].comb_cnt; i++)
	{
		cnt += (0 == pPocketConflictTHost[2][i]);
	}
	std::cout << "Groups(2) vs Pocket :" << ((float)cnt) / groups[2].comb_cnt << std::endl;

	//释放内存
	cudaFree(pcomidx_dev);
	cudaFree(pgroup_dev);
	cudaFree(pPocketConflictTDev);
	//cudaFree(pocket_atom_dev);
}
/*-------------------------------------------------------------------------------------------*/
