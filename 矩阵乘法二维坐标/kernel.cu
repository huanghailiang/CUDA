#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"

#include<time.h>

#define BLOCK_SIZE 16
//设定数组的 行数、列数,方阵N*N
#define n 12500
#define num_row 12500
#define num_col 12500

__global__ void matrix_matrix(const double *a, size_t num_a_row, const double *b, size_t num_b_row,  double *c, size_t num_c_row, int n1);
void cpu_matrix_matrix( double *h_a, size_t a_row,  double *h_b, size_t b_row,  double *h_c, size_t c_row, int n1);
void print_matrix(const double *A, const int a_row, const int a_col);
int main()
{
	
	double *H_A, *H_B, *H_C;
	double *D_A, *D_B, *D_C;
	clock_t cpu_start, cpu_end, gpu_start, gpu_end;




	//if (!InitCUDA()) return 0;
	
	H_A = (double*)malloc(sizeof(double)*num_row*num_col);
	H_B = (double*)malloc(sizeof(double)*num_row*num_col);
	H_C = (double*)malloc(sizeof(double)*num_row*num_col);

	int i, j, k;

	for (i = 0; i<n; i++)
	{
		for (j = 0; j<n; j++)
		{
			H_A[i*num_row + j] = i*num_row + j;
			H_B[i*num_row + j] = i*num_row + j;
		}
	}
cpu_start = clock();
	cpu_matrix_matrix(H_A, num_row, H_B, num_row, H_C, num_row, n);
	//print_matrix(H_C, num_row, num_col);

cpu_end = clock();
double cpu_time = (double)((cpu_end - cpu_start)) / CLOCKS_PER_SEC;

	//  cudaMalloc((void**)&D_A,sizeof(double)*row*col);
	//  cudaMalloc((void**)&D_B,sizeof(double)*row*col);
	//  cudaMalloc((void**)&D_C,sizeof(double)*row*col);
	//cudaMallocPitch函数会以适当的倍数配置内存，并把配置的宽度传回
gpu_start = clock();	
	size_t pitch_A, pitch_B, pitch_C;
	cudaMallocPitch((void**)&D_A, &pitch_A, sizeof(double)*num_row, num_col);
	cudaMallocPitch((void**)&D_B, &pitch_B, sizeof(double)*num_row, num_col);
	cudaMallocPitch((void**)&D_C, &pitch_C, sizeof(double)*num_row, num_col);

	//置零
	cudaMemset(D_A, 0, pitch_A*num_col);
	cudaMemset(D_B, 0, pitch_A*num_col);

	//  cudaMemcpy2D(D_A, sizeof(double)*row, A, sizeof(double)*row, sizeof(double)*row, col, cudaMemcpyHostToDevice );
	//  cudaMemcpy2D(D_B, sizeof(double)*row, B, sizeof(double)*row, sizeof(double)*row, col, cudaMemcpyHostToDevice );

	cudaMemcpy2D(D_A, pitch_A, H_A, sizeof(double)*num_row, sizeof(double)*num_row, num_col, cudaMemcpyHostToDevice);
	cudaMemcpy2D(D_B, pitch_B, H_B, sizeof(double)*num_row, sizeof(double)*num_row, num_col, cudaMemcpyHostToDevice);

	int bx = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 blocks(bx, bx);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	matrix_matrix << <blocks, threads >> > (D_A, pitch_A / sizeof(double), D_B, pitch_B / sizeof(double), D_C, pitch_C / sizeof(double), n);
gpu_end = clock();
	cudaMemcpy2D(H_C, sizeof(double)*num_row, D_C, pitch_C, sizeof(double)*num_row, num_col, cudaMemcpyDeviceToHost);

	cudaFree(D_A);
	cudaFree(D_B);
	cudaFree(D_C);

	//print_matrix(H_C, num_row, num_col);
	
	double gpu_time = (double)((gpu_end - gpu_start)) / CLOCKS_PER_SEC;
	printf("GPU:%.10f\n", gpu_time);
	printf("CPU:%.10f\n", cpu_time);

}

__global__ void matrix_matrix(const double *a, size_t num_a_row, const double *b, size_t num_b_row,  double *c, size_t num_c_row, int n2)
{
	__shared__ double ma[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double mb[BLOCK_SIZE][BLOCK_SIZE];

	const int tid_col = threadIdx.x;
	const int tid_row = threadIdx.y;

	const int bid_col = blockIdx.x*BLOCK_SIZE;
	const int bid_row = blockIdx.y*BLOCK_SIZE;

	int i, j;
	double sum = 0;
	double cmp = 0;
	//int n1 = n2 / 2;

	for (j = 0; j < n2; j += BLOCK_SIZE)
	{
		
		ma[tid_row][tid_col] = a[(bid_row + tid_row)*num_a_row + j + tid_col];
		mb[tid_row][tid_col] = b[(j + tid_row)*num_b_row + bid_col + tid_col];

		//__syncthreads();

		for (i = 0; i<BLOCK_SIZE; i++)
		{
			double tmp;
			cmp -= ma[tid_row][i] * mb[i][tid_col];  //保证精度
			tmp = sum - cmp;
			cmp = (tmp - sum) + cmp;
			sum = tmp;
		}
		//__syncthreads();
	}
	c[(bid_row + tid_row)*num_c_row + bid_col + tid_col] = sum;

	//for (j = n1; j < n2; j += BLOCK_SIZE)
	//{
	//	ma[tid_row][tid_col] = a[(bid_row + tid_row)*num_a_row + j + tid_col];
	//	mb[tid_row][tid_col] = b[(j + tid_row)*num_b_row + bid_col + tid_col];

	//	//__syncthreads();

	//	for (i = 0; i<BLOCK_SIZE; i++)
	//	{
	//		double tmp;
	//		cmp -= ma[tid_row][i] * mb[i][tid_col];  //保证精度
	//		tmp = sum - cmp;
	//		cmp = (tmp - sum) + cmp;
	//		sum = tmp;
	//	}
	//	//__syncthreads();
	//}
	//c[(bid_row + tid_row)*num_c_row + bid_col + tid_col] = sum;
}
void cpu_matrix_matrix( double *h_a, size_t a_row,  double *h_b, size_t b_row,  double *h_c, size_t c_row, int n1)
{
	int i, j, k;

	//for (i = 0; i<n1; i++)
	//{
	//	for (j = 0; j<n1; j++)
	//	{
	//		h_a[i*a_row + j] = i*a_row + j;
	//		h_b[i*b_row + j] = i*b_row + j;
	//	}
	//}

	for (i = 0; i<n1; i++)
	{
		for (j = 0; j<n1; j++)
		{
			double t = 0;
			for (k = 0; k<n1; k++)
			{
				t += h_a[i*a_row + k] * h_b[k*b_row + j];
			}
			h_c[i*c_row + j] = t;
		}
	}
}
void print_matrix(const double *A, const int a_row, const int a_col)
{
	int i, j;
	for (int i = 0; i < a_row; i++)
	{
		for (int j = 0; j < a_col; j++)
		{
			printf("%.6f", A[i*a_col + j]);
			printf(" ");
		}
		printf("\n");
	}
	printf("\n");
}
