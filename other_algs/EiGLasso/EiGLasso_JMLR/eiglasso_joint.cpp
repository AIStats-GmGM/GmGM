// matrix: row-major
// symmetric matrix: lower triangular


#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>

//#include <lapacke.h>
//#include <cblas.h>
#include "mkl.h"

//using namespace std;

typedef struct {
	unsigned int i;
	unsigned int j;
} uint_pair;


static inline double computeLogdet(int size_left, int size_right, const double* eigval_left, const double* eigval_right)
{
	double logdet = 0;
	for (int i = 0; i < size_left; i++) {
		for (int j = 0; j < size_right; j++) {
			logdet += log(eigval_left[i] + eigval_right[j]);
		}
	}
	return logdet;
}

static inline double computeNewtonJoint(int t_outer, int dim_the, int dim_psi, const double* S_the, const double* S_psi, 
	const double* Theta, const double* Psi, const double* W_the, const double* W_psi,
	const int num_eig, double** Ak_the, double** Ak_psi, double** Uk_the, double** Uk_psi,
	double* eigvec_the, double* eigvec_psi,
	double* D_the, double* D_psi, const double gamma_the, const double gamma_psi, const double tol, const double alpha_prev, const double tr_ratio)
{
	srand(25252);

	double subgrad = 0;
	unsigned int size_active_the = 0;
	unsigned int size_active_psi = 0;
	uint_pair* activeSet_the = (uint_pair*)calloc((dim_the * (dim_the + 1) / 2), sizeof(uint_pair));
	uint_pair* activeSet_psi = (uint_pair*)calloc((dim_psi * (dim_psi + 1) / 2), sizeof(uint_pair));

	for (int i = 0; i < dim_the; i++) {
		for (int j = 0; j <= i; j++) {
			int ij = i * dim_the + j;

			double g = dim_psi * S_the[ij] - W_the[ij];

			if (Theta[ij] != 0.0 || fabs(g) > dim_psi * gamma_the) {
				activeSet_the[size_active_the].i = (unsigned int)i;
				activeSet_the[size_active_the].j = (unsigned int)j;
				size_active_the++;
				if (Theta[ij] > 0)
					g += gamma_the;
				else if (Theta[ij] < 0)
					g -= gamma_the;
				else
					g = fabs(g) / dim_psi - gamma_the;
				subgrad += fabs(g);
			}
		}
	}

	for (int i = 0; i < dim_psi; i++) {
		for (int j = 0; j <= i; j++) {
			int ij = i * dim_psi + j;

			double g = dim_the * S_psi[ij] - W_psi[ij];

			if (Psi[ij] != 0.0 || fabs(g) > dim_the * gamma_psi) {
				activeSet_psi[size_active_psi].i = (unsigned int)i;
				activeSet_psi[size_active_psi].j = (unsigned int)j;
				size_active_psi++;
				if (Psi[ij] > 0)
					g += gamma_psi;
				else if (Psi[ij] < 0)
					g -= gamma_psi;
				else
					g = fabs(g) / dim_the - gamma_psi;
				subgrad += fabs(g);
			}
		}
	}

	// compute Newton direction
	double diffD = 0;
	double normD = 0;
	//for (int t_cd = 0; t_cd <= t_outer / 3; t_cd++) {
	for (int t_cd = 0; t_cd < 100; t_cd++) {

		diffD = 0;

		// permutation
		for (unsigned int ii = 0; ii < size_active_the; ii++) {
			unsigned int jj = ii + rand() % (size_active_the - ii);
			unsigned int k1 = activeSet_the[ii].i;
			unsigned int k2 = activeSet_the[ii].j;
			activeSet_the[ii].i = activeSet_the[jj].i;
			activeSet_the[ii].j = activeSet_the[jj].j;
			activeSet_the[jj].i = k1;
			activeSet_the[jj].j = k2;
		}
		for (unsigned int ii = 0; ii < size_active_psi; ii++) {
			unsigned int jj = ii + rand() % (size_active_psi - ii);
			unsigned int k1 = activeSet_psi[ii].i;
			unsigned int k2 = activeSet_psi[ii].j;
			activeSet_psi[ii].i = activeSet_psi[jj].i;
			activeSet_psi[ii].j = activeSet_psi[jj].j;
			activeSet_psi[jj].i = k1;
			activeSet_psi[jj].j = k2;
		}


		// cd
		unsigned int l_the = 0;
		unsigned int l_psi = 0;
		int is_the_over = 0;
		int is_psi_over = 0;
		for (unsigned int l = 0; l < size_active_the + size_active_psi; l++) {

			//int update_which; // 0 = theta, 1 = psi

			uint_pair* activeSet;
			int dim = 0;
			int dim_other = 0;
			const double* X;
			double** Ak;
			double** Uk;
			const double* S;
			const double* W;
			double* D;
			double gamma;
			unsigned int lll ;

			int theorpsi = rand() % 2;
			if ( (is_psi_over == 1) || (theorpsi == 0 && l_the < size_active_the) ) { // update D_Theta
				activeSet = activeSet_the;
				dim = dim_the;
				dim_other = dim_psi;
				X = Theta;
				Ak = Ak_the;
				Uk = Uk_the;
				S = S_the;
				W = W_the;
				D = D_the;
				gamma = gamma_the;
				lll = l_the;

				//update_which = 0;

				l_the++;
			} else if ( (is_the_over == 1) || (theorpsi == 1 && l_psi < size_active_psi) ) { // update D_Psi
				activeSet = activeSet_psi;
				dim = dim_psi;
				dim_other = dim_the;
				X = Psi;
				Ak = Ak_psi;
				Uk = Uk_psi;
				S = S_psi;
				W = W_psi;
				D = D_psi;
				gamma = gamma_psi;
				lll = l_psi;

				//update_which = 1;

				l_psi++;
			} else if (l_the >= size_active_the) {
				is_the_over = 1;
				
				activeSet = activeSet_psi;
				dim = dim_psi;
				dim_other = dim_the;
				X = Psi;
				Ak = Ak_psi;
				Uk = Uk_psi;
				S = S_psi;
				W = W_psi;
				D = D_psi;
				gamma = gamma_psi;
				lll = l_psi;

				//update_which = 1;

				l_psi++;

			} else if (l_psi >= size_active_psi) {
				is_psi_over = 1;
				
				activeSet = activeSet_the;
				dim = dim_the;
				dim_other = dim_psi;
				X = Theta;
				Ak = Ak_the;
				Uk = Uk_the;
				S = S_the;
				W = W_the;
				D = D_the;
				gamma = gamma_the;
				lll = l_the;

				//update_which = 0;

				l_the++;

			} else {
				std::cout << "error" << std::endl;
				activeSet = activeSet_the;
				dim = dim_the;
				dim_other = dim_psi;
				X = Theta;
				Ak = Ak_the;
				Uk = Uk_the;
				S = S_the;
				W = W_the;
				D = D_the;
				gamma = gamma_the;
				lll = l_the;
				break;
			}
			unsigned int i = activeSet[lll].i;
			unsigned int j = activeSet[lll].j;

			unsigned long idim = i * dim;
			unsigned long jdim = j * dim;
			unsigned long ij = idim + j;


			double a = 0;
			for (int k = 0; k < num_eig; k++) {
				a += Ak[k][ij] * Ak[k][ij];
			}
			if (num_eig < dim_other)
				a += (dim_other - num_eig) * Ak[num_eig-1][ij] * Ak[num_eig-1][ij];
			if (i != j) {
				for (int k = 0; k < num_eig; k++) {
					a += Ak[k][idim + i] * Ak[k][jdim + j];
				}
				if (num_eig < dim_other)
					a += (dim_other - num_eig) * Ak[num_eig-1][idim + i] * Ak[num_eig-1][jdim + j];
			}

			double b = dim_other * S[ij] - W[ij];
			for (int k = 0; k < num_eig; k++) {
				for (int r = 0; r < dim; r++)
					b += Ak[k][idim + r] * Uk[k][r * dim + j];
			}
			if (num_eig < dim_other) {
				for (int r = 0; r < dim; r++) {
					b += (dim_other - num_eig) * (Ak[num_eig-1][idim + r] * Uk[num_eig-1][r * dim + j]);
				}
			}

			double c = X[ij] + D[ij];
			double ll = dim_other * gamma / a;
			double f = b / a;
			double mu = 0;
			normD -= dim_other * fabs(D[ij]);
			if (i != j) {
				if (c > f) {
					mu = -f - ll;
					if (c + mu < 0) {
						mu = -c;
						D[ij] = -X[ij];
					}
					else {
						D[ij] += mu;
					}
				}
				else {
					mu = -f + ll;
					if (c + mu > 0) {
						mu = -c;
						D[ij] = -X[ij];
					}
					else {
						D[ij] += mu;
					}
				}
			}
			else {
				mu = -f;
				D[ij] += mu;
			}
			diffD += dim_other * fabs(mu);
			normD += dim_other * fabs(D[ij]);

			if (mu != 0) {
				for (int k = 0; k < num_eig; k++) {
					for (int r = 0; r < dim; r++) {
						Uk[k][idim + r] += mu * Ak[k][jdim + r];
					}				
				}
				if (i != j) {
					for (int k = 0; k < num_eig; k++) {
						for (int r = 0; r < dim; r++) {
							Uk[k][jdim + r] += mu * Ak[k][idim + r];
						}
					}
				}

			}
		}
		if (diffD <= normD * tol)
			break;
	}
	free(activeSet_the);
	free(activeSet_psi);
	return subgrad;
}

static inline double lineSearchJoint(const int dim_the, const int dim_psi, const int max_iter, const double* S_the, const double* S_psi, 
	double* D_the, double* D_psi, const double* Theta, const double* Psi, double* W_the, double* W_psi,
	const double gamma_the, const double gamma_psi, const double sigma, 
	double& fX, double& fXprev, double& logdet, double& l1normX_the, double& l1normX_psi,
	double& trSX_the, double& trSX_psi, double* eigval_the, double* eigval_psi)
{
	double trgradgD = 0;
	for (int i = 0; i < dim_the; i++) {
		for (int j = 0; j < i; j++) {
			unsigned long ij = i * dim_the + j;
			trgradgD += (dim_psi * S_the[ij] - W_the[ij]) * D_the[ij];
		}
	}
	for (int i = 0; i < dim_psi; i++) {
		for (int j = 0; j < i; j++) {
			unsigned long ij = i * dim_psi + j;
			trgradgD += (dim_the * S_psi[ij] - W_psi[ij]) * D_psi[ij];
		}
	}
	trgradgD *= 2;
	for (int i = 0; i < dim_the; i++) {
		unsigned long ii = i * dim_the + i;
		trgradgD += (dim_psi * S_the[ii] - W_the[ii]) * D_the[ii];
	}
	for (int i = 0; i < dim_psi; i++) {
		unsigned long ii = i * dim_psi + i;
		trgradgD += (dim_the * S_psi[ii] - W_psi[ii]) * D_psi[ii];
	}

	double alpha = 1;
	double beta = 0.5;

	double l1normXD = 0;

	double l1normX1_the = 0;
	double trSX1_the = 0;
	double l1normX1_psi = 0;
	double trSX1_psi = 0;

	int alpha_chosen = 0;

	for (int t_line = 0; t_line < max_iter; t_line++) {

		l1normX1_the = 0;
		trSX1_the = 0;
		l1normX1_psi = 0;
		trSX1_psi = 0;

		for (int i = 0; i < dim_the; i++) {
			for (int j = 0; j < i; j++) {
				unsigned long ij = i * dim_the + j;
				W_the[ij] = Theta[ij] + alpha * D_the[ij]; // store theta + alpha*D
				l1normX1_the += fabs(W_the[ij]);
				trSX1_the += W_the[ij] * S_the[ij];
			}
		}
		l1normX1_the *= 2 * gamma_the; // diagonals are not regularized
		trSX1_the *= 2;
		for (int i = 0; i < dim_the; i++) {
			unsigned long ii = i * dim_the + i;
			W_the[ii] = Theta[ii] + alpha * D_the[ii]; // store theta + alpha*D
			//W_the[ii] += 1e-8;
			trSX1_the += W_the[ii] * S_the[ii];
		}

		for (int i = 0; i < dim_psi; i++) {
			for (int j = 0; j < i; j++) {
				unsigned long ij = i * dim_psi + j;
				W_psi[ij] = Psi[ij] + alpha * D_psi[ij]; // store theta + alpha*D
				l1normX1_psi += fabs(W_psi[ij]);
				trSX1_psi += W_psi[ij] * S_psi[ij];
			}
		}
		l1normX1_psi *= 2 * gamma_psi; // diagonals are not regularized
		trSX1_psi *= 2;
		for (int i = 0; i < dim_psi; i++) {
			unsigned long ii = i * dim_psi + i;
			W_psi[ii] = Psi[ii] + alpha * D_psi[ii]; // store theta + alpha*D
			//W_psi[ii] += 1e-8;
			trSX1_psi += W_psi[ii] * S_psi[ii];
		}

		if (alpha == 1.0) {
			l1normXD = dim_the * l1normX1_psi + dim_psi * l1normX_the;
		}

		lapack_int info_the = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'N', 'L', dim_the, W_the, dim_the, eigval_the);
		if (info_the != 0) {
			//if (MSG >= MSG_MIN) {
			//	std::cout << t_line << ">> Line search: (the) illegal argument for dsyevd (" << info_the << "). alpha=" << alpha << std::endl;
			//}
			alpha = 0;
			break;
		}
		lapack_int info_psi = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'N', 'L', dim_psi, W_psi, dim_psi, eigval_psi);
		if (info_psi != 0) {
			//if (MSG >= MSG_MIN) {
			//	std::cout << t_line << ">> Line search: (psi) illegal argument for dsyevd (" << info_psi << "). alpha=" << alpha << std::endl;
			//}
			alpha = 0;
			break;
		}

		if (eigval_the[0] + eigval_psi[0] <= 0) {
			//if (MSG >= MSG_MIN) {
			//	std::cout << t_line << ">> Line search: OMEGA not positive definite (" << eigval_the[0] << ", " << eigval_psi[0] << "). alpha=" << alpha << std::endl;
			//}
			alpha *= beta;
			continue;
		}

		double logdetX1 = 0;
		for (int i = 0; i < dim_the; i++) {
			for (int j = 0; j < dim_psi; j++) {
				logdetX1 += log(eigval_the[i] + eigval_psi[j]);
			}
		}

		double l1normX1_joint = dim_the * l1normX1_psi + dim_psi * l1normX1_the;
		double l1normX_joint = dim_the * l1normX_psi + dim_psi * l1normX_the;
		double fX1 = dim_the * trSX1_psi + dim_psi * trSX1_the - logdetX1 + l1normX1_joint;



		//if (MSG >= MSG_VAL) {
		//	std::cout << "  fX1        = " << fX1 << std::endl;
		//	std::cout << "  fX+asd     = " << fX + alpha * sigma * (trgradgD + l1normXD - l1normX_joint) << std::endl;
		//	std::cout << "  delta      = " << trgradgD + l1normXD - l1normX_joint << std::endl;
		//
		//}

		if (fX1 <= fX + alpha * sigma * (trgradgD + l1normXD - l1normX_joint)) {
			//if (MSG >= MSG_MIN)
				//std::cout << t_line << ">> Line search: (1) step size chosen=" << alpha << std::endl;

			//fXprev = fX;
			fX = fX1;
			l1normX_the = l1normX1_the;
			l1normX_psi = l1normX1_psi;
			logdet = logdetX1;
			trSX_the = trSX1_the;
			trSX_psi = trSX1_psi;
			alpha_chosen = 1;
			break;
		}

		//if (MSG >= MSG_MIN)
		//	std::cout << t_line << ">> Line search: no sufficient decrease." << alpha << std::endl;

		//fX1prev = fX1;
	 	alpha *= beta;
	}
	if (alpha_chosen == 0) {
		alpha = -1;
	}
	return alpha;

}

static inline void computeWAk(const int dim, const double* eigvec, const double* eigval_W, double* W, int num_eig, double* eigval_Ak, double** Ak)
{
	double* tmpQD = (double*)malloc((dim * dim) * sizeof(double));
	// Ak's
	for (int k = 0; k < num_eig; k++) {
		memset(tmpQD, 0, (dim * dim) * sizeof(double));
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				tmpQD[i * dim + j] = eigval_Ak[k * dim + j] * eigvec[i * dim + j];
			}
		}
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dim, dim, dim, 1.0, tmpQD, dim, eigvec, dim, 0.0, Ak[k], dim);
	}
	memset(tmpQD, 0, (dim * dim) * sizeof(double));
	// W
	for (int j = 0; j < dim; j++) {
		for (int i = 0; i < dim; i++) {
			tmpQD[i * dim + j] = eigval_W[j] * eigvec[i * dim + j];
		}
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dim, dim, dim, 1.0, tmpQD, dim, eigvec, dim, 0.0, W, dim);
	free(tmpQD);
}

static inline void distributeEigval(int num_eig, int dim_left, int dim_right, const double* in_left, const double* in_right, double* outW_left, double* outW_right, double* outAk_left, double* outAk_right) {

	memset(outW_right, 0, dim_right * sizeof(double));
	memset(outW_left, 0, dim_left * sizeof(double));
	memset(outAk_right, 0, ((num_eig) * dim_right) * sizeof(double));
	memset(outAk_left, 0, ((num_eig) * dim_left) * sizeof(double));

	// note dsyevd's eigvals are in ascending order

	// eigval_Ak
	for (int i = 0; i < dim_left; i++) {
		for (int k = 0; k < num_eig; k++) {
			outAk_left[k * dim_left + i] = 1.0 / ((in_left[i] + in_right[k]));
		}
	}
	for (int j = 0; j < dim_right; j++) {
		for (int k = 0; k < num_eig; k++) {
			outAk_right[k * dim_right + j] = 1.0 / ((in_left[k] + in_right[j]));
		}
	}
	// eigval_W
	for (int i = 0; i < dim_left; i++) {
		for (int j = 0; j < dim_right; j++) {
			double tmp = 1.0 / (in_left[i] + in_right[j]);
			outW_left[i] += tmp;
			outW_right[j] += tmp;
		}
	}
}

static inline double adjustDiag(int dim_left, int dim_right, double* X_left, double* X_right, double& ratio) {

	double tr_left = 0;
	double tr_right = 0;
	for (int i = 0; i < dim_left; i++) {
		tr_left += X_left[i * dim_left + i];
	}
	for (int i = 0; i < dim_right; i++) {
		tr_right += X_right[i * dim_right + i];
	}

	double c = (tr_right - ratio * tr_left) / (ratio * dim_left + dim_right);
	return c;
}

int main(int argc, char** argv)
{
	std::cout.precision(8);

	// Command Line Arguments

	int p = 0;
	int q = 0;
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"-p") == 0) {
			p = atoi(argv[i+1]);
			break;
		}
	}
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"-q") == 0) {
			q = atoi(argv[i+1]);
			break;
		}
	}

	std::ifstream input_T;
	std::ifstream input_S;
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"-t") == 0) {
			input_T.open(argv[i+1]);
			break;
		}
	}
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"-s") == 0) {
			input_S.open(argv[i+1]);
			break;
		}
	}

	int K_EIG = 1;
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"-k") == 0) {
			K_EIG = atoi(argv[i+1]);
			break;
		}
	}

	std::ofstream output_psi;
	std::ofstream output_the;
	std::ofstream output_info;
	int is_output_given = 0;
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"-o") == 0) {
			std::string output_prefix(argv[i+1]);
			output_psi.open(output_prefix + "psi.txt");
			output_the.open(output_prefix + "theta.txt");
			output_info.open(output_prefix + "info.txt");

			is_output_given = 1;
			break;
		}
	}
	if (is_output_given == 0) {
		output_psi.open("./psi.txt");
		output_the.open("./theta.txt");
		output_info.open("./info.txt");
	}
	
	double gamma_psi = 0.01;
	double gamma_the = 0.01;
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"-r") == 0) {
			gamma_the = atof(argv[i+1]);
			gamma_psi = atof(argv[i+2]);
			break;
		}
	}

	int max_Newton_iter = 10000;
	int max_line_iter = 20;
	double newton_tol = 1e-3;
	double sigma = 0.01;
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"--tol") == 0) {
			newton_tol = atof(argv[i+1]);
			break;
		}
	}
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"--max-newton") == 0) {
			max_Newton_iter = atoi(argv[i+1]) + 1;
			break;
		}
	}
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"--max-line") == 0) {
			max_line_iter = atoi(argv[i+1]) + 1;
			break;
		}
	}
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"--sigma") == 0) {
			sigma = atof(argv[i+1]);
			break;
		}
	}
		
	double tr_ratio = q * 1.0 / p;
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"--trratio") == 0) {
			tr_ratio = atof(argv[i+1]);
			break;
		}
	}

	std::cout << std::endl;
	std::cout << "#############################" << std::endl;
	std::cout << "#    EiGLasso Joint v1.0    #" << std::endl;
	std::cout << "#############################" << std::endl;
	std::cout << std::endl;


	double* Psi = (double*)calloc((q * q), sizeof(double));
	double* Theta = (double*)calloc((p * p), sizeof(double));
	for (int i = 0; i < q; i++)
		Psi[i * q + i] = 1;
		
	for (int i = 0; i < p; i++)
		Theta[i * p + i] = 1;
		

	double* S_psi = (double*)calloc((q * q), sizeof(double));
	double* S_the = (double*)calloc((p * p), sizeof(double));
	if (input_T.is_open() && input_S.is_open()) {
		for (int i = 0; i < q; i++) {
			for (int j = 0; j < q; j++) {
				unsigned long ij = i * q + j;
				input_T >> S_psi[ij];
			}
		}
		for (int i = 0; i < p; i++) {
			for (int j = 0; j < p; j++) {
				unsigned long ij = i * p + j;
				input_S >> S_the[ij];
			}
		}
		input_T.close();
		input_S.close();
	}
	else {
		std::cerr << "Unable to open the input file." << std::endl;
		return -1;
	}

	std::vector<double> iter_times;
	std::vector<double> fs;
	std::vector<double> convgs;

	/////////////////////////////////
	// Eigendecomposition
	/////////////////////////////////

	double* eigval_psi = (double*)malloc(q * sizeof(double));
	double* eigval_the = (double*)malloc(p * sizeof(double));
	
	double* eigvec_psi = (double*)calloc((q * q), sizeof(double));
	double* eigvec_the = (double*)calloc((p * p), sizeof(double));
	for (int i = 0; i < q; i++) {
		for (int j = 0; j <= i; j++) {
			eigvec_psi[i * q + j] = Psi[i * q + j];
			if (i == j)
				eigvec_psi[i * q + i] += 1e-8; 
		}
	}
	for (int i = 0; i < p; i++) {
		for (int j = 0; j <= i; j++) {
			eigvec_the[i * p + j] = Theta[i * p + j];
			if (i == j)
				eigvec_the[i * p + i] += 1e-8;
		}
	}
 

	LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', q, eigvec_psi, q, eigval_psi);
	LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', p, eigvec_the, p, eigval_the);

	double* eigval_W_psi = (double*)malloc(q * sizeof(double));
	double* eigval_W_the = (double*)malloc(p * sizeof(double));

	double* eigval_Ak_psi = (double*)malloc((K_EIG) * q * sizeof(double));
	double* eigval_Ak_the = (double*)malloc((K_EIG) * p * sizeof(double));

	distributeEigval(K_EIG, p, q, eigval_the, eigval_psi, eigval_W_the, eigval_W_psi, eigval_Ak_the, eigval_Ak_psi);


	double* W_psi = (double*)calloc((q * q), sizeof(double));
	double* W_the = (double*)calloc((p * p), sizeof(double));

	double** Ak_the = (double**)malloc((K_EIG) * sizeof(double*));
	for (int i = 0; i < K_EIG; i++) {
		Ak_the[i] = (double*)malloc((p * p) * sizeof(double));
	}
	computeWAk(p, eigvec_the, eigval_W_the, W_the, K_EIG, eigval_Ak_the, Ak_the);

	double** Ak_psi = (double**)malloc((K_EIG) * sizeof(double*));
	for (int i = 0; i < K_EIG; i++) {
		Ak_psi[i] = (double*)malloc((q * q) * sizeof(double));
	}
	computeWAk(q, eigvec_psi, eigval_W_psi, W_psi, K_EIG, eigval_Ak_psi, Ak_psi);


	/////////////////////////////////
	// Some global variables
	/////////////////////////////////

	auto time_begin = std::chrono::high_resolution_clock::now();

	double cd_tol = 0.05;//0.0001;
	double fX = 1e+15;
	double fXprev = 1e+15;
	double l1normX_psi = 0;
	double l1normX_the = 0;
	double trSX_psi = 0;
	double trSX_the = 0;

	// compute the objective function (diagonal initialization assumed)
	for (int i = 0; i < q; i++) {
		for (int j = 0; j < q; j++) {
			unsigned long ij = i * q + j;
			trSX_psi += Psi[ij] * S_psi[ij];
			if (i != j)
				l1normX_psi += fabs(Psi[ij]);
		}
	}
	l1normX_psi *= gamma_psi;

	for (int i = 0; i < p; i++) {
		for (int j = 0; j < p; j++) {
			unsigned long ij = i * p + j;
			trSX_the += Theta[ij] * S_the[ij];
			if (i != j)
				l1normX_the += fabs(Theta[ij]);
		}
	}
	l1normX_the *= gamma_the;

	double f_psi = trSX_psi + l1normX_psi;
	double f_the = trSX_the + l1normX_the;
	double logdet = computeLogdet(p, q, eigval_the, eigval_psi);

	fX = p * f_psi + q * f_the - logdet;

	double* D_psi = (double*)calloc((q * q), sizeof(double));
	double* D_the = (double*)calloc((p * p), sizeof(double));

	double alpha = 0;
	double c = 0;

	/////////////////////////
	// Newton iterations
	/////////////////////////

	double** Uk_the = (double**)malloc((K_EIG) * sizeof(double*));
	double** Uk_psi = (double**)malloc((K_EIG) * sizeof(double*));
	for (int i = 0; i < K_EIG; i++) {
		Uk_the[i] = (double*)malloc((p * p) * sizeof(double));
		Uk_psi[i] = (double*)malloc((q * q) * sizeof(double));
	}

	for (int t_Newton = 1; t_Newton < max_Newton_iter; t_Newton++) {

		auto time_newton_begin = std::chrono::high_resolution_clock::now();

		// THETA
		std::cout << "Iter " << t_Newton << std::flush;

		memset(D_the, 0, (p * p) * sizeof(double));
		memset(D_psi, 0, (q * q) * sizeof(double));

		for (int i = 0; i < K_EIG; i++) {
			memset(Uk_the[i], 0, (p * p) * sizeof(double));
		}
		for (int i = 0; i < K_EIG; i++) {
			memset(Uk_psi[i], 0, (q * q) * sizeof(double));
		}

		computeNewtonJoint(t_Newton, p, q, S_the, S_psi, 
			Theta, Psi, W_the, W_psi,
			K_EIG, Ak_the, Ak_psi, Uk_the, Uk_psi,
			eigvec_the, eigvec_psi,
			D_the, D_psi, gamma_the, gamma_psi, cd_tol, alpha, tr_ratio);

		alpha = lineSearchJoint(p, q, max_line_iter, S_the, S_psi, 
			D_the, D_psi, Theta, Psi, W_the, W_psi,
			gamma_the, gamma_psi, sigma, 
			fX, fXprev, logdet, l1normX_the, l1normX_psi,
			trSX_the, trSX_psi, eigval_the, eigval_psi);

		if (alpha == -1) {
			alpha = 0;
			std::cout << "Converged. ";
			break;
		}

		for (int i = 0; i < p; i++) {
			for (int j = 0; j < i; j++) {
				unsigned long ij = i * p + j;
				Theta[ij] += alpha * D_the[ij];
			}
		}
		for (int i = 0; i < p; i++) {
			unsigned long ii = i * p + i;
			Theta[ii] += alpha * D_the[ii];
		}

		for (int i = 0; i < q; i++) {
			for (int j = 0; j < i; j++) {
				unsigned long ij = i * q + j;
				Psi[ij] += alpha * D_psi[ij];
			}
		}
		for (int i = 0; i < q; i++) {
			unsigned long ii = i * q + i;
			Psi[ii] += alpha * D_psi[ii];
		}

		for (int i = 0; i < p; i++) {
			for (int j = 0; j <= i; j++) {
				unsigned long ij = i * p + j;
				eigvec_the[ij] = Theta[ij];
			}
		}
		LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', p, eigvec_the, p, eigval_the);

		for (int i = 0; i < q; i++) {
			for (int j = 0; j <= i; j++) {
				unsigned long ij = i * q + j;
				eigvec_psi[ij] = Psi[ij];
			}
		}
		LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', q, eigvec_psi, q, eigval_psi);

		distributeEigval(K_EIG, p, q, eigval_the, eigval_psi, eigval_W_the, eigval_W_psi, eigval_Ak_the, eigval_Ak_psi);
		computeWAk(q, eigvec_psi, eigval_W_psi, W_psi, K_EIG, eigval_Ak_psi, Ak_psi);
		computeWAk(p, eigvec_the, eigval_W_the, W_the, K_EIG, eigval_Ak_the, Ak_the);

		double convg = fabs((fX - fXprev) / fX);

		auto time_newton_end = std::chrono::high_resolution_clock::now();
		double time_this_iter = std::chrono::duration<double>(time_newton_end - time_newton_begin).count();

		iter_times.push_back(time_this_iter);
		fs.push_back(fX);
		convgs.push_back(convg);

		//////////////////////// 
		// Check convergence
		////////////////////////

		std::cout << std::scientific << "    fX: " << fX << "    convg: " << convg << std::endl << std::flush;


		if (convg < newton_tol || convg == 0) {
			std::cout << "Converged." << std::endl;
			break;
		}
		fXprev = fX;
	}

	auto time_end = std::chrono::high_resolution_clock::now();
	double time_elapsed = std::chrono::duration<double>(time_end - time_begin).count();
	std::cout << "Elapsed time is " << time_elapsed << " sec." << std::endl;

	/////////////////////
	// Outputs
	/////////////////////

	c = adjustDiag(p, q, Theta, Psi, tr_ratio);
	for (int i = 0; i < p; i++) {
		Theta[i * p + i] += c;
	}
	for (int i = 0; i < q; i++) {
		Psi[i * q + i] -= c;
	}

	for (int i = 0; i < q; i++) {
		for (int j = 0; j < q; j++) {
			if (j > i)
				output_psi << Psi[j * q + i] << " ";
			else
				output_psi << Psi[i * q + j] << " ";
		}
		output_psi << "\n";
	}
	for (int i = 0; i < p; i++) {
		for (int j = 0; j < p; j++) {
			if (j > i)
				output_the << Theta[j * p + i] << " ";
			else
				output_the << Theta[i * p + j] << " ";
		}
		output_the << "\n";
	}
	output_psi.close();
	output_the.close();

	std::vector<double>::iterator i;
	for (i = iter_times.begin(); i != iter_times.end(); i++) {
		output_info << *i << " ";
	}
	output_info << "\n";
	for (i = fs.begin(); i != fs.end(); i++) {
		output_info << *i << " ";
	}
	output_info << "\n";
	output_info.close();

	free(Psi);
	free(Theta);
	free(S_psi);
	free(S_the);
	free(D_psi);
	free(D_the);
	free(W_psi);
	free(W_the);
	for (int i = 0; i < K_EIG; i++) {
		free(Ak_the[i]);
		free(Ak_psi[i]);
		free(Uk_the[i]);
		free(Uk_psi[i]);
	}
	free(Ak_psi);
	free(Ak_the);
	free(Uk_psi);
	free(Uk_the);
	free(eigval_psi);
	free(eigval_the);
	free(eigvec_psi);
	free(eigvec_the);
	free(eigval_Ak_psi);
	free(eigval_Ak_the);

	return 0;
}
