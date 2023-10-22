// matrix: row-major
// symmetric matrix: lower triangular

#include <iostream>
#include <iomanip>
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

static inline double computeNewton(int t_outer, int dim, int dim_other, const double* S, const double* X, const double* W, const int num_eig, double** Ak, double** Uk, double* D, const double gamma, const double tol)
{
	srand(25252);

	double subgrad = 0;
	unsigned int size_active = 0;
	uint_pair* activeSet = (uint_pair*)calloc((dim * (dim + 1) / 2), sizeof(uint_pair));

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j <= i; j++) {
			int ij = i * dim + j;

			double g = dim_other * S[ij] - W[ij];

			if (X[ij] != 0.0 || fabs(g) > dim_other* gamma) {
				activeSet[size_active].i = (unsigned int)i;
				activeSet[size_active].j = (unsigned int)j;
				size_active++;
				if (X[ij] > 0)
					g += gamma;
				else if (X[ij] < 0)
					g -= gamma;
				else
					g = fabs(g) / dim_other - gamma;
				subgrad += fabs(g);
			}
		}
	}

	// compute Newton direction
	double diffD = 0;
	double normD = 0;
	//for (int t_cd = 0; t_cd <= 1 + t_outer / 3; t_cd++) {
  for (int t_cd = 0; t_cd <= 100; t_cd++) {
		diffD = 0;

		// permutation
		for (unsigned int ii = 0; ii < size_active; ii++) {
			unsigned int jj = ii + rand() % (size_active - ii);
			unsigned int k1 = activeSet[ii].i;
			unsigned int k2 = activeSet[ii].j;
			activeSet[ii].i = activeSet[jj].i;
			activeSet[ii].j = activeSet[jj].j;
			activeSet[jj].i = k1;
			activeSet[jj].j = k2;
		}


		// cd
		for (unsigned int l = 0; l < size_active; l++) {

			unsigned int i = activeSet[l].i;
			unsigned int j = activeSet[l].j;

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
				for (int r = 0; r < dim; r++) {
					b += Ak[k][idim + r] * Uk[k][r * dim + j];
				}
			}
			if (num_eig < dim_other) {
				for (int r = 0; r < dim; r++)
					b += (dim_other - num_eig) * Ak[num_eig-1][idim + r] * Uk[num_eig-1][r * dim + j];
			}

			double c = X[ij] + D[ij];
			double ll = dim_other * gamma / a;
			double f = b / a;
			double mu = 0;
			normD -= fabs(D[ij]);
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
			diffD += fabs(mu);
			normD += fabs(D[ij]);

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
	free(activeSet);
	return subgrad;
}

static inline double lineSearch(const int dim, const int dim_other, const int max_iter, const double* S, double* D, const double* X, double* W,
	const double gamma, const double sigma, double& fX, double& fXprev, double& logdet, double& l1normX, double& l1normX_other,
	double& trSX, double& trSX_other, double* eigval, double* eigval_other)
{
	double trgradgD = 0;
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < i; j++) {
			unsigned long ij = i * dim + j;
			trgradgD += (dim_other * S[ij] - W[ij]) * D[ij];
		}
	}
	trgradgD *= 2;
	for (int i = 0; i < dim; i++) {
		unsigned long ii = i * dim + i;
		trgradgD += (dim_other * S[ii] - W[ii]) * D[ii];
	}


	double alpha = 1;
	double beta = 0.5;
	double l1normXD = 0;

	double l1normX1 = 0;
	double trSX1 = 0;

	int alpha_chosen = 0;

	for (int t_line = 0; t_line < max_iter; t_line++) {

		l1normX1 = 0;
		trSX1 = 0;

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < i; j++) {
				unsigned long ij = i * dim + j;
				W[ij] = X[ij] + alpha * D[ij]; // store theta + alpha*D
				l1normX1 += fabs(W[ij]);
				trSX1 += W[ij] * S[ij];
			}
		}
		l1normX1 *= 2 * gamma; // diagonals are not regularized
		trSX1 *= 2;
		for (int i = 0; i < dim; i++) {
			unsigned long ii = i * dim + i;
			W[ii] = X[ii] + alpha * D[ii]; // store theta + alpha*D
			//W[ii] += 1e-8;
			trSX1 += W[ii] * S[ii];
		}

		if (alpha == 1.0) {
			l1normXD = l1normX1 * dim_other + dim * l1normX_other;;
		}

		lapack_int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'N', 'L', dim, W, dim, eigval);
		if (info != 0) {
			//if (MSG >= MSG_MIN) {
				//std::cout << t_line << ">> Line search: illegal argument for dsyevd (" << info << "). alpha=" << alpha << std::endl;
			//}
      
			alpha = 0;
			break;
		}

		if (eigval[0] + eigval_other[0] <= 0) {
			//if (MSG >= MSG_MIN) {
				//std::cout << t_line << ">> Line search: OMEGA not positive definite (" << eigval[0] << ", " << eigval_other[0] << "). alpha=" << alpha << std::endl;
			//}
			alpha *= beta;
			continue;
		}


		double logdetX1 = 0;
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim_other; j++) {
				logdetX1 += log(eigval[i] + eigval_other[j]);
			}
		}

		double l1normX1_joint = dim_other * l1normX1 + dim * l1normX_other;
		double l1normX_joint = dim_other * l1normX + dim * l1normX_other;
		double fX1 = dim_other * trSX1 + dim * trSX_other - logdetX1 + l1normX1_joint;


		if (fX1 <= fX + alpha * sigma * (trgradgD + l1normXD - l1normX_joint)) {

			fX = fX1;
			l1normX = l1normX1;
			logdet = logdetX1;
			trSX = trSX1;
			alpha_chosen = 1;
			break;
		}



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

int main(int argc, char** argv) {


  std::vector<double> iter_times;
	std::vector<double> fs;
	std::vector<double> convgs;

  ///////////////////////////
	// Command Line Arguments
	///////////////////////////

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
    /*
	for (int i = 0; i < argc-1; i++) {
		if (strcmp(argv[i],"--reg_theta") == 0) {
      gamma_the = atof(argv[i+1]);
			break;
		}
	}
     */

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
    
    double tr_ratio = 0;
    for (int i = 0; i < argc-1; i++) {
        if (strcmp(argv[i],"--trratio") == 0) {
                tr_ratio = atof(argv[i+1]);
                break;
        }
    }

	
  std::cout << std::endl;
  std::cout << "#############################" << std::endl;
  std::cout << "#  EiGLasso Flip-flop v1.0  #" << std::endl;
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

	/////////////////////////////////
	// Eigendecomposition
	/////////////////////////////////

	double* eigval_psi = (double*)malloc(q * sizeof(double));
	double* eigval_the = (double*)malloc(p * sizeof(double));
	std::fill_n(eigval_psi, q, 1.0);
	std::fill_n(eigval_the, p, 1.0);

	double* eigvec_psi = (double*)calloc((q * q), sizeof(double));
	double* eigvec_the = (double*)calloc((p * p), sizeof(double));
	
	for (int i = 0; i < q; i++)
	  eigvec_psi[i * q + i] = 1;
     
	for (int i = 0; i < p; i++)
		eigvec_the[i * p + i] = 1;
	

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

	double cd_tol = 0.05;
	double fX = 1e+15;
	double fXprev = 1e+15;
	double l1normX_psi = 0;
	double l1normX_the = 0;
	double trSX_psi = 0;
	double trSX_the = 0;



	// compute the objective function
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

	double** Uk_the = (double**)malloc((K_EIG ) * sizeof(double*));
	double** Uk_psi = (double**)malloc((K_EIG ) * sizeof(double*));
	for (int i = 0; i < K_EIG; i++) {
		Uk_the[i] = (double*)malloc((p * p) * sizeof(double));
		Uk_psi[i] = (double*)malloc((q * q) * sizeof(double));
	}


	for (int t_Newton = 1; t_Newton < max_Newton_iter; t_Newton++) {

		auto time_newton_begin = std::chrono::high_resolution_clock::now();

		// THETA

		std::cout << "Iter " << std::setw(4) << t_Newton << ": Updating Theta, ";

		memset(D_the, 0, (p * p) * sizeof(double));
		for (int i = 0; i < K_EIG; i++) {
			memset(Uk_the[i], 0, (p * p) * sizeof(double));
		}


		computeNewton(t_Newton, p, q, S_the, Theta, W_the, K_EIG, Ak_the, Uk_the, D_the, gamma_the, cd_tol);

		alpha = lineSearch(p, q, max_line_iter, S_the, D_the, Theta, W_the,
			gamma_the, sigma, fX, fXprev, logdet, l1normX_the, l1normX_psi, trSX_the, trSX_psi, eigval_the, eigval_psi);


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

		for (int i = 0; i < p; i++) {
			for (int j = 0; j <= i; j++) {
				unsigned long ij = i * p + j;
				eigvec_the[ij] = Theta[ij];
				//if (i == j)
					//eigvec_the[i * p + i] += 1e-8;
			}
		}
   
    
   
		LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', p, eigvec_the, p, eigval_the);

		distributeEigval(K_EIG, p, q, eigval_the, eigval_psi, eigval_W_the, eigval_W_psi, eigval_Ak_the, eigval_Ak_psi);
		computeWAk(q, eigvec_psi, eigval_W_psi, W_psi, K_EIG, eigval_Ak_psi, Ak_psi);


		// PSI

    std::cout << "Psi. ";
		

		memset(D_psi, 0, (q * q) * sizeof(double));
		for (int i = 0; i < K_EIG; i++) {
			memset(Uk_psi[i], 0, (q * q) * sizeof(double));
		}

		computeNewton(t_Newton, q, p, S_psi, Psi, W_psi, K_EIG, Ak_psi, Uk_psi, D_psi, gamma_psi, cd_tol);


		alpha = lineSearch(q, p, max_line_iter, S_psi, D_psi, Psi, W_psi,
			gamma_psi, sigma, fX, fXprev, logdet, l1normX_psi, l1normX_the, trSX_psi, trSX_the, eigval_psi, eigval_the);


		if (alpha == -1) {
			alpha = 0;
			std::cout << "Converged. ";
			break;
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

		for (int i = 0; i < q; i++) {
			for (int j = 0; j <= i; j++) {
				unsigned long ij = i * q + j;
				eigvec_psi[ij] = Psi[ij];
				//if (i == j)
					//eigvec_psi[i * q + i] += 1e-8;
			}
		}

   
		LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', q, eigvec_psi, q, eigval_psi);

		distributeEigval(K_EIG, p, q, eigval_the, eigval_psi, eigval_W_the, eigval_W_psi, eigval_Ak_the, eigval_Ak_psi);
		computeWAk(p, eigvec_the, eigval_W_the, W_the, K_EIG, eigval_Ak_the, Ak_the);

		double convg = fabs((fX - fXprev) / fX);

		
		std::cout << "fX=" << std::scientific << fX << std::endl;
		//std::cout << "  fXprev     = " << std::scientific << fXprev << std::endl;
		//std::cout << "  convg      = " << std::scientific << convg << std::endl;

		auto time_newton_end = std::chrono::high_resolution_clock::now();
		double time_this_iter = std::chrono::duration<double>(time_newton_end - time_newton_begin).count();

		iter_times.push_back(time_this_iter);
		fs.push_back(fX);
		convgs.push_back(convg);


		//////////////////////// 
		// Check convergence
		////////////////////////
   

    


		if (convgs[convgs.size() - 1] < 1e-8 || (convgs[convgs.size() - 1] < newton_tol && convgs[convgs.size() - 2] < newton_tol && convgs[convgs.size() - 3] < newton_tol)) {
		  std::cout << "Converged. ";
			break;
		}
		fXprev = fX;


	}
	auto time_end = std::chrono::high_resolution_clock::now();
	double time_elapsed = std::chrono::duration<double>(time_end - time_begin).count();
	std::cout << "Elapsed time is " << std::fixed << time_elapsed << " sec." << std::endl;

	/////////////////////
	// Outputs
	/////////////////////

    if (tr_ratio > 0) {
        c = adjustDiag(p, q, Theta, Psi, tr_ratio);
        for (int i = 0; i < p; i++) {
            Theta[i * p + i] += c;
        }
        for (int i = 0; i < q; i++) {
            Psi[i * q + i] -= c;
        }
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
