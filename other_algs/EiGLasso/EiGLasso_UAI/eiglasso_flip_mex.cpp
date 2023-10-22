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

#include "lapack.h"
//#include "blas.h"
//#include "mkl.h"
#include "mex.h"


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

        ptrdiff_t info;
        ptrdiff_t dimt = (ptrdiff_t) dim;
        
        ptrdiff_t lwork = 3*dimt - 1;
        double *work = (double*)malloc(lwork * sizeof(double));
        
        
		dsyev("N", "U", &dimt, W, &dimt, eigval, work, &lwork, &info);
        free(work);
		if (info != 0) {
			//if (MSG >= MSG_MIN) {
				//std::cout << t_line << ">> Line search: illegal argument for dsyevd (" << info << "). alpha=" << alpha << std::endl;
                //mexPrintf("%d illegal %d. alpha=%e\n", t_line, info, alpha);
			//}
      
			alpha = 0;
			break;
		}

		if (eigval[0] + eigval_other[0] <= 0) {
			//if (MSG >= MSG_MIN) {
				//std::cout << t_line << ">> Line search: OMEGA not positive definite (" << eigval[0] << ", " << eigval_other[0] << "). alpha=" << alpha << std::endl;
                //mexPrintf("%d Omega non-pd. %4e, %4e. alpha=%e\n", t_line, eigval[0], eigval_other[0], alpha);
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
    const ptrdiff_t dimt = (const ptrdiff_t) dim;
    const double a = 1;
    const double c = 0;
	// Ak's
	for (int k = 0; k < num_eig; k++) {
        memset(Ak[k], 0, (dim * dim) * sizeof(double));
		memset(tmpQD, 0, (dim * dim) * sizeof(double));
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				tmpQD[i * dim + j] = eigval_Ak[k * dim + i] * eigvec[i * dim + j];
			}
		}
		//dgemm("N", "T", &dimt, &dimt, &dimt, &a, tmpQD, &dimt, eigvec, &dimt, &c, Ak[k], &dimt);
        for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
                for (int r = 0; r < dim; r++) {
                    Ak[k][i * dim + j] += tmpQD[r * dim + i] * eigvec[r * dim + j];
                }
			}
		}
	}
	memset(tmpQD, 0, (dim * dim) * sizeof(double));
    memset(W, 0, (dim * dim) * sizeof(double));
	// W
	for (int j = 0; j < dim; j++) {
		for (int i = 0; i < dim; i++) {
			tmpQD[i * dim + j] = eigval_W[i] * eigvec[i * dim + j];
		}
	}
	//dgemm("N", "T", &dimt, &dimt, &dimt, &a, tmpQD, &dimt, eigvec, &dimt, &c, W, &dimt);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int r = 0; r < dim; r++) {
                W[i * dim + j] += tmpQD[r * dim + i] * eigvec[r * dim + j];
            }
        }
    }
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {


  std::vector<double> iter_times;
	std::vector<double> fs;
	std::vector<double> convgs;

  ///////////////////////////
	// Command Line Arguments
	///////////////////////////

  
  
	
	double* S_the = mxGetPr(prhs[0]);
    double* S_psi = mxGetPr(prhs[1]);
    
    int p = mxGetM(prhs[0]);
    int q = mxGetM(prhs[1]);
    
 
    double gamma_the = mxGetScalar(prhs[2]);
    double gamma_psi = mxGetScalar(prhs[3]);
  
  int K_EIG = 1;
  int max_Newton_iter = 10000;
	int max_line_iter = 20;
  double newton_tol = 1e-3;
	double sigma = 0.01;
    double tr_ratio = 0;
  
  if (nrhs > 4) {
  	K_EIG = mxGetScalar(prhs[4]);
    max_Newton_iter = mxGetScalar(prhs[5]) + 1;
  	max_line_iter = mxGetScalar(prhs[6]) + 1;
    newton_tol = mxGetScalar(prhs[7]);
  	sigma = mxGetScalar(prhs[8]);
      tr_ratio = mxGetScalar(prhs[9]);
  }
  
  plhs[0] = mxCreateDoubleMatrix(p, p, mxREAL); // output theta
  plhs[1] = mxCreateDoubleMatrix(q, q, mxREAL); // output psi
  
    double* Theta = mxGetPr(plhs[0]);
    double* Psi = mxGetPr(plhs[1]);
	
  for (int i = 0; i < q; i++)
			Psi[i * q + i] = 1;
		
	for (int i = 0; i < p; i++)
			Theta[i * p + i] = 1;
	
    mexPrintf("\n");
    mexPrintf("#############################\n");
    mexPrintf("#  EiGLasso Flip-flop v1.0  #\n");
    mexPrintf("#############################\n");
    mexPrintf("\n");
    mexEvalString("drawnow");

	

	



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

    double flush_time = 0;
    
    
    ptrdiff_t info;
    ptrdiff_t dimt_the = (ptrdiff_t) p;
    ptrdiff_t dimt_psi = (ptrdiff_t) q;
    
    ptrdiff_t lwork_the = 3*dimt_the - 1;
    ptrdiff_t lwork_psi = 3*dimt_psi - 1;
    
    double *work_the = (double*)malloc(lwork_the * sizeof(double));
    double *work_psi = (double*)malloc(lwork_psi * sizeof(double));
    
	for (int t_Newton = 1; t_Newton < max_Newton_iter; t_Newton++) {
        
        memset(work_the, 0, lwork_the * sizeof(double));
        memset(work_psi, 0, lwork_psi * sizeof(double));

		auto time_newton_begin = std::chrono::high_resolution_clock::now();

		// THETA
        mexPrintf("Iter %4d: Updating Theta, ", t_Newton);

		memset(D_the, 0, (p * p) * sizeof(double));
		for (int i = 0; i < K_EIG; i++) {
			memset(Uk_the[i], 0, (p * p) * sizeof(double));
		}


		computeNewton(t_Newton, p, q, S_the, Theta, W_the, K_EIG, Ak_the, Uk_the, D_the, gamma_the, cd_tol);

		alpha = lineSearch(p, q, max_line_iter, S_the, D_the, Theta, W_the,
			gamma_the, sigma, fX, fXprev, logdet, l1normX_the, l1normX_psi, trSX_the, trSX_psi, eigval_the, eigval_psi);


		if (alpha == -1) {
			alpha = 0;
            mexPrintf("Converged. ");
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
   
    
   
        
        
        
        
        dsyev("V", "U", &dimt_the, eigvec_the, &dimt_the, eigval_the, work_the, &lwork_the, &info);

		distributeEigval(K_EIG, p, q, eigval_the, eigval_psi, eigval_W_the, eigval_W_psi, eigval_Ak_the, eigval_Ak_psi);
		computeWAk(q, eigvec_psi, eigval_W_psi, W_psi, K_EIG, eigval_Ak_psi, Ak_psi);


		// PSI
        mexPrintf("Psi. ");
		
		memset(D_psi, 0, (q * q) * sizeof(double));
		for (int i = 0; i < K_EIG; i++) {
			memset(Uk_psi[i], 0, (q * q) * sizeof(double));
		}

		computeNewton(t_Newton, q, p, S_psi, Psi, W_psi, K_EIG, Ak_psi, Uk_psi, D_psi, gamma_psi, cd_tol);


		alpha = lineSearch(q, p, max_line_iter, S_psi, D_psi, Psi, W_psi,
			gamma_psi, sigma, fX, fXprev, logdet, l1normX_psi, l1normX_the, trSX_psi, trSX_the, eigval_psi, eigval_the);


		if (alpha == -1) {
			alpha = 0;
            mexPrintf("Converged. ");
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

   
        dsyev("V", "U", &dimt_psi, eigvec_psi, &dimt_psi, eigval_psi, work_psi, &lwork_psi, &info);

		distributeEigval(K_EIG, p, q, eigval_the, eigval_psi, eigval_W_the, eigval_W_psi, eigval_Ak_the, eigval_Ak_psi);
		computeWAk(p, eigvec_the, eigval_W_the, W_the, K_EIG, eigval_Ak_the, Ak_the);

		double convg = fabs((fX - fXprev) / fX);
        mexPrintf("fX=%e\n",fX);

		auto time_newton_end = std::chrono::high_resolution_clock::now();
		double time_this_iter = std::chrono::duration<double>(time_newton_end - time_newton_begin).count();

		iter_times.push_back(time_this_iter);
		fs.push_back(fX);
		convgs.push_back(convg);

        flush_time += time_this_iter;
        if (flush_time > 1) { // flush every second
            mexEvalString("drawnow");
            flush_time = 0;
        }

        
        
		//////////////////////// 
		// Check convergence
		////////////////////////

		if (convgs[convgs.size() - 1] < 1e-8 || (convgs[convgs.size() - 1] < newton_tol && convgs[convgs.size() - 2] < newton_tol && convgs[convgs.size() - 3] < newton_tol)) {
            mexPrintf("Converged. ");
			break;
		}
		fXprev = fX;


	}
	auto time_end = std::chrono::high_resolution_clock::now();
	double time_elapsed = std::chrono::duration<double>(time_end - time_begin).count();
    mexPrintf("Elapsed time is %f sec.\n",time_elapsed);

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
    
    plhs[2] = mxCreateDoubleMatrix(iter_times.size(), 1, mxREAL); // output iteration runtime
    plhs[3] = mxCreateDoubleMatrix(fs.size(), 1, mxREAL); // output objective values
    
    double* out_iter_times = mxGetPr(plhs[2]);
    double* out_fs = mxGetPr(plhs[3]);
    
    std::copy(iter_times.begin(), iter_times.end(), out_iter_times);
    std::copy(fs.begin(), fs.end(), out_fs);



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
    
    free(work_the);
    free(work_psi);
}
