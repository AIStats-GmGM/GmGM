# EiGLasso (flip-flop optimization)
This software implements EiGLasso with a flip-flop optimization introduced in the following paper:  
> Jun Ho Yoon and Seyoung Kim. EiGLasso: Scalable Estimation of Cartesian Product of Sparse Inverse Covariance Matrices. *Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence (UAI)*, PMLR volume 124, 2020.  
> [[paper](https://www.junhoyoon.com/files/507_main_paper.pdf), [talk](https://youtu.be/rteWAfivvpw)]  
  
EiGLasso can be used as either 

1. command-line application (tested on Linux, MacOS),
2. MATLAB function (tested on Linux, MacOS, Windows).



## 1. Command-line application (Linux, MacOS)

  ### 1.1 Prerequisite
  #### (MacOS Only)
Install Xcode or Xcode Command Line Tools.
  
  #### Intel Math Kernel Library (MKL) ####

1. Install oneAPI MKL ([link](https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html#onemkl))
2. Specify the directory where MKL is installed in [Makefile](Makefile) (e.g. MKLROOT=/home/junhoy/intel/mkl)
3. (optional) Use the link line advisor from Intel ([link](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html)) if EiGLasso does not compile correctly.


  ### 1.2 Installation

  Go to the directory where EiGLasso codes are and type "make" in the command line.

  ```bash
  make
  ```
  
  
  ### 1.3 Input & Output Files
  
  All matrices are space-separated text files. Example files are in [demo](demo/).
  
  #### Input ####
  * S.txt: empirical covariance for Theta (the left matrix in Kronecker sum)
  * T.txt: empirical covariance for Psi   (the right matrix in Kronecker sum)

  #### Output ####
  * theta.txt: estimated Theta
  * psi.txt  : estimated Psi
  * info.txt : two-line text file. The first line contains per-iteration runtime in second, and the second line has the objective value after each iteration.
  

  ### 1.4 Usage

#### 1.4.1 Demo run of EiGLasso on p=q=100 example data only with required flags

  ```bash
  ./eiglasso_flip -p 100 -q 100 -s demo/S.txt -t demo/T.txt -r 0.1 0.08
  ```
  
 #### Required flags ####
 -p: dimension of Theta, p-by-p matrix  
 -q: dimension of Psi, q-by-q matrix  
 -s: input for Theta  
 -t: input for Psi  
 -r: regularization hyperparameters in the order, Theta Psi. For example, 0.1 for Theta and 0.08 for Psi above.  
 
 
#### 1.4.2 Demo run of EiGLasso on p=q=100 example data with all available flags
 
  ```bash
  ./eiglasso_flip -p 100 -q 100 -s demo/S.txt -t demo/T.txt -r 0.1 0.08 -o tmp/ -k 1 --max-newton 10000 --max-line 20 --tol 0.001 --sigma 0.01 --trratio 1
  ```
  
  
 #### Optional flags ####
 -o: output prefix. With "-o tmp/", outputs will be saved in "tmp/psi.txt", "tmp/theta.txt", and "tmp/info.txt". *Default: None (current directory)*  
 -k: Hessian approximation degree (i.e. the number of eigenvalues from the other graph). *Default: 1*  
 --max-newton: maximum number of iterations for EiGLasso algorithm. *Default: 10000*  
 --max-line: maxinum number of iterations for line search algorithm. *Default: 20*  
 --tol: convergence tolerance. *Defulat: 0.001*  
 --sigma: line search hyperparameter. *Default: 0.01*  
 --trratio: trace ratio for diagonal identification adjustment. *Default: not adjust*  

  

## 2. MATLAB function (Linux, MacOS, Windows)

### 2.1 Installation

Open MATLAB. Go to the directory where EiGLasso codes are. Type the following in MATLAB:

  ```bash
  mex -output eiglasso_flip eiglasso_flip_mex.cpp -lmwlapack
  ```

If it does not compile on Windows, try

  ```bash
  mex -output eiglasso_flip eiglasso_flip_mex.cpp
  ```  
  or
  ```bash
  ipath = ['-I' fullfile(matlabroot,'extern','include')];
  lapacklib = fullfile(matlabroot,'extern','lib',computer('arch'),'microsoft','libmwlapack.lib');
  mex('-output','eiglasso_flip','eiglasso_flip_mex.cpp',ipath,lapacklib)
  ```
  
  
### 2.2 Usage

EiGLasso can be used as a MATLAB function.

  ```bash
  [Theta, Psi, ts, fs] = eiglasso_flip(S, T, reg_theta, reg_psi)
  [Theta, Psi, ts, fs] = eiglasso_flip(S, T, reg_theta, reg_psi, K, max_Newton, max_line, tol, sigma, tr_ratio)
  ```
  
  #### Input ####
  
  S: empirical covariance for Theta  
  T: empirical covariance for Psi  
  reg_theta, reg_psi: regularization hyperparameters  
  K: Hessian approximation degree (i.e. the number of eigenvalues from the other graph). *Default: 1*  
  tol: convergence tolerance. *Defulat: 0.001*  
  max_Newton: maximum number of iterations for EiGLasso algorithm. *Default: 10000*  
  max_line: maxinum number of iterations for line search algorithm. *Default: 20*  
  sigma: line search hyperparameter. *Default: 0.01*  
  tr_ratio: trace ratio for diagonal identification adjustment. *Default: not adjust*  
  
  #### Output ####
  
  Theta, Psi: upper-triangular estimated Theta and Psi  
  ts: per-iteration runtime in second  
  fs: objective value after each iteration


### 2.3 Example

Example data is available in [demo/example.mat](demo/example.mat).

  ```bash
  load('demo/example.mat');
  [Theta, Psi, ts, fs] = eiglasso_flip(S, T, 0.1, 0.1);
  [Theta, Psi, ts, fs] = eiglasso_flip(S, T, 0.1, 0.1, 1, 10000, 20, 0.001, 0.01, 1);
  ```

