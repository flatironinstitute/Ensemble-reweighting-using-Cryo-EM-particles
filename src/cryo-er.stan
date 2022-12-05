functions {
  real partial_sum(array[,] real d_slice, int start, int end, vector normed_log_weights, int M, vector logNm){
    // just a function that adapt the posterior calculation to parallelization
    real total = 0;
    vector[M] gamma;
    for (i in 1:end-start+1) { // for each row (each image) in a slice of D
      for (m in 1:M){
          gamma[m] = d_slice[i][m] + normed_log_weights[m] + logNm[m]; // equivalent to likelihood(w, x) * weights
      }
      total += log_sum_exp(gamma); // log sum exp over centers M, add to log_posterior
    }
    return total;
  }
}
data {
  int<lower=0> M; // number of centers M
  int<lower=0> N; // number of images N
  vector[M] logNm; // number of cluster member Nm
  array[N, M] real Dmat; // -(.5/sigma**2)*[structure-image distance matrix]
}
parameters {
  simplex[M] weights;
}
transformed parameters {
  vector[M] log_weights = log(weights);
  vector[M] normed_log_weights = log_weights - log_sum_exp(log_weights + logNm);
}
model {
  weights ~ dirichlet(rep_vector(1.0, M));  // draw from dirichlet prior with Dirchilet parameter a = 1.0 for all dimension
  
  // Parallelization for calculating log posterior
  int grainsize = 1; 
  target += reduce_sum(partial_sum, Dmat, grainsize, normed_log_weights, M, logNm);
}
