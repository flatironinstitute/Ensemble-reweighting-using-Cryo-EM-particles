functions {
  real log_likelihood(array[,] real D_slice, int start, int end, vector normed_log_weights_logNm) {
    real total = 0;
    for (i in 1:end-start+1) { // for each row (each image) in a slice of D
      total += log_sum_exp(to_vector(D_slice[i]) + normed_log_weights_logNm); // log sum exp over centers M, add to log_posterior
    }
    return total;
  }
}
data {
  int<lower=0> M; // number of centers M
  int<lower=0> N; // number of images N
  vector[M] logNm; // log number of cluster member Nm
  array[N, M] real Dmat; // -.5 / sigma**2 * structure-image-distance-matrix
}
transformed data {
  int grain_size = 1;
  vector[M] ones = rep_vector(1, M);
}
parameters {
  simplex[M] weights;
}
transformed parameters {
  vector[M] log_weights = log(weights);
}
model {
  weights ~ dirichlet(ones);
  vector[M] normed_log_weights_logNm = log_weights - log_sum_exp(log_weights + logNm) + logNm;
  target += reduce_sum(log_likelihood, Dmat, grain_size, normed_log_weights_logNm);
}