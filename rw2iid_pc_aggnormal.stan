functions {
  // 1st order IGMRF
  real rw1_normal_lpdf(vector phi, int N) {
    return -0.5 * dot_self(phi[1:(N-1)] - phi[2:N])
      + normal_lpdf(sum(phi) | 0, 0.001 * N);  // soft mean-zero constraint
  }
  
  // 2nd order IGMRF
  real rw2_normal_lpdf(vector phi, int N) {
    real iphi = 0;
    
    for(i in 1:N)
      iphi += i*phi[i];
    
    return -0.5 * dot_self(phi[1:(N-2)] - 2*phi[2:(N-1)] + phi[3:N])
      + normal_lpdf(sum(phi) | 0, 0.001 * N)  // soft mean-zero constraint
      + normal_lpdf(iphi | 0, 0.001 * N);  // soft 2nd-order constraint
  }
  
  // aggregated normal likelihood
  real aggregate_normal_lpdf(vector smean, vector mu, real logsigma, vector svar, vector n) {
    return sum(-n .* ( exp(log(square(smean - mu) + svar) - log(2) - 2*logsigma) + logsigma ))
      + sum(exp(log(svar) - log(2) - 2*logsigma));
  }
}
data {
  int<lower=1> N_data;  // number of observations
  
  vector[N_data] x;
  vector[N_data] y_m;           // sample mean of y
  vector<lower=0>[N_data] y_v;  // sample variance of y
  vector<lower=1>[N_data] y_n;  // number of observations of y
  
  int<lower=4> N_x;  // number of distinct x values
  int<lower=1,upper=N_x> x_id[N_data];
  
  vector<lower=0,upper=1>[N_data] grade9;
  
  int<lower=1,upper=2> igmrf_order;  // 1st or 2nd order IGMRF
  real<lower=0> scaling_factor;      // standardizes IGMRF
  int<lower=1,upper=N_x> igmrf_pc;   // index of point constraint
  
  int<lower=0,upper=1> prior_only;
}
parameters {
  // linear model components
  real Intercept;
  real b_x;
  real b_g9;  // grade9 effect (size of jump discontinuity)
  real<lower=0> log_sigma;  // log(residual sd)
  
  vector[N_x-1] igmrf_free;  // unconstrained IGMRF components
  vector[N_x] hetero;        // heterogeneous effects
  
  real<lower=0, upper=1> rho;  // proportion of total variance coming from IGMRF
  real<lower=0> re_scale;      // scale of convolved effects
}
transformed parameters {
  // integrated Gaussian Markov random field
  vector[N_x] igmrf;
  
  // varying effects (convolution of IGMRF & heterogeneous effects)
  vector[N_x] convolved_re;
  
  // linear model
  vector[N_data] mu;
  
  // apply point constraint
  for(i in 1:N_x) {
    if(i == igmrf_pc)
      igmrf[i] = 0;
    else if(i < igmrf_pc)
      igmrf[i] = igmrf_free[i];
    else
      igmrf[i] = igmrf_free[i-1];
  }
  
  // convolve igmrf and heterogeneous effects
  convolved_re = (sqrt(rho/scaling_factor)*igmrf + sqrt(1-rho)*hetero) * re_scale;
  
  // construct linear model
  mu = Intercept + b_g9*grade9 + b_x*x + convolved_re[x_id];
}
model {
  // priors for linear model components
  Intercept ~ student_t(3, 152, 5);
  b_x ~ normal(0, 2);
  b_g9 ~ normal(0, 3);
  log_sigma ~ student_t(3, 2.3, 1);
  
  // priors for spatial/heterogeneous random effects
  if(igmrf_order == 1)
    igmrf ~ rw1_normal(N_x);
  else
    igmrf ~ rw2_normal(N_x);
  
  hetero ~ std_normal();
  rho ~ beta(0.5, 0.5);
  re_scale ~ exponential(1);
  
  // likelihood
  if(!prior_only)
    y_m ~ aggregate_normal(mu, log_sigma, y_v, y_n);
}
