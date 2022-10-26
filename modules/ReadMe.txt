#  Consistency diagnostics within the observation space 
#  Ref: Desroziers et al., 2005, Cressot et al., 2014, Chevallier et al., 2017
#  GPGt     = E( (H(xa) - H(xb) ) (y - H(xa))t ) 
#  GBGt     = E( (H(xa) − H (xb))(y − H (xb))t )
#  R        = E((y − H (xa))(y − H (xb))t) 
#  GBGt + R = E((y − H (xb))(y − H (xb))t)
      
#  On the left, assigned values and on the right, diagnosed values 
#  H  = observation operator
#  G  = the tangent linear of H
#  xa = optimized fluxes
#  xb = prior
#  y  = observations 
#  E  = moyenne temporelle

#  Arguments: 
#   * obs_vec: obs vector (y) and R
#   * sim_0 : H (xb)
#   * sim_opt : H(xa)
#   * sig_B: diagonal of the error covariance matrix B
#   * sigma_t: temporel correlation lengths
#   * Bpost: error posterior matrix
#   
#  Loads:
#   * G tangent linear

