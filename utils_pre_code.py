#!/usr/bin/python

# Standard modules
import numpy as np

# Custom modules
if __name__=='__main__':
    import read_earthsr_io as reo
else:
    import SW1D_earthsr.read_earthsr_io as reo

##########################################################################################################################

class model_1D:

	def __init__(self, mod1d):
		self.oreo = reo.read_modfile([mod1d])
		# print self.oreo.struc[0], len(self.oreo.struc[0])

		vp = self.oreo.alpha
		vs = self.oreo.beta
		rho = self.oreo.rho

		uvals_vp, ind_vp = np.unique(vp, return_index=True)
		uvals_vs, ind_vs = np.unique(vs, return_index=True)
		uvals_rho, ind_rho = np.unique(rho, return_index=True)

		""" NB: numpy.unique returns SORTED unique VALUES by default. This can mess things up if
		parameter values are not all increasing with depth (e.g. low velocity layer at depth). Hence
		it is important to SORT the INDICES obtained above. """
		# uvals_vp=[b for a,b in sorted(zip(ind_vp,uvals_vp))]
		# uvals_vs=[b for a,b in sorted(zip(ind_vs,uvals_vs))]
		# uvals_rho=[b for a,b in sorted(zip(ind_rho,uvals_rho))]

		vp_ind = sorted(ind_vp)
		vs_ind = sorted(ind_vs)
		rho_ind = sorted(ind_rho)
		v_ind= np.union1d(vp_ind, vs_ind)
		hif_ind = np.union1d(v_ind, rho_ind)

		self.mod_hif = self.oreo.deps[hif_ind][1:]
		# first element of the array is ignored because it corresponds to the surface; z=0

		self.deps_all = self.oreo.deps[:-1]

	#***************************************************************************

	def fix_max_depth(self, indepth):

		max_ind = np.searchsorted(self.deps_all,indepth) + 1

		self.deps_tomax = self.deps_all[:max_ind]
		self.rho_tomax = self.oreo.rho[:max_ind]
		alpha_tomax = self.oreo.alpha[:max_ind]
		beta_tomax = self.oreo.beta[:max_ind]

		self.mu_tomax = self.rho_tomax * np.square(beta_tomax)
		self.lamda_tomax = self.rho_tomax * ( np.square(alpha_tomax) - (2*np.square(beta_tomax)) )

####################################################################################################################
