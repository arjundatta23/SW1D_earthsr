#!/usr/bin/python

import itertools
import numpy as np
import scipy.integrate as spi

# Modules written by me
if __name__=='__main__':
    import read_earthsr_io as reo
else:
    import SW1D_earthsr.read_earthsr_io as reo

"""
To ascertain the quality of the eigenfunction integrals computed by this module,
see, for e.g., Fig 5.6 (Chapter 5) of:

Datta, A., 2017. Surface wave mode coupling due to lateral heterogeneity and its impact on waveform inversions, Ph.D. thesis, University of Cambridge.
"""

######################################################################################

def check_with_dispfile(dfile, nmodes, period, c_egnfile):
    # Read in theoretical group velocities for comparison with those estimated from energy integrals
    reoobj2 = reo.read_disp([dfile],0,nmodes-1)
    th_udisp = reoobj2.modudisp[0]
    # while we're at it, get the phase velocities too (read from dispersion file)
    th_cdisp = reoobj2.modcdisp[0]
    u_dispfile = np.arange(nmodes, dtype='float')
    c_dispfile = np.arange(nmodes, dtype='float')
    for m in range(nmodes):
        uval=[y for x,y in th_udisp[m] if np.round(x,6)==np.round(1./period,6)]
        cval=[y for x,y in th_cdisp[m] if np.round(x,6)==np.round(1./period,6)]
        try:
            u_dispfile[m] = uval[0]
            c_dispfile[m] = cval[0]
        except IndexError:
            raise SystemExit("From egnfn_integrals_norms: Could not find period sample in disp file")

    # A little check to make sure dispersion values read from disp file and eigen file are consistent
    cdiff = 100*abs(c_dispfile - c_egnfile)/c_dispfile
    if np.any(cdiff>1e-3):
        print(cdiff)
        raise SystemExit("Inconsistency in phase velocities read from dispersion/eigenfunction files")

    return u_dispfile, c_dispfile

######################################################################################

class energy_integrals_lov():

    def __init__(self, efile, ps, orig_dep, dcon, use_wt=True, do_all=False):

        try:
            reoobj=reo.read_egnfile_per(efile,ps)
        except Exception as e:
            # raise ValueError(str(e))
            raise Exception(str(e))
        self.usewt=use_wt
        self.omega=2*np.pi/ps
        self.ps = ps
        self.orig_l1=reoobj.utmat
        self.orig_l2=reoobj.ttmat
        if reoobj.dep.size != orig_dep.size:
            # NB July 2020: we use reoobj.dep to check the number of depth points, but DO NOT use it as the initial depth
            # array for this class ("origdep"). INFINITESIMAL discrepancies in the depth values read from the mod file
            # and the eigenfunction file, can wreak havoc with the integrations performed here. "dcon" is derived from the
            # depth values of the mod file, so should "origdep" be.
            raise Exception("Mismatch in depth points from mod file and eigenfunction file: %d %d" %(orig_dep.size, reoobj.dep.size))
        self.origdep=orig_dep #reoobj.dep
        self.dcon=dcon
        self.nmodes=self.orig_l1.shape[1]
        self.norms=np.zeros(self.nmodes) # vector of integrals
        self.matint=np.zeros((self.nmodes,self.nmodes)) # matrix of integrals
        self.omsq_I1=np.zeros(self.nmodes) # vector of integrals
        self.omsq_I2=np.zeros(self.nmodes) # vector of integrals
        self.I3=np.zeros(self.nmodes) # vector of integrals
        self.kmode=reoobj.wavnum
        self.rho=reoobj.rho
        self.mu=reoobj.mu
        self.phasevel=np.array([self.omega/self.kmode[i] for i in range(self.nmodes)])

        self.d_l1_dz = self.omega*self.orig_l2/(self.mu.reshape(reoobj.mu.size,1))
        dz_mod = self.origdep[1] - self.origdep[0] # assuming that the model is sampled uniformly in depth
        dyz=np.gradient(self.orig_l1[:,0],dz_mod)
        if __name__=='__main__' and len(sys.argv)==5:
            figlov=plt.figure()
            ax=figlov.add_subplot(111)
            ax.plot(dyz,self.origdep,label='num. d_l1_dz')
            ax.plot(self.d_l1_dz[:,0],self.origdep,label='theor. d_l1_dz')
            ax.axvline(0,ls='--')
            ax.legend(loc='best')
            ax.set_ylim(100,0)
            # ax.set_ylim(self.origdep[-1],0)

        #**********************************************************
        # Before we integrate we need to add extra depth points to
        # the arrays in order to account for discontinuities in
        # the integration

        self.dep=self.origdep
        self.l1=self.orig_l1
        self.egnfn_tt=self.orig_l2

        eps=0.000001
        for dc in self.dcon:
            valtoins=[dc-eps,dc+eps]
            ind=np.searchsorted(self.dep,valtoins)
            # Get values of eigenfunctions AT discontinuity
            efdcon_ut=self.l1[ind[0],:]
            efdcon_tt=self.egnfn_tt[ind[0],:]
            # print "For interface ", dc
            # print "ind is ", ind
            # print "eigenfunction before adding extra point: ", self.l1[ind[0]-5:ind[0]+5,:]
            # print "Value of eigenfunction at interface: ", efdcon_ut
            # Allocate value to point just above discontinuity
            self.l1=np.insert(self.l1,ind[0],efdcon_ut,axis=0)
            self.egnfn_tt=np.insert(self.egnfn_tt,ind[0],efdcon_tt,axis=0)
            # and to point just below
            self.l1=np.insert(self.l1,ind[1]+1,efdcon_ut,axis=0)
            self.egnfn_tt=np.insert(self.egnfn_tt,ind[1]+1,efdcon_tt,axis=0)
            # Add extra points to the depth, rho and mu arrays
            self.dep=np.insert(self.dep,ind,valtoins)
            #print "LOOKING FOR ", ind[0]-1, ind[1], len(self.mu)
            if len(self.mu)>ind[1]:
            	valtoins_mu=[self.mu[ind[0]-1],self.mu[ind[1]]]
            	valtoins_rho=[self.rho[ind[0]-1],self.rho[ind[1]]]
            else:
            #this happens when the very last depth point is itself an interface
            	valtoins_mu=[self.mu[ind[0]-1],self.mu[ind[1]-1]]
            	valtoins_rho=[self.rho[ind[0]-1],self.rho[ind[1]-1]]
            self.mu=np.insert(self.mu,ind,valtoins_mu)
            self.rho=np.insert(self.rho,ind,valtoins_rho)
            # print "eigenfunction after adding extra point: ", self.l1[ind[0]-5:ind[0]+5,:]
            # print "mu after extra: ", self.mu[ind[0]-5:ind[0]+5]
            # print "depth after extra: ", self.dep[ind[0]-5:ind[0]+5]

        if do_all:
            self.compute_integrals(3)
            self.orthogonality_products()
            self.quantities_from_integrals()

    #*************************************************************************************

    def cross_check(self, dfile):

        self.uth_thisper, self.c = check_with_dispfile(dfile, self.nmodes, self.ps, self.phasevel)

    #*************************************************************************************

    def compute_integrals(self, howmany):

        b2_by_mu=np.empty((self.egnfn_tt.shape[0],self.egnfn_tt.shape[1]))

        for i in range(self.nmodes):

            b2_by_mu[:,i] = self.egnfn_tt[:,i]/self.mu
            wt1=self.rho
            wt2=self.mu

            self.omsq_I1[i]=0.5*self.integrate(self.l1[:,i],self.l1[:,i],wt1)
            if howmany>1:
                self.omsq_I2[i]=0.5*self.integrate(self.l1[:,i],self.l1[:,i],wt2)
            if howmany>2:
                self.I3[i]=0.5*self.integrate(b2_by_mu[:,i],b2_by_mu[:,i],wt2)

    #*************************************************************************************

    def orthogonality_products(self):

        for ij in itertools.combinations_with_replacement(range(self.nmodes),2):
            i=ij[0]
            j=ij[1]
            if not self.usewt:
                # so far the only use case for this is SWRT, Alsop method
                wtnorm=np.ones(len(self.dep))
            else:
                wtnorm=self.kmode[i]*self.mu
            ans=(self.integrate(self.l1[:,i],self.l1[:,j],wtnorm))/self.omega
            self.matint[j][i]=ans
            if __name__=='__main__':
                print("Mode number %d and %d: %f" %(i,j,ans))
        self.norms=np.diagonal(self.matint)
        #self.omsq_I2=np.diagonal(self.matint)

    #*************************************************************************************

    def quantities_from_integrals(self):

        # Group velocity
        self.u_est = self.omsq_I2/(self.phasevel*self.omsq_I1)

        self.ksq_I2=self.omsq_I2/(self.phasevel**2)
        # Elastic or potential energy
        self.rhs = self.ksq_I2 + self.I3

    #*************************************************************************************

    def integrate(self,egf1,egf2,wfn):

        sumint=0.0
        for l in range(len(self.dcon)):
            sid=np.searchsorted(self.dep,self.dcon[l])
            if l>0:
                sid_prev=np.searchsorted(self.dep,self.dcon[l-1])
                top=sid_prev+1
            else:
                top=0
            #if __name__=='__main__':
            #	print "Depth sample excluded from integration: ",self.dep[sid]
            # integration above each horizontal interface
            phi_ij=egf1[top:sid]*egf2[top:sid]
            prod=phi_ij*wfn[top:sid]
            int_above=spi.simps(prod,self.dep[top:sid])
            sumint += int_above

        if len(self.dcon)==0:
        # exceptional case of no disontinuities, i.e. a half-space
            sid = -1

        # integration below deepest horizontal interface
        phi_ij=egf1[sid+1:]*egf2[sid+1:]
        prod=phi_ij*wfn[sid+1:]
        int_below=spi.simps(prod,self.dep[sid+1:])

        return sumint + int_below

##############################################################################################

class energy_integrals_ray():

    def __init__(self, efile, ps, orig_dep, dcon, use_wt=True, do_all=False):

        try:
            reoobj=reo.read_egnfile_per(efile,ps)
        except Exception as e:
            # raise ValueError(str(e))
            raise Exception(str(e))
        # self.usewt = use_wt # not really used in the Rayleigh case
        self.omega=2*np.pi/ps
        self.ps=ps
        self.orig_b1=reoobj.uzmat
        self.orig_b2=reoobj.urmat
        self.orig_b3=reoobj.tzmat
        self.orig_b4=reoobj.trmat
        if reoobj.dep.size != orig_dep.size:
            # NB July 2020: we use reoobj.dep to check the number of depth points, but DO NOT use it as the initial depth
            # array for this class ("origdep"). INFINITESIMAL discrepancies in the depth values read from the mod file
            # and the eigenfunction file, can wreak havoc with the integrations performed here. "dcon" is derived from the
            # depth values of the mod file, so should "origdep" be.
            raise Exception("Mismatch in depth points from mod file and eigenfunction file: %d %d" %(orig_dep.size, reoobj.dep.size))
        self.origdep=orig_dep #reoobj.dep
        self.dcon=dcon
        self.nmodes=self.orig_b1.shape[1]
        self.matint=np.zeros((self.nmodes,self.nmodes)) # matrix of integrals
        self.norms=np.zeros(self.nmodes) # vector of integrals
        self.omsq_I1=np.zeros(self.nmodes) # vector of integrals
        self.omsq_I2=np.zeros(self.nmodes) # vector of integrals
        self.omsq_I3=np.zeros(self.nmodes) # vector of integrals
        self.I4=np.zeros(self.nmodes) # vector of integrals

        self.kmode=reoobj.wavnum #.reshape(1,len(reoobj.wavnum))
        self.mu=reoobj.mu #.reshape(len(reoobj.mu),1)
        self.rho=reoobj.rho #.reshape(len(reoobj.mu),1)
        self.lamda=reoobj.lamda

        # self.orig_mu = np.copy(self.mu)
        # self.orig_rho = np.copy(self.rho)
        # self.orig_lam = np.copy(self.lamda)

        if __name__=='__main__':
            print("Original shapes of mu and kmode: ", self.mu.shape, self.kmode.shape)
        self.phasevel=np.array([self.omega/self.kmode[i] for i in range(self.nmodes)])
        mu = self.mu.reshape(reoobj.mu.size,1)
        lamda = self.lamda.reshape(reoobj.mu.size,1)
        kmode = self.kmode.reshape(1,reoobj.wavnum.size)
        kmu=np.dot(mu,kmode)
        klamda=np.dot(lamda,kmode)
        self.d_b2_dz=(self.omega*self.orig_b4-np.multiply(kmu,self.orig_b1))/mu # numpy.multiply does element wise array multiplication
        self.d_b1_dz=(np.multiply(klamda,self.orig_b2)+self.omega*self.orig_b3)/(lamda+2*mu)
        dz_mod = self.origdep[1] - self.origdep[0] # assuming that the model is sampled uniformly in depth
        dxz=np.gradient(self.orig_b2[:,0],dz_mod)
        dzz=np.gradient(self.orig_b1[:,0],dz_mod)
        if __name__=='__main__' and len(sys.argv)==5:
            figray=plt.figure()
            ax=figray.add_subplot(111)
            print("new shape is ", kmu.shape, self.d_b2_dz.shape)
            plt.plot(dxz,self.origdep,label='num. d_b2_dz')
            plt.plot(self.d_b2_dz[:,0],self.origdep,label='theor. d_b2_dz')
            ax.plot(dzz,self.origdep,label='num. d_b1_dz')
            ax.plot(self.d_b1_dz[:,0],self.origdep,label='theor. d_b1_dz')
            ax.axvline(0,ls='--')
            ax.legend(loc='best')
            ax.set_ylim(100,0)
            # ax.set_ylim(self.origdep[-1],0)

        #**********************************************************
        # Before we integrate we need to add extra depth points to
        # the arrays in order to account for discontinuities in
        # the integration

        def modify_egn(ind,egn_in):
            # Get values of eigenfunctions AT discontinuity
            efdcon=egn_in[ind[0],:]
            #print "For interface ", dc
            #print "ind is ", ind
            #print "Value of eigenfunction at interface: ", efdcon
            # Allocate value to point just above discontinuity
            egn_out=np.insert(egn_in,ind[0],efdcon,axis=0)
            # and to point just below
            egn_out=np.insert(egn_out,ind[1]+1,efdcon,axis=0)
            return egn_out

        self.b1=self.orig_b1
        self.b2=self.orig_b2
        self.b3=self.orig_b3
        self.b4=self.orig_b4
        #print "shapes are: ", (klamda+2*kmu).shape, self.b2.shape, self.d_b1_dz.shape, self.lamda.shape
        self.psi_xx=((lamda*self.d_b1_dz)-(np.multiply((klamda+2*kmu),self.orig_b2)))/self.omega
        self.dep=self.origdep

        eps=0.000001
        #print "eigenfunction before adding extra point: ", self.b1[25:35,:]
        for dc in self.dcon:
            valtoins=[dc-eps,dc+eps]
            ins_ind=np.searchsorted(self.dep,valtoins)
            self.b1=modify_egn(ins_ind,self.b1)
            self.b2=modify_egn(ins_ind,self.b2)
            self.b3=modify_egn(ins_ind,self.b3)
            self.b4=modify_egn(ins_ind,self.b4)
            self.psi_xx=modify_egn(ins_ind,self.psi_xx)
            # Next, add extra points to the depth array
            self.dep=np.insert(self.dep,ins_ind,valtoins)
            # and finally to the rho, mu and lambda arrays
            if len(self.mu)>ins_ind[1]:
            	valtoins_mu=[self.mu[ins_ind[0]-1],self.mu[ins_ind[1]]]
            	valtoins_rho=[self.rho[ins_ind[0]-1],self.rho[ins_ind[1]]]
            	valtoins_lam=[self.lamda[ins_ind[0]-1],self.lamda[ins_ind[1]]]
            else:
            # this happens when the very last depth point is itself an interface
            	valtoins_mu=[self.mu[ins_ind[0]-1],self.mu[ins_ind[1]-1]]
            	valtoins_rho=[self.rho[ins_ind[0]-1],self.rho[ins_ind[1]-1]]
            	valtoins_lam=[self.lamda[ins_ind[0]-1],self.lamda[ins_ind[1]-1]]
            self.mu=np.insert(self.mu,ins_ind,valtoins_mu)
            self.rho=np.insert(self.rho,ins_ind,valtoins_rho)
            self.lamda=np.insert(self.lamda,ins_ind,valtoins_lam)
        #print "eigenfunction after adding extra point: ", self.b1[25:35,:]
        #print "Shapes of altered eigenfunctions: ", self.b1.shape, self.psi_xx.shape

        # Define or redefine any necessary quantities
        lp2mu=self.lamda+(2*self.mu)
        kmu=np.outer(self.mu,self.kmode)
        klamda=np.outer(self.lamda,self.kmode)
        self.d_b2_dz=(self.omega*self.b4-np.multiply(kmu,self.b1))/self.mu.reshape(len(self.dep),1) # numpy.multiply does element wise array multiplication
        self.d_b1_dz=(np.multiply(klamda,self.b2)+self.omega*self.b3)/lp2mu.reshape(len(self.dep),1)
        dxz=np.gradient(self.b2,dz_mod,axis=0)
        dzz=np.gradient(self.b1,dz_mod,axis=0)

        # dxz=np.empty((len(self.dep),self.nmodes))
        # dzz=np.empty((len(self.dep),self.nmodes))
        # for m in range(self.nmodes):
        #     # loop over modes required for numpy versions older than 1.11
        #     dxz[:,m]=np.gradient(self.b2[:,m])
        #     dzz[:,m]=np.gradient(self.b1[:,m])

        if __name__=='__main__':
            print("shapes of mu and kmode: ", self.mu.shape, self.kmode.shape)
            print("shapes of kmu, klamda, mu: ", kmu.shape, klamda.shape, self.mu.shape)
            print("shapes of b1 and b2: ", self.b1.shape, self.b2.shape)
            print("shape of dxz and dzz: ", dxz.shape, dzz.shape)

        if do_all:
            self.compute_integrals(4)
            self.orthogonality_products()
            self.quantities_from_integrals()

    #*************************************************************************************

    def cross_check(self, dfile):

        self.uth_thisper, self.c = check_with_dispfile(dfile, self.nmodes, self.ps, self.phasevel)

    #*************************************************************************************

    def compute_integrals(self, howmany):

        wt1=self.rho
        wt2a=self.lamda+(2*self.mu) #lp2mu
        wt2b=self.mu
        wt3=self.lamda

        for i in range(self.nmodes):

            # print "COMPUTING INTEGRALS FOR MODE NUMBER: %d" %(i)

            self.omsq_I1[i]=0.5*(self.integrate(self.b1[:,i],self.b1[:,i],wt1,True)+self.integrate(self.b2[:,i],self.b2[:,i],wt1))
            if howmany>1:
                self.omsq_I2[i]=0.5*(self.integrate(self.b2[:,i],self.b2[:,i],wt2a)+self.integrate(self.b1[:,i],self.b1[:,i],wt2b))
            if howmany>2:
                self.omsq_I3[i]=(self.integrate(self.b1[:,i],self.d_b2_dz[:,i],wt2b)-self.integrate(self.b2[:,i],self.d_b1_dz[:,i],wt3))
            if howmany>3:
                self.I4[i]=0.5*(self.integrate(self.d_b1_dz[:,i],self.d_b1_dz[:,i],wt2a)+self.integrate(self.d_b2_dz[:,i],self.d_b2_dz[:,i],wt2b))/(self.omega**2)

    #*************************************************************************************

    def orthogonality_products(self):

        for ij in itertools.combinations_with_replacement(range(self.nmodes),2):
            i=ij[0]
            j=ij[1]
            ans=0.5*(self.normint(self.b1[:,i],self.b1[:,j],self.b2[:,i],self.b2[:,j],self.b4[:,i],self.b4[:,j],self.psi_xx[:,i],self.psi_xx[:,j]))
            self.matint[j][i]=ans
            if __name__=='__main__':
                print("Mode number %d and %d: %f" %(i,j,ans))
        self.norms=np.diagonal(self.matint)

    #*************************************************************************************

    def quantities_from_integrals(self):

        # group velocity
        self.u_est = (self.omsq_I2+(0.5*self.omsq_I3/self.kmode))/(self.phasevel*self.omsq_I1)

        # quantities that constitute the PE-KE relation
        self.ksq_I2=self.omsq_I2/(self.phasevel**2)
        self.k_I3=(self.kmode*self.omsq_I3)/(self.omega**2)
        self.rhs = self.ksq_I2 + self.k_I3 + self.I4

    #*************************************************************************************

    def integrate(self, f1, f2, wfn, print_info=False):

        sumint=0.0
        for l in range(len(self.dcon)):
            sid=np.searchsorted(self.dep,self.dcon[l])
            if l>0:
                sid_prev=np.searchsorted(self.dep,self.dcon[l-1])
                top=sid_prev+1
            else:
                top=0
            phi_ij=f1[top:sid]*f2[top:sid]
            prod=phi_ij*wfn[top:sid]
            int_above=spi.simps(prod,self.dep[top:sid])
            sumint += int_above

            if print_info:
                pass
                # print "Dicon, prev_discon, int_above, sumint: %f %f %f %f" %(self.dcon[l], self.dcon[l-1], int_above, sumint)

        if len(self.dcon)==0:
        # exceptional case of no disontinuities, i.e. a half-space
            sid = -1

        # integration below deepest horizontal interface
        phi_ij=f1[sid+1:]*f2[sid+1:]
        prod=phi_ij*wfn[sid+1:]
        int_below=spi.simps(prod,self.dep[sid+1:])
        return sumint + int_below

    #*************************************************************************************

    def normint(self,b1_m,b1_n,b2_m,b2_n,b4_m,b4_n,psi_m,psi_n):

        sumint=0.0
        for l in range(len(self.dcon)):
            sid=np.searchsorted(self.dep,self.dcon[l])
            if l>0:
                sid_prev=np.searchsorted(self.dep,self.dcon[l-1])
                top=sid_prev+1
            else:
                top=0
            #print "Depth sample excluded from integration: ",self.dep[sid]
            # integration above each horizontal interface
            integrand=b2_n[top:sid]*psi_m[top:sid] + b2_m[top:sid]*psi_n[top:sid] - b1_n[top:sid]*b4_m[top:sid] - b1_m[top:sid]*b4_n[top:sid]
            int_above=spi.simps(integrand,self.dep[top:sid])
            sumint += int_above

        if len(self.dcon)==0:
        # exceptional case of no disontinuities, i.e. a half-space
            sid = -1

        # integration below deepest horizontal interface
        integrand=b2_n[sid+1:]*psi_m[sid+1:] + b2_m[sid+1:]*psi_n[sid+1:] - b1_n[sid+1:]*b4_m[sid+1:] - b1_m[sid+1:]*b4_n[sid+1:]
        int_below=spi.simps(integrand,self.dep[sid+1:])

        return abs(sumint + int_below)

#############################################################################################

if __name__=='__main__':

    # Standard modules
    import os
    import sys
    import matplotlib.pyplot as plt

    #############################################################################################

    def usage():

        print("This script needs FOUR (or more) arguments. USAGE:")
        print("\tpython %s <mod_file> <egn_file> <disp_file> <single number (period) OR \
    three numbers (freq info) >" %(os.path.basename(sys.argv[0])))

    #*******************************************************************************************

    def read_model(mod_file):
        """ the purpose of this function is to extract elastic parameters of the model and
            identify horizontal interfaces where those parameters are discontinuous """

        oreo = reo.read_modfile([mod_file])

        vp = oreo.alpha
        vs = oreo.beta
        rho = oreo.rho
        mod_deps = oreo.deps

        uvals_vp, ind_vp = np.unique(vp, return_index=True)
        uvals_vs, ind_vs = np.unique(vs, return_index=True)
        uvals_rho, ind_rho = np.unique(rho, return_index=True)

        """ NB: numpy.unique returns SORTED unique VALUES by default. This can mess things up if
        parameter VALUES are not all increasing with depth (e.g. low velocity layer at depth). Hence
        it is important to SORT the INDICES obtained above. """

        # uvals=[b for a,b in sorted(zip(ind,uvals))]

        vp_ind = sorted(ind_vp)
        vs_ind = sorted(ind_vs)
        rho_ind = sorted(ind_rho)
        v_ind= np.union1d(vp_ind, vs_ind)
        hif_ind = np.union1d(v_ind, rho_ind)

        mod_hif = mod_deps[hif_ind][1:]
        # first element of the array is ignored because it corresponds to the surface; z=0

        return mod_deps[:-1], mod_hif

    #*******************************************************************************************

    def figs_single_per(cobj):

        egy_lhs = cobj.omsq_I1
        egy_rhs = cobj.rhs
        # flux_est=2*cobj.omsq_I1*cobj.u_est[:]
        flux_est = (cobj.omsq_I1 + cobj.rhs) * cobj.u_est[:]

        figu=plt.figure()
        axu = figu.add_subplot(111)
        axu.plot(range(cobj.nmodes),cobj.uth_thisper,'D',label='Theoretical')
        axu.plot(range(cobj.nmodes),cobj.u_est,'o',label='Estimated')
        axu.legend()
        axu.set_xlabel('Mode number')
        axu.set_ylabel('Goup velocity (km/s)')

        fige=plt.figure()
        axe = fige.add_subplot(111)
        axe.plot(range(cobj.nmodes),cobj.omsq_I1,'D',label='Kinetic energy (LHS)')
        axe.plot(range(cobj.nmodes),cobj.rhs,'o',label='Elastic energy (RHS)')
        axe.legend()
        axe.set_xlabel('Mode number')
        axe.set_ylabel('Energy density (arbitrary units)')

        fign=plt.figure()
        axn=fign.add_subplot(111)
        # axn.plot(range(cobj.nmodes)[1:],flux_est[1:],'D',markerfacecolor='w',markeredgewidth=1,label='Energy integrals')
        # axn.plot(range(cobj.nmodes)[1:],cobj.norms[1:],'o',markersize=3.0,label='Herrera\'s norm')
        axn.plot(range(cobj.nmodes),flux_est,'D',markerfacecolor='w',markeredgewidth=1,label='Energy integrals')
        axn.plot(range(cobj.nmodes),cobj.norms,'o',markersize=3.0,label='Herrera\'s norm')
        axn.legend(loc=2)
        axn.set_xlabel('Mode number')
        axn.set_ylabel('Energy Flux (arbitrary units)')

    #*******************************************************************************************

    def figs_mult_per(cobj):

        nmodes = acc_from_ee.shape[0]
        see_modes=1 #nmodes

        figmp_ee=plt.figure()
        axmp_ee=figmp_ee.add_subplot(111)
        for mode in range(see_modes):
            cname="Mode %s" %(mode)
            axmp_ee.plot(do_per, acc_from_ee[mode,:], '-o', label=cname)
        # axmp_ee.set_ylim(0.8995,1.005)
        axmp_ee.set_xlabel("Period [s]")
        axmp_ee.set_ylabel("Measure of accuracy")
        axmp_ee.legend()
        axmp_ee.set_title("From energy equivalence")

        figmp_gv=plt.figure()
        axmp_gv=figmp_gv.add_subplot(111)
        for mode in range(see_modes):
            cname="Mode %s" %(mode)
            axmp_gv.plot(do_per, acc_from_gv[mode,:], '-o', label=cname)
        # axmp_gv.set_ylim(0.8995,1.005)
        axmp_gv.set_xlabel("Period [s]")
        axmp_gv.set_ylabel("Measure of accuracy")
        axmp_gv.legend()
        axmp_gv.set_title("From group velocities")

    ############################################################################################

    try:
        modfile=sys.argv[1]
        effile=sys.argv[2] # eigenfunction file
        dispfile=sys.argv[3] # dispersion file
        dummy_chk=sys.argv[4]
        if len(sys.argv)==5:
            per=float(sys.argv[4]) # period in seconds
        elif len(sys.argv)>5:
            nfs=int(sys.argv[4]) # number of frequency samples
            freq_step=float(sys.argv[5])
            freq_start=float(sys.argv[6])
    except IndexError:
        usage()
        exit()

    #*******************************************************************************************

    dep_pts, discon = read_model(modfile)
    print("Discontinuities are: ", discon, len(discon))

    if len(sys.argv)>5:
        single_per=False
        freq_end = freq_start + (nfs-1)*freq_step
        fhz = np.linspace(freq_start, freq_end, nfs)
        per = 1/fhz
        do_per = per[per == per.astype(int)][::-1]
    else:
        single_per=True
        do_per = np.array([per])

    #*******************************************************************************************

    for i,dp in enumerate(do_per):
        print("******")
        print("PERIOD %.1f" %(dp))

        if effile.find('.lov')>0:
            try:
                olobj = energy_integrals_lov(effile, dp, dep_pts, discon, True, True)
                eiobj = olobj
            except Exception as e:
                msg=e
                if "period sample" in str(e):
                    msg="Aborted: please consult the eigenfunction file and specify a different period."
                raise SystemExit(msg)

            # olobj.cross_check(dispfile)

            print("LHS of energy equation (omsq_I1): ", olobj.omsq_I1)
            print("RHS of energy equation (ksq_I2 + I3): ", olobj.rhs)

        elif effile.find('.ray')>0:
            try:
                orobj = energy_integrals_ray(effile, dp, dep_pts, discon, True, True)
                eiobj = orobj
            except Exception as e:
                print(e)
                if "period sample" in str(e):
                    msg="Aborted: please consult the eigenfunction file and specify a different period."
                else:
                    msg="Aborted"
                raise SystemExit(msg)

            # orobj.cross_check(dispfile)

            print("LHS of energy equation (omsq_I1): ", orobj.omsq_I1)
            print("RHS of energy equation (ksq_I2 + k_I3 + I4): ", orobj.rhs)

        eiobj.cross_check(dispfile)

        ted = eiobj.omsq_I1 + eiobj.rhs
        eef = ted * eiobj.u_est

        print("Total energy density: ", ted)
        print("Estimated group velocity: ", eiobj.u_est)
        print("Theoretical group velocity: ", eiobj.uth_thisper)
        print("Estimated energy flux (energy density x group velocity): ", eef)
        print("Herrera's norm (energy flux): ", eiobj.norms)
        print(eiobj.matint)

        acc_gv_opt = eiobj.u_est/eiobj.uth_thisper
        acc_ee_opt = eiobj.rhs/eiobj.omsq_I1
        acc_hn_opt = eef/eiobj.norms

        acc_gv = acc_gv_opt if np.all(acc_gv_opt<1) else 1/acc_gv_opt
        acc_ee = acc_ee_opt if np.all(acc_ee_opt<1) else 1/acc_ee_opt
        acc_hn = acc_hn_opt if np.all(acc_hn_opt<1) else 1/acc_hn_opt

        print("")
        print("Accuracy from group velocities: ", acc_gv)
        print("Accuracy from energy relation: ", acc_ee)
        print("Accuracy from Herrera's norm (energy flux): ", acc_hn)
        print("******")
        print("")

        if not single_per:
            if i==0:
            # shortest period, and therefore maximum modes
                acc_from_gv = np.nan * np.ones((eiobj.norms.size, do_per.size))
                acc_from_ee = np.nan * np.ones((eiobj.norms.size, do_per.size))
                # NB: "nan" is useful for plotting - higher modes will simply not
                # be plotted for the longer periods at which they don't exist.

            acc_from_gv[:eiobj.norms.size,i] = acc_gv
            acc_from_ee[:eiobj.norms.size,i] = acc_ee

    if single_per and len(eiobj.kmode)>1:
    # single period
        figs_single_per(eiobj)
    elif not single_per:
    # multiple periods
        figs_mult_per(eiobj)

    plt.show()
