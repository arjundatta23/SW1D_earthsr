#!/usr/bin/python

# Standard modules
import numpy as np
import scipy.special as ssp

# Custom modules
import SW1D_earthsr.egnfunc_integrals_norms as ein

""" NB: in this code only a part of the above "ein" module is used. However, before using this code it is advisable
    to run the module in stand-alone mode, whence it is executed in its entirety, to ascertain
    the quality of the eigenfunction integrals.
"""
###################################################################################################################

sdep=0.0 # source depth (km)

comps=['x', 'y', 'z']
# crind=[-3, -2, -1]
xyz_to_xz = {0: 0, 1: 10, 2: 1}

def tensor2_for_grid(npts_x, npts_y, npts_z, dimen):

    init_surf_cmplx = lambda m,n: np.zeros((m,n), dtype='complex')
    init_vol_cmplx = lambda m,n,o:  np.zeros((m,n,o), dtype='complex')

    if dimen==2:
    # 2-D calculation
        M11=init_surf_cmplx(npts_y,npts_x)
        M21=init_surf_cmplx(npts_y,npts_x)
        M31=init_surf_cmplx(npts_y,npts_x)
        M12=init_surf_cmplx(npts_y,npts_x)
        M22=init_surf_cmplx(npts_y,npts_x)
        M32=init_surf_cmplx(npts_y,npts_x)
        M13=init_surf_cmplx(npts_y,npts_x)
        M23=init_surf_cmplx(npts_y,npts_x)
        M33=init_surf_cmplx(npts_y,npts_x)
    elif dimen==3:
    # 3-D calculation
        M11=init_vol_cmplx(npts_z,npts_y,npts_x)
        M21=init_vol_cmplx(npts_z,npts_y,npts_x)
        M31=init_vol_cmplx(npts_z,npts_y,npts_x)
        M12=init_vol_cmplx(npts_z,npts_y,npts_x)
        M22=init_vol_cmplx(npts_z,npts_y,npts_x)
        M32=init_vol_cmplx(npts_z,npts_y,npts_x)
        M13=init_vol_cmplx(npts_z,npts_y,npts_x)
        M23=init_vol_cmplx(npts_z,npts_y,npts_x)
        M33=init_vol_cmplx(npts_z,npts_y,npts_x)

    return M11, M21, M31, M12, M22, M32, M13, M23, M33

###################################################################################################################

class Green_SW_monofreq:

    """NB: The Green's function computed by this class is truncated to the desired depth
    "nzmax" (less than maximum model depth in general), by truncating the medium eigenfunctions.
    However the energy integral(s) required in the Green's function is/are computed using the ENTIRE
    depth extent of the eigenfunctions. So the GF truncation is WITHOUT loss of accuracy.
    """

#    def __init__(self, egnfile, dispfile, per, mod_deps, discon, nzmax):
    def __init__(self, per, mod_deps, discon, nzmax):

        self.per = per
        self.mod_deps = mod_deps
        self.discon = discon
        self.nz = nzmax #ei_obj.orig_b1.shape[0]

    #********************************************************************************

    def prepare_egn(self, egnfile, dispfile, sw_type):

        try:
            if sw_type=='ray':
                ei_obj = ein.energy_integrals_ray(egnfile, self.per, self.mod_deps, self.discon)
                rep_egnfn = ei_obj.orig_b1
            elif sw_type=='lov':
                self.love_included=True
                ei_obj = ein.energy_integrals_lov(egnfile, self.per, self.mod_deps, self.discon)
                rep_egnfn = ei_obj.orig_l1
            else:
                raise Exception("Unknown surface wave type: cannot prepare EFs for GF calculation")
        except Exception as e:
            print(e)
            if "period sample" in str(e):
                msg="Frequency arrays (stepping) mismatch: code vs. input eigenfunction file"
            elif "Error reading" in str(e):
                msg="Supplied eigenfunction file incompatible with reading module"
            elif "Mismatch in depth" in str(e):
                msg=e
            else:
                msg="Unknown error with ein module"
            raise Exception(msg)

        ei_obj.cross_check(dispfile)
        ei_obj.compute_integrals(1)
        # ei_obj.orthogonality_products()

        if ( (ei_obj.origdep.size != rep_egnfn.shape[0]) or (ei_obj.c.size != rep_egnfn.shape[1]) ):
            raise SystemExit("Inconsistency in depth samples/mode numbers of eigenfunctions")

        nm = ei_obj.c.size

        # self.deps_mod = ei_obj.origdep
        c=ei_obj.c
        U=ei_obj.uth_thisper
        oI1=ei_obj.omsq_I1
        kmode=ei_obj.kmode

        cfac = 1./(8*c*U*oI1)
        # common factor; independent of receiver location

        # h=np.argwhere(ei_obj.origdep==sdep)[0][0]
        h=np.searchsorted(ei_obj.origdep, sdep)

        if sw_type=='ray':

            self.nm_ray = nm
            self.cfac_ray = cfac
            self.kmode_ray = kmode
            self.hm_ray = 0 #nm-1
            # set hm=0 to incorporate fundamental mode only, =(nm-1) for sum of modes, in G calculations

            # print(ei_obj.orig_b1.shape)

            self.b1b1 = ei_obj.orig_b1[h,:self.hm_ray+1] * ei_obj.orig_b1[:self.nz,:self.hm_ray+1]
            self.b1b2 = ei_obj.orig_b1[h,:self.hm_ray+1] * ei_obj.orig_b2[:self.nz,:self.hm_ray+1]
            self.b2b1 = ei_obj.orig_b2[h,:self.hm_ray+1] * ei_obj.orig_b1[:self.nz,:self.hm_ray+1]
            self.b2b2 = ei_obj.orig_b2[h,:self.hm_ray+1] * ei_obj.orig_b2[:self.nz,:self.hm_ray+1]

            # print(self.b1b1.shape)

            self.b1db1 = ei_obj.orig_b1[h,:self.hm_ray+1] * ei_obj.d_b1_dz[:self.nz,:self.hm_ray+1]
            self.b1db2 = ei_obj.orig_b1[h,:self.hm_ray+1] * ei_obj.d_b2_dz[:self.nz,:self.hm_ray+1]
            self.b2db1 = ei_obj.orig_b2[h,:self.hm_ray+1] * ei_obj.d_b1_dz[:self.nz,:self.hm_ray+1]
            self.b2db2 = ei_obj.orig_b2[h,:self.hm_ray+1] * ei_obj.d_b2_dz[:self.nz,:self.hm_ray+1]

        elif sw_type=='lov':

            self.nm_lov = nm
            self.cfac_lov = cfac
            self.kmode_lov = kmode
            self.hm_lov = 0 #nm-1

            self.l1l1  = ei_obj.orig_l1[h,:self.hm_lov+1] * ei_obj.orig_l1[:self.nz,:self.hm_lov+1]
            self.l1dl1 = ei_obj.orig_l1[h,:self.hm_lov+1] * ei_obj.d_l1_dz[:self.nz,:self.hm_lov+1]

    #********************************************************************************

    def G_cartesian_grid(self, x, y, r, dimflag):

        ny=r.shape[0]
        try:
            nx=r.shape[1]
        except IndexError:
        # "r" not a 2-D array
            nx=1

        G11, G21, G31, G12, G22, G32, G13, G23, G33 = tensor2_for_grid(nx, ny, self.nz, dimflag)

        if dimflag==2:
            take_dep=0
        elif dimflag==3:
            take_dep=slice(self.nz)

        # Rayleigh wave part
        for m in range(self.hm_ray+1):

            # print("Working on Rayleigh mode %d" %(m))
            # print(self.cfac_ray[m]/self.cfac_ray[0])

            # scal_fac_m = self.cfac_ray[m] * np.sqrt(2./(np.pi*self.kmode_ray[m]*r)) * np.exp(-1j*(self.kmode_ray[m]*r + np.pi/4))
            # far-field approximation from Aki-Richards
            scal_fac_m = self.cfac_ray[m] * ssp.hankel1(0,self.kmode_ray[m]*r) * 1j * 0.25

            # NB: NumPy's array broadcasting used for multiplying arrays with 'incompatible' shapes.
            # Explanation of broadcasting as applied here: b1b2 etc. are 1-D arrays (for a given mode),
            # we want to multiply them with some value at each point on a 2-D grid; we simply "broadcast"
            # them onto the grid, rather than actually store their values on each point of the grid.

            G11 += ( self.b2b2[:,m,None,None] * ( (x**2/r**2) * scal_fac_m )[None,...] )[take_dep,...]
            G12 += ( self.b2b2[:,m,None,None] * ( (x*y/r**2) * scal_fac_m )[None,...] )[take_dep,...]
            G13 += ( 1j * self.b1b2[:,m,None,None] * ( (x/r) * scal_fac_m )[None,...] )[take_dep,...]

            # G21 = G12, so it is not explicitly computed
            G22 += ( self.b2b2[:,m,None,None] * ( (y**2/r**2) * scal_fac_m )[None,...] )[take_dep,...]
            G23 += ( 1j * self.b1b2[:,m,None,None] * ( (y/r) * scal_fac_m )[None,...] )[take_dep,...]

            G31 += ( -1j * self.b2b1[:,m,None,None] * ( (x/r) * scal_fac_m )[None,...] )[take_dep,...]
            G32 += ( -1j * self.b2b1[:,m,None,None] * ( (y/r) * scal_fac_m )[None,...] )[take_dep,...]
            G33 += ( self.b1b1[:,m,None,None] * scal_fac_m[None,...] )[take_dep,...]

        if hasattr(self, 'love_included'):
        # Love wave part
            for m in range(self.hm_lov+1):

                # print("Working on Love mode %d" %(m))

                # scal_fac_m = self.cfac_lov[m] * np.sqrt(2./(np.pi*self.kmode_lov[m]*r)) * np.exp(-1j*(self.kmode_lov[m]*r + np.pi/4))
                # far-field approximation from Aki-Richards
                scal_fac_m = self.cfac_lov[m] * ssp.hankel1(0,self.kmode_lov[m]*r) * 1j * 0.25

                G11 += ( self.l1l1[:,m,None,None] * ( (y**2/r**2) * scal_fac_m )[None,...] )[take_dep,...]
                G12 += ( -1 * self.l1l1[:,m,None,None] * ( (x*y/r**2) * scal_fac_m )[None,...] )[take_dep,...]

                # G21 = G12 again
                G22 += ( self.l1l1[:,m,None,None] * ( (x**2/r**2) * scal_fac_m )[None,...] )[take_dep,...]

        # both parts done, now build the full tensor
        self.Gtensor = np.array(([G11,G12,G13],[G12,G22,G23],[G31,G32,G33]))

    #********************************************************************************

    def gradG_cartesian(self, x, y, r, dimflag):

        ny=r.shape[0]
        try:
            nx=r.shape[1]
        except IndexError:
        # "r" not a 2-D array
            nx=1

        if dimflag==2:
            take_dep=0
        elif dimflag==3:
            take_dep=slice(self.nz)

        for ic, comp in enumerate(comps):

            M1_11, M1_21, M1_31, M1_12, M1_22, M1_32, M1_13, M1_23, M1_33 = tensor2_for_grid(nx, ny, self.nz, dimflag)
            M2_11, M2_21, M2_31, M2_12, M2_22, M2_32, M2_13, M2_23, M2_33 = tensor2_for_grid(nx, ny, self.nz, dimflag)

            if ic==0:
                # self.gradG = np.zeros( ((3,3,3) + M1_11.shape), dtype='complex' )
                self.gradG = np.zeros( ((2,3,3) + M1_11.shape), dtype='complex' )
                """ the complete gradG is actually 3x3x3, since G is 3x3. However in this code
                    we only need gradG for x and z sources (see theory), i.e. the 1st and 3rd
                    columns of the G tensor, hence we use 2x3x3 to save memory.
                """

            # Rayleigh wave part
            for m in range(self.hm_ray+1):

                M1_scalf_m = self.cfac_ray[m] * ssp.hankel1(0,self.kmode_ray[m]*r) * 1j * 0.25
                M2_scalf_m = self.cfac_ray[m] * ssp.hankel1(1,self.kmode_ray[m]*r) * self.kmode_ray[m] * -1j * 0.25

                if ic==0:
                # gradient of G_x (first column of G tensor)
                    #******* compute M1
                    M1_11 += ( self.b2b2[:,m,None,None] * ( (2*x*(y**2)/r**4) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_12 += (-self.b2b2[:,m,None,None] * ( (2*y*(x**2)/r**4) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_13 += ( self.b2db2[:,m,None,None] * ( (x**2/r**2) * M1_scalf_m )[None,...] )[take_dep,...]

                    M1_21 += ( self.b2b2[:,m,None,None] * ( (y*(y**2-x**2)/r**4) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_22 += ( self.b2b2[:,m,None,None] * ( (x*(x**2-y**2)/r**4) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_23 += ( self.b2db2[:,m,None,None] * ( ((x*y)/r**2) * M1_scalf_m )[None,...] )[take_dep,...]

                    M1_31 += ( -1j * self.b2b1[:,m,None,None] * ( (y**2/r**3) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_32 += ( 1j * self.b2b1[:,m,None,None] * ( ((x*y)/r**3) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_33 += ( -1j * self.b2db1[:,m,None,None] * ( (x/r) * M1_scalf_m )[None,...] )[take_dep,...]

                    #******* compute M2
                    M2_11 += ( self.b2b2[:,m,None,None] * ( (x**3/r**3) * M2_scalf_m )[None,...] )[take_dep,...]
                    M2_12 += ( self.b2b2[:,m,None,None] * ( ((y*(x**2))/r**3) * M2_scalf_m )[None,...] )[take_dep,...]
                    # M2_13 = 0

                    M2_21 += ( self.b2b2[:,m,None,None] * ( ((y*(x**2))/r**3) * M2_scalf_m )[None,...] )[take_dep,...]
                    M2_22 += ( self.b2b2[:,m,None,None] * ( ((x*(y**2))/r**3) * M2_scalf_m )[None,...] )[take_dep,...]
                    # M2_23 = 0

                    M2_31 += ( -1j * self.b2b1[:,m,None,None] * ( (x**2/r**2) * M2_scalf_m )[None,...] )[take_dep,...]
                    M2_32 += ( -1j * self.b2b1[:,m,None,None] * ( ((x*y)/r**2) * M2_scalf_m )[None,...] )[take_dep,...]
                    # M2_33 = 0

                elif ic==1:
                # gradient of G_y (second column of G tensor); presently NOT REQUIRED in the structure kernels code
                    pass

                elif ic==2:
                # gradient of G_z (third column of G tensor)
                    #******* compute M1
                    M1_11 += ( 1j * self.b1b2[:,m,None,None] * ( (y**2/r**3) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_12 += ( -1j * self.b1b2[:,m,None,None] * ( ((x*y)/r**3) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_13 += ( 1j * self.b1db2[:,m,None,None] * ( (x/r) * M1_scalf_m )[None,...] )[take_dep,...]

                    M1_21 += ( -1j * self.b1b2[:,m,None,None] * ( ((x*y)/r**3) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_22 += ( 1j * self.b1b2[:,m,None,None] * ( (x**2/r**3) * M1_scalf_m )[None,...] )[take_dep,...]
                    M1_23 += ( 1j * self.b1db2[:,m,None,None] * ( (y/r) * M1_scalf_m )[None,...] )[take_dep,...]

                    # M1_31 = 0
                    # M1_32 = 0
                    M1_33 = ( self.b1db1[:,m,None,None] * M1_scalf_m[None,...] )[take_dep,...]

                    #******* compute M2
                    M2_11 += ( 1j * self.b1b2[:,m,None,None] * ( (x**2/r**2) * M2_scalf_m )[None,...] )[take_dep,...]
                    M2_12 += ( 1j * self.b1b2[:,m,None,None] * ( ((x*y)/r**2) * M2_scalf_m )[None,...] )[take_dep,...]
                    # M2_13 = 0

                    M2_21 += ( 1j * self.b1b2[:,m,None,None] * ( ((x*y)/r**2) * M2_scalf_m )[None,...] )[take_dep,...]
                    M2_22 += ( 1j * self.b1b2[:,m,None,None] * ( (y**2/r**2) * M2_scalf_m )[None,...] )[take_dep,...]
                    # M2_23 = 0

                    M2_31 += ( self.b1b1[:,m,None,None] * ( (x/r) * M2_scalf_m )[None,...] )[take_dep,...]
                    M2_32 += ( self.b1b1[:,m,None,None] * ( (y/r) * M2_scalf_m )[None,...] )[take_dep,...]
                    # M2_33 = 0

            if hasattr(self, 'love_included'):
            # Love wave part
                for m in range(self.hm_lov+1):

                    M1_scalf_m = self.cfac_lov[m] * ssp.hankel1(0,self.kmode_lov[m]*r) * 1j * 0.25
                    M2_scalf_m = self.cfac_lov[m] * ssp.hankel1(1,self.kmode_lov[m]*r) * self.kmode_lov[m] * -1j * 0.25

                    if ic==0:
                    # gradient of G_x (first column of G tensor)
                        #******* compute M1
                        M1_11 += ( -1 * self.l1l1[:,m,None,None] * ( (2*x*(y**2)/r**4) * M1_scalf_m )[None,...] )[take_dep,...]
                        M1_12 += ( self.l1l1[:,m,None,None] * ( (2*y*(x**2)/r**4) * M1_scalf_m )[None,...] )[take_dep,...]
                        M1_13 += ( self.l1dl1[:,m,None,None] * ( (y**2/r**2) * M1_scalf_m )[None,...] )[take_dep,...]

                        M1_21 += ( -1 * self.l1l1[:,m,None,None] * ( (y*(y**2-x**2)/r**4) * M1_scalf_m )[None,...] )[take_dep,...]
                        M1_22 += ( -1 * self.l1l1[:,m,None,None] * ( (x*(x**2-y**2)/r**4) * M1_scalf_m )[None,...] )[take_dep,...]
                        M1_23 += ( -1 * self.l1dl1[:,m,None,None] * ( ((x*y)/r**2) * M1_scalf_m )[None,...] )[take_dep,...]

                        #******* compute M2
                        M2_11 += ( self.l1l1[:,m,None,None] * ( ((x*(y**2))/r**3) * M2_scalf_m )[None,...] )[take_dep,...]
                        M2_12 += ( self.l1l1[:,m,None,None] * ( (y**3/r**3) * M2_scalf_m )[None,...] )[take_dep,...]

                        M2_21 += ( -1 * self.l1l1[:,m,None,None] * ( ((y*(x**2))/r**3) * M2_scalf_m )[None,...] )[take_dep,...]
                        M2_22 += ( -1 * self.l1l1[:,m,None,None] * ( ((x*(y**2))/r**3) * M2_scalf_m )[None,...] )[take_dep,...]

                    else:
                        # gradient of G_y NOT REQUIRED, and G_z=0 for the Love wave case
                        pass

            # both parts done, now build the full tensor
            M1 = np.array(([M1_11,M1_12,M1_13],[M1_21,M1_22,M1_23],[M1_31,M1_32,M1_33]))
            M2 = np.array(([M2_11,M2_12,M2_13],[M2_21,M2_22,M2_23],[M2_31,M2_32,M2_33]))

            if ic==1:
                pass
            else:
                # try:
                #     self.gradG[ic] = M1 + M2
                # except IndexError:
                    # self.gradG[crind[ic]] = M1 + M2
                self.gradG[xyz_to_xz[ic]] = M1 + M2

    #********************************************************************************

    def divG_cartesian(self, x, y, r, dimflag):

        self.divG = np.zeros((self.gradG.shape[2:]), dtype='complex')
        for ic in range(len(comps)):
            if ic==1:
                pass
            else:
                # try:
                #     self.divG[ic,...] = self.gradG[ic,0,0,...] + self.gradG[ic,1,1,...] + self.gradG[ic,2,2,...]
                # except IndexError:
                    # self.divG[ic,...] = self.gradG[crind[ic],0,0,...] + self.gradG[crind[ic],1,1,...] + self.gradG[crind[ic],2,2,...]
                self.divG[ic,...] = self.gradG[xyz_to_xz[ic],0,0,...] + self.gradG[xyz_to_xz[ic],1,1,...] + self.gradG[xyz_to_xz[ic],2,2,...]

    #********************************************************************************

########################################################################################
