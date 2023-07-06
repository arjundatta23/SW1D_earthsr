#!/usr/bin/python

""" Standard Python modules """
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

""" Modules written by me """
import read_earthsr_io as reo
# import read_saito_io as rso

######################################################################################################

class show_egn_onemodel:

	""" Class that operates on a single eigenfunction file produced by earth and
	    plots the corresponding eigenfunctions for user-specified modes and periods
	"""

	def __init__(self,flist,m,p,sdcomp,fig_save=False):

		self.egn = {'Ux': 'ur', 'Uz': 'uz', 'Uy': 'ut', 'szz': 'tz', 'sxz': 'tr'}
		egn_name = {'Ux': 'Horizontal displacement', 'Uz': 'Vertical displacement','Uy': 'Transverse displacement', 'szz': 'Normal stress', 'sxz': 'Radial Shear stress','syz': 'Transverse shear stress'}
		self.figtitle = 'Normalized Eigenfunctions - '+egn_name[sdcomp]
		self.allm = m
		self.allp = p
		self.sdvcomp = sdcomp
		if not (sdcomp in self.egn):
				print("Eigenfunction component must be one of the following:")
				for d in egn:
					print(d)
				raise SystemExit("Please try again")
		self.output_save=fig_save
		self.egn_allf=list(range(len(flist)))
		for fn,fl in enumerate(flist):
			self.egn_allf[fn]=self.read_single_file(fl)
		print(m,p)
		[self.max_plotx, self.min_plotx, self.max_ploty] = self.get_plot_axes()
		print("plotting limits xmax xmin ymax: ", self.max_plotx, self.min_plotx, self.max_ploty)
		if len(self.allp)==2:
			self.plotegn_bymode()
		else:
			self.plotegn_byperiod()

	def read_single_file(self,infile):
		egn_allm=[]
		for mode in self.allm:
			egn_onem=[]
			for per in self.allp:
				print("From see_egnfns: ", mode, per)
				try:
					this_mp = reo.read_egnfile([infile],mode,per)
					plot_quantity = getattr(this_mp,self.egn[self.sdvcomp])[0]
				except IndexError:
					# means there is an error in reading by reo.
					# read with rso instead
					print("Failed with reo, trying to read with rso...")
					try:
						this_mp = rso.read_yifile([infile],mode,per)
						plot_quantity = this_mp.modegn[0]
					except IndexError:
						# error reading by rso because mode does not exist at
						# specified period
						print("Mode %d does not exist at period %.1f " %(mode,per))
						plot_quantity=None
				try:
					dep=np.array([x for x,y in plot_quantity])
					egnfn=np.array([y for x,y in plot_quantity])
					egn_onem.append([this_mp.mpp,this_mp.ppp,dep,egnfn])
				except TypeError:
					print("!!!! Came here !!!!!")
					# eigenfunction has not been returned by reo/rso because
					# period is too long for this mode (freq. is below the cut-off)
					pass
			egn_allm.append(egn_onem)
		return egn_allm

	def get_plot_axes(self):

		""" Function to determine the range on the plot axes. X-axis limits are determined
		    by the min and max values of the eigenfunctions, y-axis limit is determined by
		    the depth at which the eigenfunctions for the deepest peneterating component
	            decay to zero
		"""
		modes_minval=[]
		modes_maxval=[]
		modes_grdep=[]
		# x-axis limits
		for egn_mode in self.egn_allf[0]:
		# egn_allf is eigenfunctions for all input files. If using index 0 above, it means x-axis limits decided by 1st input file
		# If using index -1, it means x-axis limits decided by last input file
			try:
				mode_minval = min([min(z) for x,y,d,z in egn_mode])
				mode_maxval = max([max(z) for x,y,d,z in egn_mode])
			except ValueError:
				# mode does not exist
				mode_minval = 0
				mode_maxval = 0
			modes_minval.append(mode_minval)
			modes_maxval.append(mode_maxval)
			# print(egn_mode[-1])
		max_x=math.ceil(max(modes_maxval)*100)/100
		min_x=math.floor(min(modes_minval)*100)/100
		# y-axis limits
		for egn_mode in self.egn_allf[0]:
			# print("max_x is: ", max_x)
			mode_relindex = np.where(abs(egn_mode[0][3])<(max_x*1e-03))[0]
			# first index of egn_mode is the index of the frequency considered for looking at
			# the depth peneteration
			# print("mode_relindex is: ", mode_relindex)
			try:
				gr_dep = egn_mode[-1][2][mode_relindex[1]]
			except IndexError:
				gr_dep = 2500
			print("Mode: %d Period: %f Greatest depth: %f" %(egn_mode[-1][0],egn_mode[-1][1],gr_dep))
			modes_grdep.append(gr_dep)
		max_dep=max(modes_grdep)
		if max_dep>100:
			max_y = max_dep #+ (100-(max_dep%100))
		elif max_dep<1:
			max_y = max_dep + (0.1-(max_dep%0.1))
		else:
			max_y = max_dep
		return max_x, min_x, max_y

	def plotegn_bymode(self):

		""" Function that produces mode-wise plots
		"""
		flname=['Saito','Earth']
		#flname=['Background Medium','Perturbed Region']
		fig=plt.figure() #(figsize=(15,10))
		mfigs=list(range(len(self.egn_allf[0])))
		for j,egnfnum in enumerate(self.egn_allf):
			for i,egn_mode in enumerate(egnfnum):
				if j==0:
				# the first input file
					mode_num = egn_mode[0][0]
					caption = "Mode %d" %(mode_num)
					mfigs[i]=fig.add_subplot(1,len(egnfnum),i+1)
					mfigs[i].set_title(caption)
					#mfigs[i].yaxis.grid(True)
					mfigs[i].axvline(x=0,color='k',ls='--',lw=0.5)
				if i==0 and j==0: mfigs[i].set_ylabel("Depth [km]") #, labelpad=20)
				for egn_p in egn_mode:
					egn_values=egn_p[3]
					egn_depth = egn_p[2]
					curve_name = "%d s" %(int(egn_p[1]))
					if len(self.egn_allf)==1:
						mfigs[i].plot(egn_values,egn_depth,label=curve_name)
					else:
						mfigs[i].plot(egn_values,egn_depth,label=flname[j])
					# print(len(egn_values), len(egn_depth))
					mfigs[i].set_ylim([1500,0])
					#mfigs[i].set_ylim([self.max_ploty,0])
					mfigs[i].set_xlim([self.min_plotx,self.max_plotx])
				if j==0:
					def_xtics = mfigs[i].get_xticks()
					show_xtics = [ x for k,x in enumerate(def_xtics) if k%3==0 ]
					mfigs[i].set_xticks(show_xtics)
					def_ytics = mfigs[i].get_yticks()
					print(def_ytics)
				if i>0: mfigs[i].set_yticklabels([])
				l=mfigs[i].legend(loc=4)
				l.draw_frame(False)
			if len(self.egn_allf)==1:
				#fig.suptitle(self.figtitle)
				pass
			else:
				ft="Period "+curve_name
				#fig.suptitle(ft)
		if self.output_save: plt.savefig('egn_mwise_'+self.name+'.png')
		else: plt.show()

	def plotegn_byperiod(self):

		""" Function that produces period-wise plots
		"""
		#flname=['Saito','Earth']
		flname=['Background','Perturbation','dist 1','dist 2','dist 3','dist 4']
		#flname=['Medium 1','Medium 2']
		#flname=['Layer 4 km','Layer 16 km', 'Layer 28 km', 'Layer 40 km']

		# first need to reorder the list containing the eigenfunctions
		#  so that they are listed period-wise rather than mode-wise
		p_funda = [y for x,y,d,z in self.egn_allf[0][0]]
		print("pfunda is ", p_funda, len(self.egn_allf))
		self.egn_allfp=list(range(len(self.egn_allf)))
		for i in range(len(self.egn_allf)):
			egn_allp=[]
			for per in p_funda:
				egn_onep=[]
				for egn_m in self.egn_allf[i]:
					p_thism = [x for x in egn_m if x[1]==per]
					if len(p_thism) != 0: egn_onep.append(p_thism[0])
				egn_allp.append(egn_onep)
			self.egn_allfp[i]=egn_allp
		print("No. of periods is ", len(self.egn_allfp[0]))
		# Ask user if he would like any shading across a depth range
		shading = 'n'
		shading=input("Shade any part of figure ? (y/n): ")
		if shading=='y':
			dboth=input("Enter top and bottom depths: ")
			d1=float(dboth.split()[0])
			d2=float(dboth.split()[1])
		# then do the plotting
		pfigs=list(range(len(self.egn_allfp[0])))
		fig=plt.figure() #(figsize=(15,10))
		for j,egnfnum in enumerate(self.egn_allfp):
			for i,egn_per in enumerate(egnfnum):
				if j==0:
				# the first  input file
					pfigs[i]=fig.add_subplot(1,len(egnfnum),i+1)
					period = egn_per[0][1]
					#caption = "%.3f Hz" %(1/float(period))
					caption = "%d s" %(int(period))
					pfigs[i].set_title(caption)
					if i==0: pfigs[i].set_ylabel("Depth [km]") #, labelpad=30)
					#pfigs[i].yaxis.grid(True,which='both')
					pfigs[i].axvline(x=0,color='k',ls='--',lw=0.5)
				for egn_m in egn_per:
					egn_depth=egn_m[2]
					egn_values=egn_m[3]
					curve_name = "Mode %d" %(int(egn_m[0]))
					if len(self.egn_allfp)==1:
						pfigs[i].plot(egn_values,egn_depth,label=curve_name)
					else:
						if j==0 or j==(len(self.egn_allfp)-1):
							pfigs[i].plot(egn_values,egn_depth,label=flname[j])
						else:
							pfigs[i].plot(egn_values,egn_depth,label=flname[j])
							#pfigs[i].plot(egn_values,egn_depth,'--',label=flname[j])
					if shading=='y':
						pfigs[i].axhspan(d1, d2, facecolor='0.5', alpha=0.1)
					#pfig.set_ylim([self.max_ploty,0])
					pfigs[i].set_ylim(200,0)
					#pfigs[i].set_ylim(self.max_ploty,0)
					pfigs[i].set_xlim([self.min_plotx,self.max_plotx])
				if j==0:
					def_xtics = pfigs[i].get_xticks()
					show_xtics = [ x for k,x in enumerate(def_xtics) if k%3==0 ]
					def_ytics = pfigs[i].get_yticks()
					minortics = np.zeros(def_ytics.size-1)
					for t,y in enumerate(def_ytics):
						if not t==def_ytics.size-1:
							minortics[t]=(y+def_ytics[t+1])/2
					pfigs[i].set_yticks(minortics,minor=True)
					pfigs[i].set_xticks(show_xtics)
				if i>0: pfigs[i].set_yticklabels([])
				l=pfigs[i].legend(loc=4,prop={'size':12})
				l.draw_frame(False)
		if len(self.egn_allf)==1:
			#fig.suptitle(self.figtitle)
			pass
		else:
			ft=curve_name
			#fig.suptitle(ft)
		if self.output_save: plt.savefig('egn_pwise_'+self.name+'.png')
		else: plt.show()

######################## Main body of program  ###############################################

nfiles=len(sys.argv)
filenums=range(1,nfiles)
xnames=[]
for fn in filenums:
	xnames.append(sys.argv[fn])

# eigenfile=sys.argv[1]
modes = input("Enter modes (0 for fundamental, 1 for first overtone and so on): ")
freqs = input("Enter frequencies (in Hz): ")
quantity = input("Enter component of stress-displacement vector you wish to see (Ux,Uy,Uz,szz,sxz,syz): ")
modes = [int(x) for x in modes.split()]
periods = [ float('%.4f' %(1/float(y))) for y in freqs.split()]
print("periods are ", periods)
station_egn = show_egn_onemodel(xnames,modes,periods,quantity)
