# -*- coding: utf-8 -*-
"""
Nipype stroke resting state preprocessing script

Created on Wed Mar  2 08:59:09 2016

Peter Goodin

This script takes the stroke data from the Connect and Prepare studies and runs 
a resting state analysis cleaning and normalisation regime specialised for stroke
data. 

v3. 
*Made seperate mask calcs for WM and CSF. CSF didn't survive after erosion for
some subs. Upped threshold for csf to 1
*Full preproc script.

v4.
*Removed "manual" SVD of noise vars in favour of using code from sklearn
(validated)

v5.
*Outputs global and noglobal signal files, makes filtered and non-filtered
versions (filtered for connectivity, non-filtered for ALFF / FALFF)
NOTE - ReHo images can be collected from either the warped EPIs or the warped 
non-filtered EPIs.

Readded manual SVD. Quicker, results identical, better control. 

v6.
*Changed order from segment > coregister > make masks to segment > make masks > coregister
Helped remove problems with participants with small ventricles having 0 voxels
for CSF after thresh + ero. Changed thresh to .99 + added 2nd erosion

v7. 

Added 1% STD signal regressor as an option...

v8.
Added FFT filter (code from nipype resting state script)
Changed erosion from FSL to custom function using scipy.ndimage
(faster, more control) 

v9.
Added new erosion algorithm (scipy.ndimage) - faster than FSL erosion + more
control on erosion properties. 

Split WM + CSF into two compcor calls with 5 components each (same as CONN toolbox).
Thresh @ .99 with 2 erosions using a 27 voxel structure element.

Outputs correlation matrices from the AAL and Gordon (2014) atlases for further
analysis.

v10. 

Split motion "correction" from CompCor to seperate function (reason: assess impact of MC on original data).
Removed creation of subject noise masks node because...
Normalisation after MC, before CompCor.
Using white and csf MNI space masks for noise mask (eroded).


"""


####Import####
from __future__ import division
from nipype.interfaces import spm, dcmstack, ants, afni
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.algorithms import misc
from nipype.pipeline.engine import Workflow, Node, MapNode
import multiprocessing
import os
import time


start = time.time()
#Set up directories and info

rawdir = '/home/peter/Desktop/test/stroke/raw/'
writedir = '/home/peter/Desktop/test/dummydata/preproc_v10/'
workdir = writedir + 'working/'
crashdir=writedir + 'crash/'
outdir = writedir + 'output/'




try:
    os.mkdir(outdir)
except:
    print 'Outdir: ' + outdir + ' already exists. Not creating new folder' 

try:
    os.mkdir(crashdir)
except:
    print 'Crashdir: ' + crashdir + ' already exists. Not creating new folder' 
    

os.chdir(crashdir)


###SETTINGS###

n_cores = multiprocessing.cpu_count()-1 #Note, takes the maximum available cores leaves 1 for OS. 

#Select files
template={'anat':rawdir + '{subject_id}/*t1*/*.dcm',
          'epi':rawdir + '{subject_id}/*RESTING+*/*.dcm',
          'flair':rawdir + '{subject_id}/flair/*.nii',
          'mask':rawdir + '{subject_id}/mask/*.nii',
          'wm_noise': '/home/peter/Desktop/test/templates/t1_3mm_wm_ero.nii',
          'csf_noise': '/home/peter/Desktop/test/templates/t1_3mm_csf_ero.nii',
          'mni_template':'/home/peter/Desktop/test/templates/template_3mm_brain.nii'}

#Smoothing kernal
fwhm=6


#Get participant list
subject_list = os.listdir(rawdir)
subject_list.sort()

#Place custom functions here

def metaread(nifti):
    """
    Combines metadata read from the header, populates the SPM slice timing
    correction inputs and outputs the time corrected epi image.
    Uses dcmstack.lookup to get TR and slice times, and NiftiWrapper to get
    image dimensions (number of slices is the z [2]).
    """
    from nipype.interfaces import dcmstack
    from dcmstack.dcmmeta import NiftiWrapper
    nii=NiftiWrapper.from_filename(nifti)
    imdims=nii.meta_ext.shape
    sliceno=imdims[2]
    lookup=dcmstack.LookupMeta()
    lookup.inputs.meta_keys={'RepetitionTime':'TR','CsaImage.MosaicRefAcqTimes':'ST'}
    lookup.inputs.in_file=nifti
    lookup.run()
    slicetimes=[int(lookup.result['ST'][0][x]) for x in range(0,imdims[2])] #Converts slice times to ints. 
    tr=lookup.result['TR']/1000 #Converts tr to seconds.
    ta=tr-(tr/sliceno)
    return (sliceno, slicetimes, tr, ta)
           
metadata=Node(Function(function=metaread,input_names=['nifti'],output_names=['sliceno', 'slicetimes', 'tr', 'ta']),name='metadata')
#Outputs: tr, slicetimes, imdims

   
def voldrop(epilist):
    """
    Drops volumes > nvols.
    """
    import numpy as np
    import os
    nvols=140 #<--------See if there's a way to call a variable outside of a function as input for the function
    vols=len(epilist)
    if vols>nvols:
        epilist=epilist[0:nvols]
    volsdropped=[vols-nvols]
    print 'Dropped ' + str(volsdropped) + ' volumes.'
    volsdropped_filename=os.path.join(os.getcwd(),'volsdropped.txt')
    np.savetxt(volsdropped_filename,volsdropped,fmt="%.5f",delimiter=',')
    
    return (epilist,volsdropped_filename)         

dropvols=Node(Function(function=voldrop,input_names=['epilist'],output_names=['epilist','volsdropped_filename']),name='dropvols')
#Outputs: epilist 



def get_flair_2_anat_files(flair,flairmask, seg):
    """
    Makes a list of outputs from the FLAIR / mask to T1 coregistration 
    and segmented T1 to be passed to the coregistration from T1 to EPI. 
    """
    gm = seg[0]
    wm = seg[1]
    csf = seg[2]
    flair = flair
    flairmask = flairmask
    anat_coreglist = [gm, wm, csf, flair, flairmask]
    return (flair, flairmask, anat_coreglist)

flair2anat_list=Node(Function(function=get_flair_2_anat_files,input_names=['seg','flair','flairmask'],output_names=['flair','flairmask','anat_coreglist']),name='flair2anat_list')


def get_anat_2_epi_files(coreg_files,source):
    """
    Makes a list of outputs from the T1 to EPI coregistration. 
    """
    
    source = source
    gm = coreg_files[0]
    wm = coreg_files[1]
    csf = coreg_files[2]
    flair = coreg_files[3]
    flairmask = coreg_files[4]

    return (source, gm, wm, csf, flair, flairmask)

anat2epi_list=Node(Function(function=get_anat_2_epi_files,input_names=['coreg_files','source'],output_names=['source', 'gm', 'wm', 'csf','flair', 'flairmask']),name='anat2epi_list')
#Outputs: gm, wm, csf, coregistered source image


def calc_mmask(gm,wm,csf):
    """
    Calculates participant specific brain mask using gm, wm and csf co-registered output.
    """
    import nibabel as nb
    import os
    from scipy.ndimage import binary_fill_holes as bfh    
    
    gm_mask=nb.load(gm)
    wm_mask=nb.load(wm)
    csf_mask=nb.load(csf)
    m_mask=gm_mask.get_data()+wm_mask.get_data()+csf_mask.get_data()
    m_mask[m_mask>.1]=1
    m_mask=bfh(m_mask)
    brain_ss=gm_mask.get_data()+wm_mask.get_data()
    img1 = nb.Nifti1Image(m_mask, header=gm_mask.get_header(), affine=gm_mask.get_affine())
    filename1 = os.path.join(os.getcwd(),'mmask.nii')
    img1.to_filename(filename1)
    img2 = nb.Nifti1Image(brain_ss, header=gm_mask.get_header(), affine=gm_mask.get_affine())
    filename2 = os.path.join(os.getcwd(),'brain_ss.nii')
    img2.to_filename(filename2)
    return (filename1,filename2)

mmaskcalc=Node(Function(function=calc_mmask,input_names=['gm','wm','csf'],output_names=['filename1','filename2']),name='m_mask')
#Outputs: binarised "matter" mask


def mc(epifile, mmaskfile, motionparams):
    """
    Uses the Friston 24 motion parameters to "correct" for motion.
    """
    
    import nibabel as nb
    import numpy as np
    import os
    
    epi=nb.load(epifile) #Needed for header and affine info
    data=epi.get_data()
    global_mask=nb.load(mmaskfile).get_data()
    global_mask[np.isnan(global_mask)]=0
    global_mask=global_mask>0

    motion=np.genfromtxt(motionparams)
    
    #CALCULATE FRISTON 24 MODEL (6 motion params + preceeding vol + each values squared.)
    motion_squared = motion ** 2
    new_motion = np.concatenate((motion, motion_squared), axis=1)
    motion_roll = np.roll(motion, 1, axis=0)
    motion_roll[0] = 0
    new_motion = np.concatenate((new_motion, motion_roll), axis=1)
    motion_roll_squared = motion_roll ** 2
    motion24 = np.concatenate((new_motion, motion_roll_squared), axis=1)
    
    X = motion24
    Y = data[global_mask].T
    
    Ym = np.mean(Y, axis = 0)
    
    B = np.linalg.lstsq(X, Y)[0] #Calculate least squares
    Y_resid = Y - X.dot(B) #Regresses nuisance from data
    
    Y_resid = Y + Ym #Adds mean back to data
    
    data[global_mask] = Y_resid.T
    data = data*(np.repeat(global_mask[:,:,:,np.newaxis], repeats = np.shape(data)[3],axis = 3)) #Zeros everything outside brain mask.
    
    img = nb.Nifti1Image(data, header = epi.get_header(), affine = epi.get_affine())
    mc_filename = os.path.join(os.getcwd(), 'motion_corrected.nii')
    img.to_filename(mc_filename) 
    
    return(mc_filename)

motion_correction=Node(Function(function=mc,input_names=['epifile','mmaskfile', 'motionparams'],output_names=['mc_filename']),name='motion_correction')

def make_noise_masks(wm_in,csf_in,mask_in):
    """
    Creates noises masks to be used for compcor. Inputs are the template wm, csf images
    and the lesion mask (for cleaning wm). 
    """
    import numpy as np
    import nibabel as nb
    import os
	
    csfinfo=nb.load(csf_in)
    wminfo=nb.load(wm_in)
    
    maskdata = nb.load(mask_in).get_data() == 255 #Invert image so lesion region is removed from noise masks.
    maskdata = np.invert(maskdata)

    csfinfo=nb.load(csf_in)
    wminfo=nb.load(wm_in)

    csfdata=csfinfo.get_data()>0.90
    wmdata=wminfo.get_data()>0.90

    csfdata[np.isnan(csfdata)==1]=0
    wmdata[np.isnan(wmdata)==1]=0

    csf_trim = csfdata*maskdata
    wm_trim = wmdata*maskdata
    
    wm_img = nb.Nifti1Image(wm_trim, header = wminfo.get_header(), affine = wminfo.get_affine())
    csf_img = nb.Nifti1Image(csf_trim, header = csfinfo.get_header(), affine = csfinfo.get_affine())
    wmmask_file=os.path.join(os.getcwd(),'wm_trim.nii')
    csfmask_file=os.path.join(os.getcwd(),'csf_trim.nii')
    wm_img.to_filename(wmmask_file)
    csf_img.to_filename(csfmask_file)
    
    return(wmmask_file,csfmask_file)

noisemask=Node(Function(function=make_noise_masks,input_names=['wm_in','csf_in','mask_in'],output_names=['wmmask_file','csfmask_file']),name='noisemask')



def data_clean(epi_file, mmask_file, wmnoise_file, csfnoise_file):
    """
    Regresses out noise time series using the aCompCor method (Behzadi et al. (2007)
    Seperate WM and CSF components (5 a piece, similar to CONN) are used as the noise signal
    Scrubbing is not done (see Muschelli et al, 2014).
    Global signal is the mean signal of a whole brain (GM, WM, CSF) mask.
    Output is residuals with global signal NOT removed and global removed.  
    """
    import nibabel as nb
    import numpy as np
    import os
    from scipy.signal import detrend
    
    epi=nb.load(epi_file) #Needed for header and affine info
    data=epi.get_data()
    wm_mask=nb.load(wmnoise_file).get_data()
    wm_mask[np.isnan(wm_mask)]=0
    wm_mask=wm_mask>0
    csf_mask=nb.load(csfnoise_file).get_data()
    csf_mask[np.isnan(csf_mask)]=0
    csf_mask=csf_mask>0
    global_mask=nb.load(mmask_file).get_data()
    global_mask[np.isnan(global_mask)]=0
    global_mask=global_mask>0
    
    wm_ts=data[wm_mask].T
    csf_ts=data[csf_mask].T
    
    #Remove constant and linear trends from WM
    Ycon_wm=detrend(wm_ts, axis=0, type='constant')
    Ylin_wm=detrend(Ycon_wm, axis=0, type='linear')
    
    #Normalise variance
    Yn_wm=(Ylin_wm-np.mean(Ylin_wm,axis=0))/np.std(Ylin_wm,axis=0)
    dropped=Yn_wm.shape[1]
    
    #Converts nan values to 0
    Yn_wm[np.isnan(Yn_wm)==1]=0    
    
    #Remove 0 variance time series
    Yn_wm=Yn_wm[:,np.std(Yn_wm,axis=0)!=0]
    wm_dropts=dropped-Yn_wm.shape[1]
    print 'Dropped '+ str(wm_dropts)+' WM time series'
    
    wmdrop_filename=os.path.join(os.getcwd(),'wm_drop.txt')
    np.savetxt(wmdrop_filename,[wm_dropts],delimiter=',')
    
    #Compute SVD
    print 'Calculating SVD decomposition.'
    u_wm,s_wm,v_wm=np.linalg.svd(Yn_wm)
    var_wm=(s_wm**2/np.sum(s_wm**2))*100
    cumvar_wm=np.cumsum(s_wm**2)/np.sum(s_wm**2)*100
    
    #figure();plot(cumvar);title('Cumulative Variance')
    var_filename_wm=os.path.join(os.getcwd(),'variance_exp_wm.txt')
    cumvar_filename_wm=os.path.join(os.getcwd(),'cum_variance_exp_wm.txt')
    np.savetxt(var_filename_wm,var_wm,delimiter=',')
    np.savetxt(cumvar_filename_wm,cumvar_wm,delimiter=',')
    print 'File written to '+var_filename_wm
    print 'File written to '+cumvar_filename_wm
    
    #Get components of interest
    nComp=5 #Number of components
    
    comps_wm=u_wm[:,:nComp]
    
     
    #Remove constant and linear trends from CSF
    Ycon_csf=detrend(csf_ts, axis=0, type='constant')
    Ylin_csf=detrend(Ycon_csf, axis=0, type='linear')
    
    #Normalise variance
    Yn_csf=(Ylin_csf-np.mean(Ylin_csf,axis=0))/np.std(Ylin_csf,axis=0)

    dropped=Yn_csf.shape[1]
    
    #Converts nan values to 0
    Yn_csf[np.isnan(Yn_csf)==1]=0      
    
    #Remove 0 variance time series
    Yn_csf=Yn_csf[:,np.std(Yn_csf,axis=0)!=0]
    csf_dropts=dropped-Yn_csf.shape[1]
    print 'Dropped '+ str(csf_dropts)+' CSF time series'
    
    csfdrop_filename=os.path.join(os.getcwd(),'csf_drop.txt')
    np.savetxt(csfdrop_filename,[csf_dropts],delimiter=',')
    
    
    #Compute SVD
    print 'Calculating SVD decomposition.'
    u_csf,s_csf,v_csf=np.linalg.svd(Yn_csf)
    var_csf=(s_csf**2/np.sum(s_csf**2))*100
    cumvar_csf=np.cumsum(s_csf**2)/np.sum(s_csf**2)*100
    
    #figure();plot(cumvar);title('Cumulative Variance')
    var_filename_csf=os.path.join(os.getcwd(),'variance_exp_csf.txt')
    cumvar_filename_csf=os.path.join(os.getcwd(),'cum_variance_exp_csf.txt')
    np.savetxt(var_filename_csf,var_csf,delimiter=',')
    np.savetxt(cumvar_filename_csf,cumvar_csf,delimiter=',')
    print 'File written to '+var_filename_csf
    print 'File written to '+cumvar_filename_csf
    
    #Get components of interest
    nComp=5 #Number of components
    
    comps_csf=u_csf[:,:nComp]    
             
    X1=[]
    B1=[]
    Y1=[]
    Y_resid1=[]      
    
    X1=np.column_stack((comps_wm,comps_csf)) #Build regressors file

    X1n=(X1-np.mean(X1,axis=0))/np.std(X1,axis=0) #Normalise regressors
    
    Y1=data[global_mask].T
    #Y1=detrend(Y1,axis=0,type='linear') #Remove linear trend of TS

    B1=np.linalg.lstsq(X1n,Y1)[0]
    Y_resid1=Y1-X1n.dot(B1) #Regresses nuisance from data
    
    regsglobal_filename=os.path.join(os.getcwd(),'global_regressors.txt')
    np.savetxt(regsglobal_filename,X1,fmt="%.5f",delimiter=',')
    print 'File written to '+regsglobal_filename
    
    data[global_mask]=Y_resid1.T
    data=data*(np.repeat(global_mask[:,:,:,np.newaxis],repeats=np.shape(data)[3],axis=3)) #Zeros everything outside brain mask.
    img=nb.Nifti1Image(data, header=epi.get_header(), affine=epi.get_affine())
    file_global=os.path.join(os.getcwd(),'residuals_global.nii')
    img.to_filename(file_global) 
    
    #REMOVING GLOBAL
    
    global_ts=data[global_mask].T
    data=epi.get_data() #Reload data
    
    X2=[]
    B2=[]
    Y2=[]
    Y_resid2=[]      
    
    X2=np.column_stack((comps_wm,comps_csf,np.mean(global_ts,axis=1))) #Build regressors file
      
    X2n=(X2-np.mean(X2,axis=0))/np.std(X2,axis=0) #Voxel wise variance normalise
    
    Y2=data[global_mask].T
    Y2=detrend(Y1,axis=0,type='linear') #Remove linear trend
    
    B2=np.linalg.lstsq(X2n,Y2)[0]
    Y_resid2=Y2-X2n.dot(B2) #Regresses nuisance from data
    
    regsnoglobal_filename=os.path.join(os.getcwd(),'noglobal_regressors.txt')
    np.savetxt(regsnoglobal_filename,X2,fmt="%.5f",delimiter=',')
    print 'File written to '+regsnoglobal_filename
    
    data[global_mask]=Y_resid2.T
    data=data*(np.repeat(global_mask[:,:,:,np.newaxis],repeats=np.shape(data)[3],axis=3)) #Zeros everything outside brain mask.
    img = nb.Nifti1Image(data, header=epi.get_header(), affine=epi.get_affine())
    file_noglobal=os.path.join(os.getcwd(),'residuals_noglobal.nii')
    img.to_filename(file_noglobal)

    return(wmdrop_filename,csfdrop_filename,var_filename_wm,cumvar_filename_wm,var_filename_csf,cumvar_filename_csf,file_noglobal,file_global,regsglobal_filename,regsnoglobal_filename)
    
compcor_clean=Node(Function(function=data_clean,input_names=['epi_file','mmask_file','wmnoise_file','csfnoise_file'],output_names=['wmdrop_filename','csfdrop_filename','var_filename_wm','cumvar_filename_wm','var_filename_csf','cumvar_filename_csf','file_noglobal','file_global','regsglobal_filename','regsnoglobal_filename']),name='compcor_clean')


def get_clean_files(file1,file2):

    """
    Makes a list of the output files from compcor to pass to other functions.
    """
    clean_global=file1
    clean_noglobal=file2
    cleaned=[clean_global, clean_noglobal]
    return (cleaned)

clean_list=Node(Function(function=get_clean_files,input_names=['file1','file2'],output_names=['cleaned']),name='clean_list')


def bandpass_filter(in_file,brainmask):
    """Bandpass filter the input files

    Parameters
    ----------
    files: list of 4d nifti files
    lowpass_freq: cutoff frequency for the low pass filter (in Hz)
    highpass_freq: cutoff frequency for the high pass filter (in Hz)
    fs: sampling rate (in Hz)
    """
    import nibabel as nb
    import numpy as np
    import os
    
    lowpass_freq=0.08
    highpass_freq=0.01
    fs=1/3.0

    img = nb.load(in_file)
    global_mask=nb.load(brainmask).get_data().astype(bool)
    timepoints = img.shape[-1]
    F = np.zeros((timepoints))
    lowidx = timepoints/2 + 1
    if lowpass_freq > 0:
        lowidx = np.round(float(lowpass_freq) / fs * timepoints)
    highidx = 0
    if highpass_freq > 0:
        highidx = np.round(float(highpass_freq) / fs * timepoints)
    F[highidx:lowidx] = 1
    F = ((F + F[::-1]) > 0).astype(int)
    data = img.get_data()
    filter_data=data[global_mask].T
    if np.all(F == 1):
        data[global_mask]=filter_data.T
    else:
        filter_data = np.real(np.fft.ifftn(np.fft.fftn(filter_data)*F[:,np.newaxis]))
        data[global_mask]=filter_data.T
    img_out = nb.Nifti1Image(data, img.get_affine(),img.get_header())
                             
    out_file=os.path.join(os.getcwd(),'bp_'+in_file.split('/')[-1])
    img_out.to_filename(out_file)
  
    return (out_file)
    
bpfilter=MapNode(Function(function=bandpass_filter,input_names=['in_file','brainmask'],output_names=['out_file']),iterfield='in_file',name='bpfilter')


def get_ants_files(ants_output):

    """
    Gets output from ANTs to pass to normalising all the things. 
    """
    trans=[ants_output[0],ants_output[1]]
    return (trans)

ants_list=Node(Function(function=get_ants_files,input_names=['ants_output'],output_names=['trans']),name='ants_list')
#Outputs: transformation matrix, inverse image

def smoothing_files(list1,list2,list3):

    """
    Makes a list of the filtered, non-filtered, global, no-global and non-cleaned files
    for smoothing
    """
    smoothing_files=list1+list2+[list3]
    print smoothing_files
    return (smoothing_files)

smooth_list=Node(Function(function=smoothing_files,input_names=['list1','list2','list3'],output_names=['smoothing_files']),name='smooth_list')
#smooth_list=Node(Function(function=smoothing_files,input_names=['list1'],output_names=['smoothing_files']),name='smooth_list')

def mmask_files(filename1,filename2):

    """
    Makes a list of the filtered, non-filtered, global, no-global and non-cleaned files
    for smoothing
    """
    mmask_files=[filename1,filename2]
    print mmask_files
    return (mmask_files)

mmask_list=Node(Function(function=mmask_files,input_names=['filename1','filename2'],output_names=['mmask_files']),name='mmask_list')


def make_aal_corrmat(smoothed_files):
    """
    Reads in a merged version of the AAL atlas and
    calculates the correlation matrix of all regions.
    Outputs both transformed and non-transformed versions. 
    """
    import nibabel as nb
    import numpy as np
    import os
    aalatlas=nb.load('/home/peter/Desktop/test/templates/aal_pa_3mm.nii').get_data()
    
    glob_data=nb.load([s for s in smoothed_files if "sbp_residuals_global.nii" in s][0]).get_data()
    noglob_data=nb.load([s for s in smoothed_files if "sbp_residuals_noglobal.nii" in s][0]).get_data()
    noclean_data=nb.load([s for s in smoothed_files if "smotion_corrected_trans.nii" in s][0]).get_data()
    
    #Pre-allocate regional ts matrix
    aalatlas_ts_glob=np.zeros([glob_data.shape[3],len(np.unique(aalatlas))-1])
    aalatlas_ts_noglob=np.zeros([noglob_data.shape[3],len(np.unique(aalatlas))-1])
    aalatlas_ts_noclean=np.zeros([noclean_data.shape[3],len(np.unique(aalatlas))-1])
    
    #Loop through unique values (skipping background, 0), populate with mean regional ts.
    for x in range(1,len(np.unique(aalatlas))):
        roi=np.squeeze(aalatlas==x)
        aalatlas_ts_glob[:,x-1]=np.mean(glob_data[roi].T,axis=1)
        aalatlas_ts_noglob[:,x-1]=np.mean(noglob_data[roi].T,axis=1)
        aalatlas_ts_noclean[:,x-1]=np.mean(noclean_data[roi].T,axis=1)
        
    #Run correlations     
    glob_corrmat=np.corrcoef(aalatlas_ts_glob.T)
    noglob_corrmat=np.corrcoef(aalatlas_ts_noglob.T)
    noclean_corrmat=np.corrcoef(aalatlas_ts_noclean.T)
    
    #Save data as csv files.
    #Save global
    file_global=os.path.join(os.getcwd(),'global_correlation_aal.csv')
    np.savetxt(file_global,glob_corrmat,fmt="%.5f",delimiter=',')
    file_global_trans=os.path.join(os.getcwd(),'global_correlation_aal_trans.csv')
    np.savetxt(file_global_trans,np.arctanh(glob_corrmat),fmt="%.5f",delimiter=',')
    
    #Save no global
    file_noglobal=os.path.join(os.getcwd(),'noglobal_correlation_aal.csv')
    np.savetxt(file_noglobal,noglob_corrmat,fmt="%.5f",delimiter=',')
    file_noglobal_trans=os.path.join(os.getcwd(),'noglobal_correlation_aal_trans.csv')
    np.savetxt(file_noglobal_trans,np.arctanh(noglob_corrmat),fmt="%.5f",delimiter=',')
    
    #Save no clean - Baseline
    file_noclean=os.path.join(os.getcwd(),'noclean_correlation_aal.csv')
    np.savetxt(file_noclean,noclean_corrmat,fmt="%.5f",delimiter=',')
    

    return(file_global,file_global_trans,file_noglobal,file_noglobal_trans,file_noclean)

aal_corrmat=Node(Function(function=make_aal_corrmat,input_names=['smoothed_files'],output_names=['file_global','file_global_trans','file_noglobal','file_noglobal_trans','file_noclean']),name='aal_corrmat')

####Nipype script begins below####






#Set up iteration over subjects
infosource=Node(IdentityInterface(fields=['subject_id']),name='infosource')
infosource.iterables=('subject_id',subject_list)

selectfiles=Node(SelectFiles(template),name='selectfiles')
selectfiles.inputs.base_directory=rawdir
selectfiles.inputs.sort_files=True
#Outputs: anat, epi, flair, mask, wm_noise, csf_noise, mni_template

####EPI preprocessing####

#Convert EPI dicoms to nii (with embeded metadata)
epi_stack=Node(dcmstack.DcmStack(),name='epistack')
epi_stack.inputs.embed_meta=True
epi_stack.inputs.out_format='epi'
epi_stack.inputs.out_ext='.nii'
#Outputs: out_file

#Despiking using afni (position based on Jo et al. (2013)).
despike=Node(afni.Despike(),name='despike')
despike.inputs.outputtype='NIFTI'
#Outputs: out_file

#Slice timing corrected (gets timing from header)
st_corr=Node(spm.SliceTiming(),name='slicetiming_correction')
st_corr.inputs.ref_slice=1
#Outputs: timecorrected_files

#Realignment using SPM <--- Maybe just estimate and apply all transforms at the end?
realign=Node(spm.Realign(),name='realign')
realign.inputs.register_to_mean=False
realign.inputs.quality=1.0
#Outputs: realignment_parameters, reliced epi images (motion corrected)

tsnr=Node(misc.TSNR(),name='tsnr')
tsnr.inputs.regress_poly=2
#Outputs: detrended_file, mean_file, stddev_file, tsnr_file


smooth=Node(spm.Smooth(),name='smooth')
smooth.inputs.fwhm=fwhm


####Anatomical preprocessing####


#dcmstack - Convert dicoms to nii (with embeded metadata)
anat_stack=Node(dcmstack.DcmStack(),name='anatstack')
anat_stack.inputs.embed_meta=True
anat_stack.inputs.out_format='anat'
anat_stack.inputs.out_ext='.nii'
#Outputs: out_file

#Coregisters FLAIR & mask to T1 (NOTE: settings taken from Clinical Toolbox)
flaircoreg=Node(spm.Coregister(),name='coreg2anat')
flaircoreg.inputs.cost_function='nmi'
flaircoreg.inputs.separation=[4,2]
flaircoreg.inputs.tolerance=[0.02,0.02,0.02,0.001,0.001,0.001,0.01,0.01,0.01,0.001,0.001,0.001]
flaircoreg.inputs.fwhm = [7,7]
flaircoreg.inputs.write_interp=1
flaircoreg.inputs.write_wrap=[0,0,0]
flaircoreg.inputs.write_mask=False

#Coregisters T1, FLAIR + mask to EPI (NOTE: settings taken from Clinical Toolbox)
coreg=MapNode(spm.Coregister(),iterfield='apply_to_files',name='coreg2epi')
coreg.inputs.cost_function='nmi'
coreg.inputs.separation=[4,2]
coreg.inputs.tolerance=[0.02,0.02,0.02,0.001,0.001,0.001,0.01,0.01,0.01,0.001,0.001,0.001]
coreg.inputs.fwhm = [7,7]
coreg.inputs.write_interp=1
coreg.inputs.write_wrap=[0,0,0]
coreg.inputs.write_mask=False
#Output: coregistered_files

#Segment anatomical
seg=Node(spm.NewSegment(),name='segment')
#Outputs: 

#Warps to MNI space using a 3mm template image
#Note - The template is warped to subj space (with mask as 
#cost function region) then the inverse transform (subj space > MNI) is used
#to warp the data.
antsnorm=Node(ants.Registration(),name='antsnorm')
antsnorm.inputs.output_transform_prefix = "new"
antsnorm.inputs.collapse_output_transforms=True
antsnorm.inputs.initial_moving_transform_com=True
antsnorm.inputs.num_threads=1
antsnorm.inputs.output_inverse_warped_image=True
antsnorm.inputs.output_warped_image=True
antsnorm.inputs.sigma_units=['vox']*3
antsnorm.inputs.transforms=['Rigid', 'Affine', 'SyN']
antsnorm.inputs.terminal_output='file'
antsnorm.inputs.winsorize_lower_quantile=0.005
antsnorm.inputs.winsorize_upper_quantile=0.995
antsnorm.inputs.convergence_threshold=[1e-06]
antsnorm.inputs.convergence_window_size=[10]
antsnorm.inputs.metric=['MI', 'MI', 'CC']
antsnorm.inputs.metric_weight=[1.0]*3
antsnorm.inputs.number_of_iterations=[[1000, 500, 250, 100],[1000, 500, 250, 100],[100, 70, 50, 20]]
antsnorm.inputs.radius_or_number_of_bins=[32, 32, 4]
antsnorm.inputs.sampling_percentage=[0.25, 0.25, 1]
antsnorm.inputs.sampling_strategy=['Regular','Regular','None']
antsnorm.inputs.shrink_factors=[[8, 4, 2, 1]]*3
antsnorm.inputs.smoothing_sigmas=[[3, 2, 1, 0]]*3
antsnorm.inputs.transform_parameters=[(0.1,),(0.1,),(0.1, 3.0, 0.0)]
antsnorm.inputs.use_histogram_matching=True
antsnorm.inputs.write_composite_transform=True

#Normalise anatomical
apply2anat=Node(ants.ApplyTransforms(),name='apply2anat')
apply2anat.inputs.default_value=0
apply2anat.inputs.input_image_type=0
apply2anat.inputs.interpolation='Linear'
apply2anat.inputs.invert_transform_flags=[True,False]
apply2anat.inputs.num_threads=1
apply2anat.inputs.terminal_output='file'

#Normalise EPI
apply2epi=Node(ants.ApplyTransforms(),iterfield='input_image',name='apply2epi')
apply2epi.inputs.default_value=0
apply2epi.inputs.input_image_type=3
apply2epi.inputs.interpolation='Linear'
apply2epi.inputs.invert_transform_flags=[True,False]
apply2epi.inputs.num_threads=1
apply2epi.inputs.terminal_output='file'

#Normalise lesion mask
apply2lesionmask=Node(ants.ApplyTransforms(),name='apply2lesionmask')
apply2lesionmask.inputs.default_value=0
apply2lesionmask.inputs.input_image_type=0
apply2lesionmask.inputs.interpolation='Linear'
apply2lesionmask.inputs.invert_transform_flags=[True,False]
apply2lesionmask.inputs.num_threads=1
apply2lesionmask.inputs.terminal_output='file'

#Normalise matter mask
apply2mmask=Node(ants.ApplyTransforms(),iterfield='input_image',name='apply2mmask')
apply2mmask.inputs.default_value=0
apply2mmask.inputs.input_image_type=0
apply2mmask.inputs.interpolation='Linear'
apply2mmask.inputs.invert_transform_flags=[True,False]
apply2mmask.inputs.num_threads=1
apply2mmask.inputs.terminal_output='file'

#Normalise sanity check
apply2epiNC=Node(ants.ApplyTransforms(),name='apply2epiNC')
apply2epiNC.inputs.default_value=0
apply2epiNC.inputs.input_image_type=3
apply2epiNC.inputs.interpolation='Linear'
apply2epiNC.inputs.invert_transform_flags=[True,False]
apply2epiNC.inputs.num_threads=1
apply2epiNC.inputs.terminal_output='file'

#Apply transform to non-filtered EPIs (for FALFF ETC)
apply2epiNF=MapNode(ants.ApplyTransforms(),iterfield='input_image',name='apply2epiNF')
apply2epiNF.inputs.default_value=0
apply2epiNF.inputs.input_image_type=3
apply2epiNF.inputs.interpolation='Linear'
apply2epiNF.inputs.invert_transform_flags=[True,False]
apply2epiNF.inputs.num_threads=1
apply2epiNF.inputs.terminal_output='file'

#Datasink
substitutions=('_subject_id_', '')
sink=Node(DataSink(),name="sink")
sink.inputs.base_directory=outdir
sink.inputs.substitutions = substitutions

preproc=Workflow(name='stroke_preproc')
preproc.base_dir=workdir

###DATA IN MNI SPACE
                 ####POPULATE INPUTS, GET DATA, DROP EPI VOLS, GENERAL HOUSEKEEPING###
preproc.connect([(infosource,selectfiles,[('subject_id','subject_id')]),
                 (selectfiles,dropvols, [('epi','epilist')]),
                 (dropvols,epi_stack, [('epilist', 'dicom_files')]),
                 (epi_stack,metadata, [('out_file', 'nifti')]),
                 (epi_stack,despike, [('out_file', 'in_file')]),
                
                 ###HERE BE SLICE TIMING###
                 (metadata,st_corr, [('sliceno','num_slices'),
                                     ('slicetimes','slice_order'),
                                     ('tr','time_repetition'),
                                     ('ta','time_acquisition')]),
                 (despike,st_corr, [('out_file', 'in_files')]),             
                 
                 ###REALIGNMENT / TSNR / SEGMENTATION###
                 (st_corr,realign, [('timecorrected_files','in_files')]),                 
                 (realign,tsnr,[('realigned_files','in_file')]),
                 (selectfiles,anat_stack,[('anat','dicom_files')]),
                 (anat_stack,seg,[('out_file','channel_files')]),
                 

                 ###COREG FLAIR TO T1 START###
                 (anat_stack,flaircoreg,[('out_file','target')]),
                 (selectfiles,flaircoreg,[('flair','source')]),
                 (selectfiles,flaircoreg,[('mask','apply_to_files')]),

                 ###MAKE LIST OF DATA FOR FLAIR / T1 COREG TO EPI###
                 (flaircoreg,flair2anat_list,[('coregistered_source','flair')]),
                 (flaircoreg,flair2anat_list,[('coregistered_files','flairmask')]),
                 (seg,flair2anat_list,[('native_class_images','seg')]),

                 ###COREG TO EPI STARTS###             
                 (tsnr,coreg, [('mean_file','target')]),
                 (anat_stack,coreg,[('out_file','source')]),
                 (flair2anat_list,coreg,[('anat_coreglist','apply_to_files')]),

                 ###POPULATE COREG LISTS, MAKE MATTER MASKS
                 (coreg,anat2epi_list,[('coregistered_files','coreg_files'),
                                       ('coregistered_source','source')]),
                 (anat2epi_list,mmaskcalc,[('gm','gm'),
                                           ('wm','wm'),
                                           ('csf','csf')]),
                 (mmaskcalc,mmask_list,[('filename1','filename1'),
                                       ('filename2','filename2')]),
                                        
                 ###MOTION CORRECTION###
                 (realign,motion_correction, [('realignment_parameters','motionparams'),
                                              ('realigned_files', 'epifile')]),
                 
                 (mmaskcalc,motion_correction, [('filename1','mmaskfile')]),


                 ###COMPUTE TRANSFORM TO MNI###
                 (mmaskcalc,antsnorm,[('filename2','fixed_image')]),
                 (selectfiles,antsnorm,[('mni_template','moving_image')]),
                 (anat2epi_list,antsnorm,[('flairmask','fixed_image_mask')]),

                 ###POPULATE ANTS OUTPUT LIST###
                 (antsnorm, ants_list,[('reverse_transforms','ants_output')]),

                 ###APPLY TRANSFORM TO EPI###
                 (motion_correction,apply2epi,[('mc_filename','input_image')]),
                 (selectfiles,apply2epi,[('mni_template','reference_image')]),
                 (ants_list,apply2epi,[('trans','transforms')]),
                 
                 ###APPLY TRANSFORM TO MATTER MASK###
                 (mmaskcalc,apply2mmask,[('filename1','input_image')]),
                 (selectfiles,apply2mmask,[('mni_template','reference_image')]),
                 (ants_list,apply2mmask,[('trans','transforms')]),
                 
                                 
                 ###APPLY TRANSFORM TO T1 (test warp quality)###
                 (anat2epi_list,apply2anat,[('source','input_image')]),
                 (selectfiles,apply2anat,[('mni_template','reference_image')]),
                 (ants_list,apply2anat,[('trans','transforms')]),                 
                 

                 ###APPLY TRANSFORM LESION MASK###
                 (anat2epi_list,apply2lesionmask,[('flairmask','input_image')]),
                 (selectfiles,apply2lesionmask,[('mni_template','reference_image')]),
                 (ants_list,apply2lesionmask,[('trans','transforms')]),

                
###DATA IN MNI SPACE###
                 

                 ###CREATE MASKS FOR aCOMPCOR###
                 (selectfiles,noisemask,[('wm_noise','wm_in'),
                                       ('csf_noise','csf_in')]),
                 (apply2lesionmask,noisemask,[('output_image','mask_in')]),

                                          
                 ###CLEANING###
                 (noisemask,compcor_clean, [('wmmask_file','wmnoise_file'),
                                            ('csfmask_file','csfnoise_file')]),                 
                 (apply2mmask,compcor_clean, [('output_image','mmask_file')]),
                 (apply2epi,compcor_clean,[('output_image','epi_file')]),
                 
                 (compcor_clean,clean_list,[('file_global','file1'),
                                            ('file_noglobal','file2')]),
                 
                
                 ###FILTERING### 
                 (clean_list, bpfilter,[('cleaned','in_file')]),
                 (apply2mmask,bpfilter,[('output_image','brainmask')]),
            
                 
                 ###LIST FOR SMOOTHING###
                 (bpfilter,smooth_list,[('out_file','list1')]),
                 (clean_list,smooth_list,[('cleaned','list2')]),
                 (apply2epi,smooth_list,[('output_image','list3')]),

                 ###SMOOTH EPI###
                 (smooth_list,smooth,[('smoothing_files', 'in_files')]),
                 
                 ###COMPUTE AAL CORRELATION MATRIX###
                 (smooth,aal_corrmat,[('smoothed_files','smoothed_files')]),
                 
                 ###GRAB OUTPUTS###
                 (infosource,sink,[('subject_id','container')]),
                 (infosource,sink,[('subject_id','strip_dir')]),
                 (dropvols,sink,[('volsdropped_filename','QC.@vols')]),
                 (smooth,sink,[('smoothed_files','preproc_epis')]),
                 (realign,sink,[('realignment_parameters','QC')]),
                 (tsnr,sink,[('stddev_file','QC.@std'),
                             ('tsnr_file','QC.@tsnr')]),
                 (apply2anat,sink,[('output_image','mni_warped.@anat')]),
                 (apply2mmask,sink,[('output_image','mni_warped.@mmask')]),
                 (apply2lesionmask,sink,[('output_image','mni_warped.@lesionmask')]),
                 (compcor_clean,sink,[('wmdrop_filename','QC.@wmdrop'),
                                      ('csfdrop_filename','QC.@csfdrop'),
                                      ('var_filename_wm','QC.@variance_wm'),
                                      ('cumvar_filename_wm','QC.@cumvariance_wm'),
                                      ('var_filename_csf','QC.@variance_csf'),
                                      ('cumvar_filename_csf','QC.@cumvariance_csf'),
                                      ('regsglobal_filename','QC.@regsglobal'),
                                      ('regsnoglobal_filename','QC.@regsnoglobal')]),
                 (antsnorm,sink,[('forward_transforms','transforms.@forwardtrans'),
                                 ('warped_image','transforms.@warped'),
                                 ('inverse_warped_image','transforms.@invwarped')]),
                 (aal_corrmat,sink,[('file_global','corrmat.@global'),
                                    ('file_global_trans','corrmat.@global_trans'),
                                    ('file_noglobal','corrmat.@noglobal'),
                                    ('file_noglobal_trans','corrmat.@noglobal_trans'),
                                    ('file_noclean','corrmat.@noclean')]),

                 ])
                 
preproc.write_graph(graph2use='colored', format='svg', simple_form=False)
preproc.run(plugin='MultiProc',plugin_args={'n_procs': n_cores})

stop=time.time()-start
print "Time taken to complele analysis was " + str(stop) + " seconds"
