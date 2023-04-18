import time
import numpy as np
from functools import wraps
import cv2
import os.path

def simplest_color_balance(img_msrcr,s1,s2):
    '''see section 3.1 in “Simplest Color Balance”(doi: 10.5201/ipol.2011.llmps-scb).
    Only suitable for 1-channel image'''
    sort_img=np.sort(img_msrcr,None)
    N=img_msrcr.size
    Vmin=sort_img[int(N*s1)]
    Vmax=sort_img[int(N*(1-s2))-1]
    img_msrcr[img_msrcr<Vmin]=Vmin
    img_msrcr[img_msrcr>Vmax]=Vmax
    return (img_msrcr-Vmin)*255/(Vmax-Vmin)
def get_gauss_kernel(sigma,dim=2):
    '''1D gaussian function: G(x)=1/(sqrt{2π}σ)exp{-(x-μ)²/2σ²}. Herein, μ:=0, after
       normalizing the 1D kernel, we can get 2D kernel version by
       matmul(1D_kernel',1D_kernel), having same sigma in both directions. Note that
       if you want to blur one image with a 2-D gaussian filter, you should separate
       it into two steps(i.e. separate the 2-D filter into two 1-D filter, one column
       filter, one row filter): 1) blur image with first column filter, 2) blur the
       result image of 1) with the second row filter. Analyse the time complexity: if
       m&n is the shape of image, p&q is the size of 2-D filter, bluring image with
       2-D filter takes O(mnpq), but two-step method takes O(pmn+qmn)'''
    ksize=int(np.floor(sigma*6)/2)*2+1 #kernel size("3-σ"法则) refer to
    #https://github.com/upcAutoLang/MSRCR-Restoration/blob/master/src/MSRCR.cpp
    k_1D=np.arange(ksize)-ksize//2
    k_1D=np.exp(-k_1D**2/(2*sigma**2))
    k_1D=k_1D/np.sum(k_1D)
    if dim==1:
        return k_1D
    elif dim==2:
        return k_1D[:,None].dot(k_1D.reshape(1,-1))
def gauss_blur(img,sigma):
    '''suitable for 1 or 3 channel image'''
    row_filter=get_gauss_kernel(sigma,1)
    t=cv2.filter2D(img,-1,row_filter[...,None])
    return cv2.filter2D(t,-1,row_filter.reshape(1,-1))

def MultiScaleRetinex(img,sigmas=[15,80,250],weights=None,flag=True):
    '''equal to func retinex_MSR, just remove the outer for-loop. Practice has proven
       that when MSR used in MSRCR or Gimp, we should add stretch step, otherwise the
       result color may be dim. But it's up to you, if you select to neglect stretch,
       set flag as False, have fun'''
    if weights==None:
        weights=np.ones(len(sigmas))/len(sigmas)
    elif not abs(sum(weights)-1)<0.00001:
        raise ValueError('sum of weights must be 1!')
    r=np.zeros(img.shape,dtype='double')
    img=img.astype('double')
    for i,sigma in enumerate(sigmas):
        r+=(np.log(img+1)-np.log(gauss_blur(img,sigma)+1))*weights[i]
    if flag:
        mmin=np.min(r,axis=(0,1),keepdims=True)
        mmax=np.max(r,axis=(0,1),keepdims=True)
        r=(r-mmin)/(mmax-mmin)*255 #maybe indispensable when used in MSRCR or Gimp, make pic vibrant
        r=r.astype('uint8')
    return r

def retinex_MSRCR(img,sigmas=[12,80,250],s1=0.01,s2=0.01):
    '''r=βlog(αI')MSR, I'=I/∑I, I is one channel of image, ∑I is the sum of all channels,
       C:=βlog(αI') is named as color recovery factor. Last we improve previously used
       linear stretch: MSRCR:=r, r=G[MSRCR-b], then doing linear stretch. In practice, it
       doesn't work well, so we take another measure: Simplest Color Balance'''
    alpha=125
    img=img.astype('double')+1 #
    csum_log=np.log(np.sum(img,axis=2))
    msr=MultiScaleRetinex(img-1,sigmas) #-1
    r=(np.log(alpha*img)-csum_log[...,None])*msr
    #beta=46;G=192;b=-30;r=G*(beta*r-b) #deprecated
    #mmin,mmax=np.min(r),np.max(r)
    #stretch=(r-mmin)/(mmax-mmin)*255 #linear stretch is unsatisfactory
    for i in range(r.shape[-1]):
        r[...,i]=simplest_color_balance(r[...,i],0.01,0.01)
    return r.astype('uint8')

if __name__ == '__main__':
    path = 'image/'