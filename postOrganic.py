#!/usr/bin/env python

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from astropy.io import fits
import sys
import matplotlib.pyplot as plt

class cube:
    def __init__(self, dir, file = 'ImageCube.fits', nPCA=3, nKmeans=3):
        self.dir = dir
        self.file = file
        self.nKmeans = nKmeans
        self.nPCA = nPCA
        self.read()
        self.doPCA()
        self.doKmeans()
        #self.plotPCA()
        self.plotKmeans()
        self.plotImages()

    def read(self):
        hdul = fits.open(self.dir + self.file)
        cube = hdul[0].data
        self.ps = hdul[0].header['CDELT2']
        self.n = hdul[0].header['NAXIS2']
        self.cube = cube

    def doPCA(self):
        cube = self.cube
        shape = cube.shape
        cube = np.reshape(cube, [shape[0], shape[1]*shape[2]])
        pca = PCA(n_components = self.nPCA)
        cube_pca = pca.fit(cube).transform(cube)
        self.pca = cube_pca

    def doKmeans(self):
        pca = self.pca
        kmeans = KMeans(n_clusters=self.nKmeans, n_init=500)
        clusters = kmeans.fit_predict(pca)
        centers = kmeans.cluster_centers_
        self.kmeans = clusters
        self.centers = centers

    def plotKmeans(self):
        fig, ax = plt.subplots()
        plt.scatter(self.pca[:,0], self.pca[:,1], c=self.kmeans)
        centroids = self.centers
        for centre, k in zip(centroids,np.arange(self.nKmeans)):
            plt.text(centre[0], centre[1], s=f"{k}", size='large', backgroundcolor='lightgray', alpha=0.6)
        plt.show()
        plt.close()

    def plotPCA(self):
        fig, ax = plt.subplots()
        plt.scatter(self.pca[:,0], self.pca[:,1])
        plt.show()
        plt.close()

    def groupImages(self):
        idx = list(self.kmeans)
        cube = self.cube
        medians, stds = [], []
        counts, groups = [], []
        for k in np.arange(self.nKmeans):
            counts.append(idx.count(k))
            cub = []
            for image, i in zip(cube, idx):
                if i == k:
                    cub.append(image)
            cub = np.array(cub)
            groups.append(cub)
            medians.append(np.median(cub, axis=0))
            stds.append(np.std(cub, axis=0))
        self.medians = medians
        self.stds = stds
        self.counts = counts
        self.groups = groups

    def plotImages(self):
        self.groupImages()
        images = np.array(self.medians)
        fig, axs = plt.subplots(ncols=self.nKmeans, sharey=True)
        for ax,i in zip(axs,np.arange(self.nKmeans)):
            ax.imshow(images[i,::-1,::-1], cmap='inferno')
            ax.text(20, 20, f"{i}: {self.counts[i]} images", color='white', size=7)
        plt.show()
        plt.close()

    def plotSet(self, set):
        images = self.groups[set]
        number = self.counts[set]

        fig, axs = plt.subplots(ncols=5,nrows=3)
        i = -1
        for axy in axs:
            for ax in axy:
                i += 1
                if i < number-1:
                    ax.imshow(images[i,::-1,::-1], cmap='inferno')
        plt.tight_layout
        plt.show()
        plt.close()

    def radial_profiles(self, R=20, set=-1):
        if set == -1:
            cube = self.cube
            image = np.median(cube, axis=0)
            std = np.std(cube, axis=0)
        else:
            image = np.array(self.medians)[set,:,:]
            std = np.array(self.stds)[set,:,:]

        self.set_coord()
        self.compute_profile(R, image, std)
        self.plot_profile()
        #self.plot_std_profile(R, image, std)

    def set_coord(self):
        n = self.n
        xr = np.arange(n)+1
        x =  - (xr-self.n/2)*self.ps
        y =  - (xr-self.n/2)*self.ps
        d = n*self.ps/2.
        x = np.linspace(d, -d, n)
        y = np.linspace(-d, d, n)
        x2, y2 = np.meshgrid(x, y)
        R = np.sqrt(x2**2+y2**2)
        angle = np.arctan2(y2, x2)

        # fig,ax = plt.subplots()
        # plt.imshow(y2)
        # plt.show()
        # plt.close()
        self.d = d
        self.x = x2
        self.y = -y2
        self.R = R
        self.angle = angle

    def compute_profile(self, R, image, noise):
        #noise /= np.max(image)
        #image /= np.max(image)
        dr = (self.d - self.ps) / R
        radii_m = np.linspace(self.ps, self.d-dr, R)
        radii_p = np.linspace(self.ps+dr, self.d, R)
        self.radii = (radii_m + radii_p)/2.
        profile, noises, rms = [], [], []
        for rad_m, rad_p in zip(radii_m, radii_p):
            mask = (self.R >= rad_m) * (self.R < rad_p)
            reduced_image = image[mask!=0]
            reduced_noise = noise[mask!=0]
            fluxi = np.average(reduced_image)
            noisei = np.average(reduced_noise)
            rmsi = np.std(reduced_image ) #/len(reduced_image)
            profile.append(fluxi)
            noises.append(noisei)
            rms.append(rmsi)

        self.profile = np.array(profile)
        self.noises = np.array(noises)
        self.rms = np.array(rms)

    def plot_profile(self):

        fig, ax = plt.subplots()
        ax.plot(self.radii, self.profile, label = 'profile', color='black')
        ax.plot(self.radii, self.noises, label = 'noise', color='blue')
        ax.plot(self.radii, self.rms, label = 'azim. std.', color='red')
        plt.yscale('log')
        plt.legend()
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.radii, self.profile/self.noises, label = 'SNR-noise', color='blue')
        ax.plot(self.radii, self.profile/self.rms, label = 'SNR-std.', color='red')
        #plt.ylim(0,5)
        plt.legend()
        plt.show()
        plt.close()

def main():
    arg = sys.argv[1:]
    Cube0 = cube(arg[0], nPCA = 10, nKmeans=3)
    Cube0.radial_profiles(R=100)
    Cube0.radial_profiles(R=100,set=0)
    Cube0.radial_profiles(R=100,set=1)
    Cube0.radial_profiles(R=100,set=2)
    #Cube0.plotSet(0)

if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv)
