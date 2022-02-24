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
            medians.append(np.mean(cub, axis=0))
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

def main():
    arg = sys.argv[1:]
    Cube0 = cube(arg[0], nPCA = 3, nKmeans=2)
    #Cube0.plotSet(0)

if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv)
