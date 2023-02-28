# postORGANIC

These are some scripts to run on the result of the image reconstructions from [Organic](https://github.com/kluskaj/ORGANIC).
They will make a first preprocessing of the cube of images dividing it onto 3 different sets.
It allows to filter out the non converged images and target the over-fitted ones.
It will also create fits files from the different sets of images as well as from the best images.
Finally, it will generate some plots to locate the images and sets agains f_data and f_rgl.
Enjoy!

## Installation

To install postOrganic it is quite simple.
1. Git clone this repository somewhere on your computer.
2. Type this commands:
```
cd postORGANIC
python setup.py install
```

That's it!
You will find a postORGANIC command in your terminal.
Please read the following to see how to use it

## Usage

In your terminal type the command:
```
postORGANIC Path/to/the/folder/with/image/reconstruction/from/ORGANIC/
```

Where ```Path/to/the/folder/with/image/reconstruction/from/ORGANIC/``` is the path to a folder where you have the output of the ORGANIC reconstructions.
