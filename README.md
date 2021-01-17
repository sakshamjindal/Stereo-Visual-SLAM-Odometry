# **Real-time Stereo Visual Odometry with Local Non-Linear Least Square Optimisation**

## **About**

This proect implements Stereo Visual Odometry using motion from 3D structure and Image correspondences

![Result on 00](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f12c500f-4a0b-4fe5-b5a6-4f51741c38ff/KITT_VO_00.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210117%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210117T223613Z&X-Amz-Expires=86400&X-Amz-Signature=ce065138fcf9c3ba2017dcaf1a1e9bd63c81f2ad2481d1668a9dfd266cd9ca5d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22KITT_VO_00.gif%22)

## **Installation**

```bash
$ conda env create -f setup/environment.yml
$ pip install -e .
```

## **Usage**

For simulation of visual odometry, run the followig command

```bash
# python main.py --config_path configs/params.yaml
```

The `params.yaml` needs to be edited to configure the sequence to run the simulation.

## **About the Problem !**

**1. What is Visual Odometry ?** <br>
**2. Formulation of the Problem**  <br>
**3. Algorith Implemented** <br>

### **Visual Odometry**

Visual Odometry is the process of incrementally estimating the pose and trajectory of a robot or a vehicle (orientation and translation of a camera configuration rigidly attached to it) using video stream coming from the camera.

### **Problem Formulation**

An agent is moving through an environment and taking
images with a rigidly attached camera system at discrete
time instants. Let the stream of images coming from the pair of camera
(assumed stere system) be denoted by I<sub>L, 0:n</sub> = {I<sub>L, 0</sub>,......, I<sub>L, n</sub>} and I<sub>R, 0:n</sub> = {I<sub>R, 0</sub>,......, I<sub>R, n</sub>}. We have prior knowledge of all the intrinsic as well as extrinsic calibration parameters of the stereo rig 

We need to estimate the relative rotation *R* and translation *t* between stereo configuration at time instants *k* and *k+1* and then to concatenate the transformation to incrementally recover the full trajectory C<sub>0:n</sub> of the camera

### **Algorithm Implemented**
Algorithm<sup>[1]</sup> : 3D-to-2D: Structure to feature correspondences** <br>
- &nbsp;Compute the first stereo image frames I<sub>L,K</sub> and I<sub>R,K</sub> <br>
- &nbsp;Extract and match stereo features f<sub>L,K</sub> and f<sub>R,K</sub> <br>
- &nbsp;Triangulate features to build point cloud X<sub>k</sub> <br>
- &nbsp;Set initial camera pose C<sub>k</sub> <br>
- &nbsp;Store information from the first frame as I<sub>L,k-1</sub>, I<sub>R,k-1</sub>, f<sub>L,k-1</sub>, f<sub>R,k-1</sub>, X<sub>k-1</sub> <br>
While Exists a new image frame <br>
    - Compute the new stereo image pair I<sub>L,K</sub> and <sub>R,K</sub> <br>
    - Extract and match stereo features f<sub>L,K</sub> and f<sub>R,K</sub> <br>
    - Triangulate features to build point cloud X<sub>k</sub> <br>
    - Track 2D features f<sub>L,k-1</sub> at I<sub>L,k-1</sub> to f<sub>L,k</sub> at I<sub>L,K</sub> and thus obtain <sup>t</sup>f<sub>k-1,k</sub> <br>
    - Compute correspondence for the tracked features <sup>t</sup>f<sub>k-1,k</sub> <br> and X<sub>k-1</sub> <br>
    - Compute camera pose estimation (P3P), thus T = [R|t]
    - Concatenate transformation by C<sub>k</sub> = C<sub>k-1</sub> T<sub>k</sub>
    - If Optimisation is enabled, do non-linear least squares optimisation of T 
    - Store informaton from first frame as I<sub>L,k-1</sub>, I<sub>R,k-1</sub>, f<sub>L,k-1</sub>, f<sub>R,k-1</sub>, X<sub>k-1</sub> and C<sub>k-1</sub> <br>

<p align="center"> 
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9af3a309-7fb5-4826-ab00-a035b9478d91/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210116%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210116T202235Z&X-Amz-Expires=86400&X-Amz-Signature=16cee78b2060f593b4ef30179fdcd62662ba99ee8fb9a72dfeb75aade2d8d849&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width="400" height="300" />  <br>
<p align="center" style="font-size:70%;"> <em> Relative Camera Pose and Concatenation of Transformations (Source: E. F. Aguilar Calzadillas [1] </em>) </p>
</p>


### **References**

[1]  ****E. F. Aguilar Calzadillas****, **"Sparse Stereo Visual Odometry with Local Non-Linear Least-Squares Optimization for Navigation of Autonomous Vehicles"**,  M. A. Sc. Thesis, Department of Mechanical and Aerospace Engineering, Carleton University, Ottawa ON, Canada, 2019
<br />

[2]  **D. Scaramuzza, F. Fraundorfer**, *"Visual Odometry: Part I - The First 30 Years and Fundamentals"*,  IEEE Robotics and Automation Magazine, Volume 18, issue 4, 2011
<br />

[3]  **F. Fraundorfer, D. Scaramuzza**, *"Visual odometry: Part II - Matching, robustness, optimization, and applications"*, IEEE Robotics and Automation Magazine, Volume 19, issue 2, 2012

[4] **Avi Singh**, *Visual Odmetry from scratch - A tutorial for beginners*, Avi Singh's Blog