/*
 * Emanuele Rodola
 * TU Munich, 23/09/2014
 * <rodola@in.tum.de>
 */
 
// mex -v calc_shot.cpp shot_descriptor.cpp -DUSE_FLANN -I"C:\Program Files\flann\include" -I../Eigen

#include "mex.h"
#include <vector>
#include "shot_descriptor.h"

//DEBUG
#include <fstream>
#include <iostream>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
	if (nrhs != 6 || nlhs != 1)
      mexErrMsgTxt("Usage: descr = calc_shot(vertices, faces, indices, n_bins, radius, min_neighs).");
	  
	// Parse input

	const double* const vertices = mxGetPr(prhs[0]);
	const int nv = int( mxGetN(prhs[0]) );
	
	if (int( mxGetM(prhs[0]) ) != 3)
		mexErrMsgTxt("Vertices should be given as a 3xN matrix.");
		
	const double* const faces = mxGetPr(prhs[1]);
	const int nt = int( mxGetN(prhs[1]) );
	
	if (int( mxGetM(prhs[1]) ) != 3)
		mexErrMsgTxt("Faces should be given as a 3xM matrix.");
	
	// SHOT descriptors will be computed only for these points (1-based indices)
	const double* const idx = mxGetPr(prhs[2]);
	const int np = std::max( int( mxGetM(prhs[2]) ) , int( mxGetN(prhs[2]) ) );
	
	// Create mesh structure
	
	mesh_t mesh;
	{
		std::vector< vec3d<double> > V(nv);
		
		for (int i=0; i<nv; ++i)
		{
			vec3d<double>& p = V[i];
			p.x = vertices[i*3];
			p.y = vertices[i*3+1];
			p.z = vertices[i*3+2];
			//std::cout << p.x << " " << p.y << " " << p.z << std::endl;
		}
		
		mesh.put_vertices(V);
		
		for (int i=0; i<nt; ++i)
			mesh.add_triangle( faces[i*3]-1 , faces[i*3+1]-1 , faces[i*3+2]-1 );
	}
	
	mesh.calc_normals();
	
	// Compute SHOT descriptors at the desired point indices
	
	unibo::SHOTParams params;
	params.radius = *mxGetPr(prhs[4]);
	params.localRFradius = params.radius;
	params.minNeighbors = *mxGetPr(prhs[5]);
	params.bins = *mxGetPr(prhs[3]);
	
	unibo::SHOTDescriptor sd(params);
	const size_t sz = sd.getDescriptorLength();
	
	plhs[0] = mxCreateDoubleMatrix(sz, np, mxREAL);
	double* D = mxGetPr(plhs[0]);
	
	std::cout << "Computing SHOTs on " << np << " points... " << std::flush;

	for (size_t i=0; i<np; ++i)
	{
		unibo::shot s;
		sd.describe(mesh, static_cast<int>(idx[i]-1), s);
		for (size_t j=0; j<sz; ++j)
			D[i*sz+j] = s(j);
	}
	
	std::cout << "done." << std::endl;
}
