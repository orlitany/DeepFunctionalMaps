function showshape(shape,h,viewpoint)
   
    if isfield(shape,'VERT'),
        X = [shape.VERT(:,1) shape.VERT(:,2) shape.VERT(:,3)];
    else
        X = [shape.X(:) shape.Y(:) shape.Z(:)];                     % Vertices
    end
    
    F = shape.TRIV;                                             % Faces
    nv = size(X,1);                                             % # of vertices
    nf = size(F,1);                                             % # of faces
    
    if (nargin==1)
        h=zeros(nv,1);
    end
    
%      viewpoint = [180 -90];
%     figure;
    trisurf(F, X(:,1), X(:,2), X(:,3), h); 
    if exist('viewpoint','var')
       view(viewpoint); 
    end
%      view(viewpoint);
    axis image; 
    axis off;
    shading flat;
    lighting phong; 
    camlight head;
%     title('Original shape');

end
