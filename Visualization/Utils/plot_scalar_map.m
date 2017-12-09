function plot_scalar_map(M, f)
    trisurf(M.TRIV,M.VERT(:,1),M.VERT(:,2),M.VERT(:,3),double(f));
    axis equal
    if isempty(strfind(version, '2010'))
        shading flat
    else
        shading interp
    end
    rotate3d on
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
end
